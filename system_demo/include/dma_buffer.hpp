// Linux DMA-BUF 与 RGA 的资源封装。FD 用于设备共享，mmap 地址用于 CPU 访问。
// 共享同一物理缓冲不等于无需缓存同步；CPU 读写必须显式使用 CpuAccess。

#pragma once
#include <atomic>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cerrno>
#include <fcntl.h>
#include <linux/dma-buf.h>
#include <linux/dma-heap.h>
#include <linux/videodev2.h>
#include <memory>
#include <mutex>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#include <im2d.h>
#include <rga/im2d_buffer.h>

namespace camera_dma {
// 系统调用被信号中断（EINTR）时重试，其他错误交给调用方判断。
inline int Ioctl(int fd, unsigned long request, void* arg) {
  int ret;
  // 先执行一次系统调用；只有被信号中断才重试，设备错误等其他失败交给调用方报告。
  do {
    ret = ioctl(fd, request, arg);
  } while (ret < 0 && errno == EINTR);
  return ret;
}

inline void Check(bool ok, const std::string& operation) {
  if (!ok) {
    throw std::runtime_error(operation + ": " + std::strerror(errno));
  }
}

// 独占管理一个 DMA heap 分配和映射；禁止复制，防止重复 close/munmap。
class DmaBuffer {
 public:
  DmaBuffer(const std::string& heap, size_t bytes) : size_(bytes) {
    int heap_fd = open(heap.c_str(), O_RDWR | O_CLOEXEC);
    Check(heap_fd >= 0, "open DMA heap " + heap);
    dma_heap_allocation_data allocation{};
    allocation.len = bytes;
    allocation.fd_flags = O_RDWR | O_CLOEXEC;
    const int ret = Ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &allocation);
    // 关闭临时 heap FD 可能改变 errno，先保存分配调用的错误以便准确报告。
    const int saved_errno = errno;
    close(heap_fd);
    errno = saved_errno;
    Check(ret == 0, "DMA_HEAP_IOCTL_ALLOC");
    fd_ = static_cast<int>(allocation.fd);
    address_ = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (address_ == MAP_FAILED) {
      close(fd_);
      fd_ = -1;
      throw std::runtime_error("mmap DMA buffer failed");
    }
  }

  ~DmaBuffer() {
    if (address_ != MAP_FAILED) {
      munmap(address_, size_);
    }
    if (fd_ >= 0) {
      close(fd_);
    }
  }

  DmaBuffer(const DmaBuffer&) = delete;
  DmaBuffer& operator=(const DmaBuffer&) = delete;

  int Fd() const {
    return fd_;
  }

  void* Data() const {
    return address_;
  }

  size_t Size() const {
    return size_;
  }

  void Sync(uint64_t flags) const {
    dma_buf_sync sync{};
    sync.flags = flags;
    Check(Ioctl(fd_, DMA_BUF_IOCTL_SYNC, &sync) == 0, "DMA_BUF_IOCTL_SYNC");
  }

 private:
  int fd_ = -1;
  void* address_ = MAP_FAILED;
  size_t size_;
};

// Scope device/CPU ownership explicitly, including cache synchronization.
// 作用域内声明 CPU 的读/写方向。正常路径 Finish 可抛出同步错误，
// 析构只做兜底收尾并吞掉异常，避免栈展开时二次抛错。
class CpuAccess {
 public:
  CpuAccess(const CpuAccess&) = delete;
  CpuAccess& operator=(const CpuAccess&) = delete;

  CpuAccess(const DmaBuffer& buffer, uint64_t direction) : buffer_(buffer), direction_(direction) {
    buffer_.Sync(DMA_BUF_SYNC_START | direction_);
  }

  ~CpuAccess() {
    if (active_) {
      try {
        buffer_.Sync(DMA_BUF_SYNC_END | direction_);
      }
      catch (...) {
      }
    }
  }

  // 显式收尾用于正常路径：同步失败可向调用者报告；成功后标记完成，析构不再重复同步。
  void Finish() {
    buffer_.Sync(DMA_BUF_SYNC_END | direction_);
    active_ = false;
  }

 private:
  const DmaBuffer& buffer_;
  uint64_t direction_;
  bool active_ = true;
};

// 在 DMA 缓冲上注册 RGA 图像；stride 以像素为单位按 16 对齐，RowBytes 转为字节。
// 逻辑宽度与物理行宽不同，CPU 归一化必须按 RowBytes 逐行读取。
class DmaImage {
 public:
  DmaImage(const std::string& heap, int width, int height, int format)
      : width_(width),
        height_(height),
        stride_((width + 15) & ~15),
        buffer_(heap, static_cast<size_t>(stride_) * height * 3) {
    handle_ = importbuffer_fd(buffer_.Fd(), static_cast<int>(buffer_.Size()));
    if (!handle_) {
      throw std::runtime_error("RGA importbuffer_fd failed for output");
    }
    image_ = wrapbuffer_handle(handle_, width, height, format, stride_, height);
  }

  ~DmaImage() {
    if (handle_) {
      releasebuffer_handle(handle_);
    }
  }

  DmaImage(const DmaImage&) = delete;
  DmaImage& operator=(const DmaImage&) = delete;

  rga_buffer_t Image() const {
    return image_;
  }

  DmaBuffer& Buffer() {
    return buffer_;
  }

  int Width() const {
    return width_;
  }

  int Height() const {
    return height_;
  }

  size_t RowBytes() const {
    return static_cast<size_t>(stride_) * 3;
  }

 private:
  int width_, height_, stride_;
  DmaBuffer buffer_;
  rga_buffer_handle_t handle_ = 0;
  rga_buffer_t image_{};
};

inline void RgaCheck(IM_STATUS status, const char* operation) {
  if (status != IM_STATUS_SUCCESS && status != IM_STATUS_NOERROR) {
    throw std::runtime_error(std::string(operation) + ": " + imStrError(status));
  }
}

// 同步 RGA 操作完成后才返回，调用者可据此决定何时释放输入 Frame。
inline void Blit(rga_buffer_t src, rga_buffer_t dst, im_rect source, im_rect destination) {
  // Synchronous completion makes the capture buffer safe to requeue afterwards.
  RgaCheck(improcess(src, dst, {}, source, destination, {}, IM_SYNC), "RGA improcess");
}

}  // namespace camera_dma
