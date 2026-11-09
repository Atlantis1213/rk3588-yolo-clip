// 早期单文件版本的 Linux 摄像头 DMA/RGA 实现，供根目录历史 benchmark 使用。
// 当前主工程已拆分到 system_demo/include/，阅读和后续维护优先从该目录开始。

#pragma once

// Linux-only V4L2 capture and DMA-BUF/RGA helpers. No FFmpeg is used here.
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

inline int Ioctl(int fd, unsigned long request, void* arg) {
  int ret;
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

// 独占 DMA heap FD 和 CPU 映射，析构按顺序解除映射、关闭 FD，避免句柄泄漏。
class DmaBuffer {
 public:
  DmaBuffer(const std::string& heap, size_t bytes) : size_(bytes) {
    int heap_fd = open(heap.c_str(), O_RDWR | O_CLOEXEC);
    Check(heap_fd >= 0, "open DMA heap " + heap);
    dma_heap_allocation_data allocation{};
    allocation.len = bytes;
    allocation.fd_flags = O_RDWR | O_CLOEXEC;
    const int ret = Ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &allocation);
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
// CPU 缓存同步的作用域：START 获取访问权，END 结束访问；DMA 共享不能省略同步。
class CpuAccess {
 public:
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

  void Finish() {
    buffer_.Sync(DMA_BUF_SYNC_END | direction_);
    active_ = false;
  }

 private:
  const DmaBuffer& buffer_;
  uint64_t direction_;
  bool active_ = true;
};

// 把 DMA 内存导入 RGA，逻辑宽度与按 16 对齐的物理 stride 分开记录。
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

inline void Blit(rga_buffer_t src, rga_buffer_t dst, im_rect source, im_rect destination) {
  // Synchronous completion makes the capture buffer safe to requeue afterwards.
  RgaCheck(improcess(src, dst, {}, source, destination, {}, IM_SYNC), "RGA improcess");
}

class V4l2Capture;

// 摄像头缓冲的生命周期凭证；最后一个引用销毁时重新入队，驱动才可覆写像素。
struct Frame {
  std::shared_ptr<V4l2Capture> owner;
  unsigned int index = 0;
  uint32_t sequence = 0;
  rga_buffer_t image{};
  ~Frame();
};

// 初始化协商格式并导出缓冲；运行中 Acquire 出队，Frame 析构归还。
class V4l2Capture : public std::enable_shared_from_this<V4l2Capture> {
 public:
  static std::shared_ptr<V4l2Capture> Open(const std::string& device,
                                           int width,
                                           int height,
                                           const std::string& format,
                                           int buffers,
                                           double fps,
                                           int minimum) {
    auto capture = std::shared_ptr<V4l2Capture>(new V4l2Capture());
    capture->Initialize(device, width, height, format, buffers, fps, minimum);
    return capture;
  }

  ~V4l2Capture() {
    if (streaming_) {
      v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      Ioctl(fd_, VIDIOC_STREAMOFF, &type);
    }
    for (auto& buffer : buffers_) {
      if (buffer.handle) {
        releasebuffer_handle(buffer.handle);
      }
      if (buffer.dma_fd >= 0) {
        close(buffer.dma_fd);
      }
    }
    if (fd_ >= 0) {
      close(fd_);
    }
  }

  int Width() const {
    return width_;
  }

  int Height() const {
    return height_;
  }

  const std::string& Description() const {
    return description_;
  }

  void Start() {
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    Check(Ioctl(fd_, VIDIOC_STREAMON, &type) == 0, "VIDIOC_STREAMON");
    streaming_ = true;
  }

  // nullptr means a bounded timeout or a corrupt frame, not end-of-stream.
  std::shared_ptr<Frame> Acquire(int timeout_ms = 200) {
    if (requeue_failed_.load()) {
      throw std::runtime_error("V4L2 buffer requeue failed");
    }
    pollfd descriptor{fd_, POLLIN, 0};
    int ret;
    do {
      ret = poll(&descriptor, 1, timeout_ms);
    } while (ret < 0 && errno == EINTR);
    Check(ret >= 0, "V4L2 poll");
    if (ret == 0) {
      return nullptr;
    }
    if (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) {
      throw std::runtime_error("V4L2 poll: camera unavailable");
    }
    v4l2_buffer buffer{};
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = V4L2_MEMORY_MMAP;
    std::lock_guard<std::mutex> lock(io_mutex_);
    if (Ioctl(fd_, VIDIOC_DQBUF, &buffer) < 0) {
      if (errno == EAGAIN) {
        return nullptr;
      }
      Check(false, "VIDIOC_DQBUF");
    }
    if (buffer.index >= buffers_.size()) {
      throw std::runtime_error("V4L2 returned invalid buffer index");
    }
    if ((buffer.flags & V4L2_BUF_FLAG_ERROR) || buffer.bytesused < required_bytes_) {
      Check(Ioctl(fd_, VIDIOC_QBUF, &buffer) == 0, "VIDIOC_QBUF corrupt frame");
      return nullptr;
    }
    auto frame = std::make_shared<Frame>();
    frame->owner = shared_from_this();
    frame->index = buffer.index;
    frame->sequence = buffer.sequence;
    frame->image = buffers_[buffer.index].image;
    return frame;
  }

  void Requeue(unsigned int index) noexcept {
    std::lock_guard<std::mutex> lock(io_mutex_);
    v4l2_buffer buffer{};
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = V4L2_MEMORY_MMAP;
    buffer.index = index;
    if (Ioctl(fd_, VIDIOC_QBUF, &buffer) < 0) {
      requeue_failed_.store(true);
    }
  }

 private:
  struct Buffer {
    int dma_fd = -1;
    rga_buffer_handle_t handle = 0;
    rga_buffer_t image{};
  };

  V4l2Capture() = default;

  void Initialize(const std::string& device,
                  int width,
                  int height,
                  const std::string& format,
                  int count,
                  double fps,
                  int minimum) {
    fd_ = open(device.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
    Check(fd_ >= 0, "open camera " + device);
    v4l2_capability capability{};
    Check(Ioctl(fd_, VIDIOC_QUERYCAP, &capability) == 0, "VIDIOC_QUERYCAP");
    uint32_t caps = capability.capabilities;
    if (caps & V4L2_CAP_DEVICE_CAPS) {
      caps = capability.device_caps;
    }
    if (!(caps & V4L2_CAP_VIDEO_CAPTURE) || !(caps & V4L2_CAP_STREAMING)) {
      throw std::runtime_error("camera must support single-plane V4L2 streaming capture");
    }
    const uint32_t fourcc = format == "YUYV" ? V4L2_PIX_FMT_YUYV : V4L2_PIX_FMT_NV12;
    v4l2_format config{};
    config.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    config.fmt.pix.width = width;
    config.fmt.pix.height = height;
    config.fmt.pix.pixelformat = fourcc;
    config.fmt.pix.field = V4L2_FIELD_NONE;
    Check(Ioctl(fd_, VIDIOC_S_FMT, &config) == 0, "VIDIOC_S_FMT");
    const auto& pixels = config.fmt.pix;
    if (pixels.pixelformat != fourcc || pixels.field != V4L2_FIELD_NONE) {
      throw std::runtime_error(
          "camera did not accept progressive YUYV/NV12; check v4l2-ctl formats");
    }
    width_ = static_cast<int>(pixels.width);
    height_ = static_cast<int>(pixels.height);
    if (width_ <= 0 || height_ <= 0 || width_ > 8192 || height_ > 8192) {
      throw std::runtime_error("camera dimensions outside supported range");
    }
    if (width_ % 2 || (format == "NV12" && height_ % 2)) {
      throw std::runtime_error("camera YUV dimensions must satisfy chroma alignment");
    }
    const unsigned int pixel_bytes = format == "YUYV" ? 2 : 1;
    const unsigned int row_bytes = pixels.bytesperline ? pixels.bytesperline : width_ * pixel_bytes;
    if (row_bytes < width_ * pixel_bytes || row_bytes % pixel_bytes) {
      throw std::runtime_error("unsupported V4L2 bytesperline");
    }
    required_bytes_ = static_cast<size_t>(row_bytes) * height_;
    if (format == "NV12") {
      required_bytes_ += static_cast<size_t>(row_bytes) * height_ / 2;
    }
    if (pixels.sizeimage < required_bytes_) {
      throw std::runtime_error("V4L2 sizeimage too small for negotiated layout");
    }
    v4l2_streamparm parameters{};
    parameters.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parameters.parm.capture.timeperframe.numerator = 1000;
    parameters.parm.capture.timeperframe.denominator = static_cast<unsigned int>(fps * 1000);
    // Some drivers cannot set FPS. Capture still follows the device's negotiated FPS.
    if (Ioctl(fd_, VIDIOC_S_PARM, &parameters) < 0 && errno != EINVAL && errno != ENOTTY) {
      Check(false, "VIDIOC_S_PARM");
    }
    v4l2_requestbuffers request{};
    request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    request.memory = V4L2_MEMORY_MMAP;
    request.count = count;
    Check(Ioctl(fd_, VIDIOC_REQBUFS, &request) == 0, "VIDIOC_REQBUFS");
    if (request.count < static_cast<unsigned int>(minimum)) {
      throw std::runtime_error("too few camera buffers; reduce --frame-queue / --yolo-threads");
    }
    buffers_.resize(request.count);
    for (unsigned int i = 0; i < request.count; ++i) {
      v4l2_buffer buffer{};
      buffer.type = request.type;
      buffer.memory = request.memory;
      buffer.index = i;
      Check(Ioctl(fd_, VIDIOC_QUERYBUF, &buffer) == 0, "VIDIOC_QUERYBUF");
      if (buffer.length < required_bytes_) {
        throw std::runtime_error("V4L2 buffer is undersized");
      }
      v4l2_exportbuffer exported{};
      exported.type = request.type;
      exported.index = i;
      exported.flags = O_RDWR | O_CLOEXEC;
      Check(Ioctl(fd_, VIDIOC_EXPBUF, &exported) == 0,
            "VIDIOC_EXPBUF (camera driver must support DMA-BUF export)");
      auto& slot = buffers_[i];
      slot.dma_fd = exported.fd;
      slot.handle = importbuffer_fd(slot.dma_fd, static_cast<int>(buffer.length));
      if (!slot.handle) {
        throw std::runtime_error("RGA cannot import camera DMA-BUF");
      }
      slot.image = wrapbuffer_handle(slot.handle,
                                     width_,
                                     height_,
                                     format == "YUYV" ? RK_FORMAT_YUYV_422 : RK_FORMAT_YCbCr_420_SP,
                                     row_bytes / pixel_bytes,
                                     height_);
      // Explicit YUV colorimetry; the full/limited range is negotiated by V4L2.
      const unsigned int encoding =
          pixels.ycbcr_enc == V4L2_YCBCR_ENC_DEFAULT
              ? static_cast<unsigned int>(V4L2_MAP_YCBCR_ENC_DEFAULT(pixels.colorspace))
              : pixels.ycbcr_enc;
      if (encoding != V4L2_YCBCR_ENC_601 && encoding != V4L2_YCBCR_ENC_709) {
        throw std::runtime_error("camera YUV colorimetry must be BT.601 or BT.709");
      }
      const unsigned int quantization =
          pixels.quantization == V4L2_QUANTIZATION_DEFAULT
              ? static_cast<unsigned int>(
                    V4L2_MAP_QUANTIZATION_DEFAULT(false, pixels.colorspace, encoding))
              : pixels.quantization;
      const bool full = quantization == V4L2_QUANTIZATION_FULL_RANGE;
      const bool bt709 = encoding == V4L2_YCBCR_ENC_709;
      if (bt709 && full) {
        throw std::runtime_error("installed RGA API cannot express BT.709 full-range conversion");
      }
      slot.image.color_space_mode =
          bt709 ? IM_YUV_TO_RGB_BT709_LIMIT
                : (full ? IM_YUV_TO_RGB_BT601_FULL : IM_YUV_TO_RGB_BT601_LIMIT);
      Check(Ioctl(fd_, VIDIOC_QBUF, &buffer) == 0, "VIDIOC_QBUF initial");
    }
    description_ = device + " " + format + " " + std::to_string(width_) + "x" +
                   std::to_string(height_) + " DMA-BUF buffers=" + std::to_string(request.count);
  }

  int fd_ = -1, width_ = 0, height_ = 0;
  size_t required_bytes_ = 0;
  bool streaming_ = false;
  std::mutex io_mutex_;
  std::atomic<bool> requeue_failed_{false};
  std::vector<Buffer> buffers_;
  std::string description_;
};

inline Frame::~Frame() {
  if (owner) {
    owner->Requeue(index);
  }
}

// One instance per YOLO worker; all images are imported once and reused.
// 复用整图/方形/ROI 缓冲；YOLO 左上角对齐补边，CLIP 直接将 ROI 缩放到 224。
class Preprocessor {
 public:
  explicit Preprocessor(std::string heap) : heap_(std::move(heap)) {
  }

  void Convert(const Frame& frame) {
    const int width = frame.owner->Width(), height = frame.owner->Height();
    ConvertRaw(frame.image, width, height);
  }

  void ConvertRaw(rga_buffer_t source, int width, int height) {
    Ensure(full_, width, height, RK_FORMAT_BGR_888);
    Blit(source, full_->Image(), {0, 0, width, height}, {0, 0, width, height});
  }

  DmaImage& YoloRgb() {
    const int width = full_->Width(), height = full_->Height();
    const int side = std::max(width, height);
    Ensure(square_, side, side, RK_FORMAT_BGR_888);
    Ensure(yolo_, 640, 640, RK_FORMAT_RGB_888);
    // This board's RGA driver rejects color-fill jobs. Initialize constant padding
    // once per source geometry instead; each frame overwrites only the image area.
    if (padding_width_ != width || padding_height_ != height) {
      CpuAccess write(square_->Buffer(), DMA_BUF_SYNC_WRITE);
      std::memset(square_->Buffer().Data(), 114, square_->Buffer().Size());
      write.Finish();
      padding_width_ = width;
      padding_height_ = height;
    }
    Blit(full_->Image(), square_->Image(), {0, 0, width, height}, {0, 0, width, height});
    Blit(square_->Image(), yolo_->Image(), {0, 0, side, side}, {0, 0, 640, 640});
    return *yolo_;
  }

  DmaImage& ClipRgb(int x, int y, int width, int height) {
    Ensure(clip_, 224, 224, RK_FORMAT_RGB_888);
    Blit(full_->Image(), clip_->Image(), {x, y, width, height}, {0, 0, 224, 224});
    return *clip_;
  }

 private:
  void Ensure(std::unique_ptr<DmaImage>& image, int width, int height, int format) {
    if (!image || image->Width() != width || image->Height() != height) {
      image = std::make_unique<DmaImage>(heap_, width, height, format);
    }
  }

  std::string heap_;
  int padding_width_ = 0, padding_height_ = 0;
  std::unique_ptr<DmaImage> full_, square_, yolo_, clip_;
};

// Exercise real DMA heap allocation, FD import, color conversion, resize and CPU cache sync.
inline void SelfTest(const std::string& heap) {
  DmaBuffer yuyv(heap, 640 * 480 * 2);
  {
    CpuAccess write(yuyv, DMA_BUF_SYNC_WRITE);
    auto* pixels = static_cast<unsigned char*>(yuyv.Data());
    for (int i = 0; i < 640 * 480 * 2; i += 4) {
      pixels[i] = 82;
      pixels[i + 1] = 90;
      pixels[i + 2] = 82;
      pixels[i + 3] = 240;
    }
    write.Finish();
  }
  rga_buffer_handle_t handle = importbuffer_fd(yuyv.Fd(), static_cast<int>(yuyv.Size()));
  if (!handle) {
    throw std::runtime_error("self-test input FD import failed");
  }
  try {
    auto source = wrapbuffer_handle(handle, 640, 480, RK_FORMAT_YUYV_422);
    source.color_space_mode = IM_YUV_TO_RGB_BT601_LIMIT;
    DmaImage rgb(heap, 224, 224, RK_FORMAT_RGB_888);
    Blit(source, rgb.Image(), {0, 0, 640, 480}, {0, 0, 224, 224});
    CpuAccess read(rgb.Buffer(), DMA_BUF_SYNC_READ);
    const auto* pixel = static_cast<const unsigned char*>(rgb.Buffer().Data());
    if (pixel[0] < 240 || pixel[1] > 15 || pixel[2] > 15) {
      throw std::runtime_error("DMA/RGA self-test color mismatch");
    }
    read.Finish();
    Preprocessor prep(heap);
    prep.ConvertRaw(source, 640, 480);
    auto& padded = prep.YoloRgb();
    {
      CpuAccess access(padded.Buffer(), DMA_BUF_SYNC_READ);
      const auto* base = static_cast<const unsigned char*>(padded.Buffer().Data());
      const auto* padding = base + 600 * padded.RowBytes() + 320 * 3;
      if (base[0] < 240 || base[1] > 15 || base[2] > 15 ||
          std::abs(static_cast<int>(padding[0]) - 114) > 2 ||
          std::abs(static_cast<int>(padding[1]) - 114) > 2 ||
          std::abs(static_cast<int>(padding[2]) - 114) > 2) {
        throw std::runtime_error("DMA/RGA letterbox self-test mismatch");
      }
      access.Finish();
    }
    // Odd RGB ROI coordinates ensure no accidental YUV chroma-alignment restriction.
    auto& cropped = prep.ClipRgb(11, 13, 101, 99);
    {
      CpuAccess access(cropped.Buffer(), DMA_BUF_SYNC_READ);
      const auto* base = static_cast<const unsigned char*>(cropped.Buffer().Data());
      if (base[0] < 240 || base[1] > 15 || base[2] > 15) {
        throw std::runtime_error("DMA/RGA ROI self-test mismatch");
      }
      access.Finish();
    }
  }
  catch (...) {
    releasebuffer_handle(handle);
    throw;
  }
  releasebuffer_handle(handle);
  DmaBuffer nv12(heap, 640 * 480 * 3 / 2);
  {
    CpuAccess write(nv12, DMA_BUF_SYNC_WRITE);
    std::memset(nv12.Data(), 128, nv12.Size());
    write.Finish();
  }
  handle = importbuffer_fd(nv12.Fd(), static_cast<int>(nv12.Size()));
  if (!handle) {
    throw std::runtime_error("self-test NV12 FD import failed");
  }
  try {
    auto source = wrapbuffer_handle(handle, 640, 480, RK_FORMAT_YCbCr_420_SP);
    source.color_space_mode = IM_YUV_TO_RGB_BT601_LIMIT;
    Preprocessor prep(heap);
    prep.ConvertRaw(source, 640, 480);
    auto& rgb = prep.ClipRgb(11, 13, 101, 99);
    CpuAccess read(rgb.Buffer(), DMA_BUF_SYNC_READ);
    const auto* pixel = static_cast<const unsigned char*>(rgb.Buffer().Data());
    for (int c = 0; c < 3; ++c) {
      if (std::abs(static_cast<int>(pixel[c]) - 130) > 5) {
        throw std::runtime_error("DMA/RGA NV12 color self-test mismatch");
      }
    }
    read.Finish();
  }
  catch (...) {
    releasebuffer_handle(handle);
    throw;
  }
  releasebuffer_handle(handle);
}

}  // namespace camera_dma
