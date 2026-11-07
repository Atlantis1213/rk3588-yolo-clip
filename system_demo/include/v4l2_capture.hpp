// 原始 USB 摄像头采集：V4L2 分配缓冲 → 导出 DMA-BUF → RGA 导入 → DQBUF/QBUF 循环。
// 仅处理单平面渐进式 YUYV/NV12；这里不负责压缩视频解码。

#pragma once
#include "dma_buffer.hpp"

namespace camera_dma {
class V4l2Capture;
// Frame 是已出队缓冲的持有凭证，不复制像素；owner 保持采集设备和底层缓冲存活。
struct Frame {
  std::shared_ptr<V4l2Capture> owner;
  unsigned int index = 0;
  uint32_t sequence = 0;
  int dma_fd = -1;
  const uint8_t* data = nullptr;
  size_t bytes = 0, row_bytes = 0;
  uint32_t fourcc = 0;
  rga_buffer_t image{};
  ~Frame();
};

class V4l2Capture : public std::enable_shared_from_this<V4l2Capture> {
 public:
  static std::shared_ptr<V4l2Capture> Open(const std::string& device, int width,
      int height, const std::string& format, int buffers, double fps, int minimum, bool cpu_copy = false) {
    auto capture = std::shared_ptr<V4l2Capture>(new V4l2Capture());
    capture->Initialize(device, width, height, format, buffers, fps, minimum, cpu_copy);
    return capture;
  }
  ~V4l2Capture() {
    if (streaming_) {
      v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      Ioctl(fd_, VIDIOC_STREAMOFF, &type);
    }
    for (auto& buffer : buffers_) {
      if (buffer.handle) releasebuffer_handle(buffer.handle);
      if (buffer.mapping != MAP_FAILED) munmap(buffer.mapping, buffer.length);
      if (buffer.dma_fd >= 0) close(buffer.dma_fd);
    }
    if (fd_ >= 0) close(fd_);
  }
  int Width() const { return width_; }
  int Height() const { return height_; }
  const std::string& Description() const { return description_; }
  void Start() {
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    Check(Ioctl(fd_, VIDIOC_STREAMON, &type) == 0, "VIDIOC_STREAMON");
    streaming_ = true;
  }

  // nullptr means a bounded timeout or a corrupt frame, not end-of-stream.
  // poll 有界等待使上层能定期检查结束时间；空返回表示暂时无有效帧。
  std::shared_ptr<Frame> Acquire(int timeout_ms = 200) {
    if (requeue_failed_.load()) throw std::runtime_error("V4L2 buffer requeue failed");
    pollfd descriptor{fd_, POLLIN, 0};
    int ret;
    do { ret = poll(&descriptor, 1, timeout_ms); } while (ret < 0 && errno == EINTR);
    Check(ret >= 0, "V4L2 poll");
    if (ret == 0) return nullptr;
    if (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL))
      throw std::runtime_error("V4L2 poll: camera unavailable");
    v4l2_buffer buffer{};
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = V4L2_MEMORY_MMAP;
    // DQBUF/QBUF 可能分别在生产线程和消费线程触发，用同一锁串行访问驱动队列。
    std::lock_guard<std::mutex> lock(io_mutex_);
    if (Ioctl(fd_, VIDIOC_DQBUF, &buffer) < 0) {
      if (errno == EAGAIN) return nullptr;
      Check(false, "VIDIOC_DQBUF");
    }
    if (buffer.index >= buffers_.size())
      throw std::runtime_error("V4L2 returned invalid buffer index");
    // 损坏或不完整的帧立即归还，不交给预处理读取越界/无效内容。
    if ((buffer.flags & V4L2_BUF_FLAG_ERROR) || buffer.bytesused < required_bytes_) {
      Check(Ioctl(fd_, VIDIOC_QBUF, &buffer) == 0, "VIDIOC_QBUF corrupt frame");
      return nullptr;
    }
    std::shared_ptr<Frame> frame;
    try { frame = std::make_shared<Frame>(); }
    catch (...) {
      if (Ioctl(fd_, VIDIOC_QBUF, &buffer) < 0) requeue_failed_.store(true);
      throw;
    }
    // shared_from_this 保证最后一个 Frame 销毁前设备不会被关闭。
    frame->owner = shared_from_this();
    frame->index = buffer.index;
    frame->sequence = buffer.sequence;
    const auto& storage = buffers_[buffer.index];
    frame->dma_fd = storage.dma_fd;
    frame->data = storage.mapping == MAP_FAILED ? nullptr : static_cast<const uint8_t*>(storage.mapping);
    frame->bytes = required_bytes_;
    frame->row_bytes = row_bytes_;
    frame->fourcc = fourcc_;
    frame->image = buffers_[buffer.index].image;
    return frame;
  }

  // Frame 析构不能抛错；归还失败记录到原子标志，在后续 Acquire 中报告。
  void Requeue(unsigned int index) noexcept {
    // DQBUF/QBUF 可能分别在生产线程和消费线程触发，用同一锁串行访问驱动队列。
    std::lock_guard<std::mutex> lock(io_mutex_);
    v4l2_buffer buffer{};
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = V4L2_MEMORY_MMAP;
    buffer.index = index;
    if (Ioctl(fd_, VIDIOC_QBUF, &buffer) < 0) requeue_failed_.store(true);
  }
 private:
  struct Buffer {
    int dma_fd = -1;
    rga_buffer_handle_t handle = 0;
    rga_buffer_t image{};
    void* mapping = MAP_FAILED;
    size_t length = 0;
  };
  V4l2Capture() = default;
  void Initialize(const std::string& device, int width, int height,
      const std::string& format, int count, double fps, int minimum, bool cpu_copy) {
    fd_ = open(device.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
    Check(fd_ >= 0, "open camera " + device);
    v4l2_capability capability{};
    Check(Ioctl(fd_, VIDIOC_QUERYCAP, &capability) == 0, "VIDIOC_QUERYCAP");
    uint32_t caps = capability.capabilities;
    if (caps & V4L2_CAP_DEVICE_CAPS) caps = capability.device_caps;
    if (!(caps & V4L2_CAP_VIDEO_CAPTURE) || !(caps & V4L2_CAP_STREAMING))
      throw std::runtime_error("camera must support single-plane V4L2 streaming capture");
    const uint32_t fourcc = format == "YUYV" ? V4L2_PIX_FMT_YUYV : V4L2_PIX_FMT_NV12;
    v4l2_format config{};
    config.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    config.fmt.pix.width = width;
    config.fmt.pix.height = height;
    config.fmt.pix.pixelformat = fourcc;
    config.fmt.pix.field = V4L2_FIELD_NONE;
    Check(Ioctl(fd_, VIDIOC_S_FMT, &config) == 0, "VIDIOC_S_FMT");
    // VIDIOC_S_FMT 后必须使用驱动实际协商的尺寸和行宽，不能假设等于请求值。
    const auto& pixels = config.fmt.pix;
    if (pixels.pixelformat != fourcc || pixels.field != V4L2_FIELD_NONE)
      throw std::runtime_error("camera did not accept progressive YUYV/NV12; check v4l2-ctl formats");
    width_ = static_cast<int>(pixels.width);
    height_ = static_cast<int>(pixels.height);
    if (width_ <= 0 || height_ <= 0 || width_ > 8192 || height_ > 8192)
      throw std::runtime_error("camera dimensions outside supported range");
    if (width_ % 2 || (format == "NV12" && height_ % 2))
      throw std::runtime_error("camera YUV dimensions must satisfy chroma alignment");
    const unsigned int pixel_bytes = format == "YUYV" ? 2 : 1;
    const unsigned int row_bytes = pixels.bytesperline ? pixels.bytesperline : width_ * pixel_bytes;
    row_bytes_ = row_bytes;
    fourcc_ = fourcc;
    if (row_bytes < width_ * pixel_bytes || row_bytes % pixel_bytes)
      throw std::runtime_error("unsupported V4L2 bytesperline");
    // NV12 在 Y 平面后还有半高 UV 平面；按带 padding 的 row_bytes 计算最低有效字节数。
    required_bytes_ = static_cast<size_t>(row_bytes) * height_;
    if (format == "NV12") required_bytes_ += static_cast<size_t>(row_bytes) * height_ / 2;
    if (pixels.sizeimage < required_bytes_)
      throw std::runtime_error("V4L2 sizeimage too small for negotiated layout");
    v4l2_streamparm parameters{};
    parameters.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parameters.parm.capture.timeperframe.numerator = 1000;
    parameters.parm.capture.timeperframe.denominator = static_cast<unsigned int>(fps * 1000);
    // Some drivers cannot set FPS. Capture still follows the device's negotiated FPS.
    if (Ioctl(fd_, VIDIOC_S_PARM, &parameters) < 0 && errno != EINVAL && errno != ENOTTY)
      Check(false, "VIDIOC_S_PARM");
    v4l2_requestbuffers request{};
    request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    request.memory = V4L2_MEMORY_MMAP;
    request.count = count;
    Check(Ioctl(fd_, VIDIOC_REQBUFS, &request) == 0, "VIDIOC_REQBUFS");
    if (request.count < static_cast<unsigned int>(minimum))
      throw std::runtime_error("too few camera buffers; reduce --frame-queue / --yolo-threads");
    // 使用驱动实际提供的缓冲数量；每个 FD 和 RGA handle 仅初始化一次，逐帧复用。
    buffers_.resize(request.count);
    for (unsigned int i = 0; i < request.count; ++i) {
      v4l2_buffer buffer{};
      buffer.type = request.type;
      buffer.memory = request.memory;
      buffer.index = i;
      Check(Ioctl(fd_, VIDIOC_QUERYBUF, &buffer) == 0, "VIDIOC_QUERYBUF");
      if (buffer.length < required_bytes_) throw std::runtime_error("V4L2 buffer is undersized");
      v4l2_exportbuffer exported{};
      exported.type = request.type;
      exported.index = i;
      exported.flags = O_RDWR | O_CLOEXEC;
      Check(Ioctl(fd_, VIDIOC_EXPBUF, &exported) == 0,
            "VIDIOC_EXPBUF (camera driver must support DMA-BUF export)");
      auto& slot = buffers_[i];
      slot.dma_fd = exported.fd;
      slot.length = buffer.length;
      // 仅 CPU 对照路径需要把采集缓冲映射到 CPU；DMA 路径直接传 FD 给 RGA。
      if (cpu_copy) {
        slot.mapping = mmap(nullptr, buffer.length, PROT_READ, MAP_SHARED, slot.dma_fd, 0);
        Check(slot.mapping != MAP_FAILED, "mmap exported camera buffer for CPU baseline");
      }
      slot.handle = importbuffer_fd(slot.dma_fd, static_cast<int>(buffer.length));
      if (!slot.handle) throw std::runtime_error("RGA cannot import camera DMA-BUF");
      slot.image = wrapbuffer_handle(slot.handle, width_, height_,
          format == "YUYV" ? RK_FORMAT_YUYV_422 : RK_FORMAT_YCbCr_420_SP,
          row_bytes / pixel_bytes, height_);
      // Explicit YUV colorimetry; the full/limited range is negotiated by V4L2.
      // 色彩矩阵（601/709）与 full/limited 范围会改变 RGB 数值，必须显式匹配驱动配置。
      const unsigned int encoding = pixels.ycbcr_enc == V4L2_YCBCR_ENC_DEFAULT ?
          static_cast<unsigned int>(V4L2_MAP_YCBCR_ENC_DEFAULT(pixels.colorspace)) : pixels.ycbcr_enc;
      if (encoding != V4L2_YCBCR_ENC_601 && encoding != V4L2_YCBCR_ENC_709)
        throw std::runtime_error("camera YUV colorimetry must be BT.601 or BT.709");
      const unsigned int quantization = pixels.quantization == V4L2_QUANTIZATION_DEFAULT ?
          static_cast<unsigned int>(V4L2_MAP_QUANTIZATION_DEFAULT(false, pixels.colorspace, encoding)) : pixels.quantization;
      const bool full = quantization == V4L2_QUANTIZATION_FULL_RANGE;
      const bool bt709 = encoding == V4L2_YCBCR_ENC_709;
      if (bt709 && full)
        throw std::runtime_error("installed RGA API cannot express BT.709 full-range conversion");
      slot.image.color_space_mode = bt709 ?
          IM_YUV_TO_RGB_BT709_LIMIT :
          (full ? IM_YUV_TO_RGB_BT601_FULL : IM_YUV_TO_RGB_BT601_LIMIT);
      if (cpu_copy && (bt709 || full))
        throw std::runtime_error("OpenCV CPU baseline requires BT.601 limited-range YUV");
      Check(Ioctl(fd_, VIDIOC_QBUF, &buffer) == 0, "VIDIOC_QBUF initial");
    }
    description_ = device + " " + format + " " + std::to_string(width_) + "x" +
                   std::to_string(height_) + " DMA-BUF buffers=" + std::to_string(request.count);
  }
  int fd_ = -1, width_ = 0, height_ = 0;
  size_t required_bytes_ = 0;
  size_t row_bytes_ = 0;
  uint32_t fourcc_ = 0;
  bool streaming_ = false;
  std::mutex io_mutex_;
  std::atomic<bool> requeue_failed_{false};
  std::vector<Buffer> buffers_;
  std::string description_;
};

// 最后一个 shared_ptr 释放时归还驱动；丢帧和正常消费走相同的生命周期。
inline Frame::~Frame() { if (owner) owner->Requeue(index); }


}
