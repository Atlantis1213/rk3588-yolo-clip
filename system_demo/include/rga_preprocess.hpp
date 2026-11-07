// 摄像头 DMA 预处理：先转完整 BGR 图，再分别生成 YOLO 的方形图和 CLIP 的 ROI。
// 每个 YOLO worker 独占一个实例，内部 DMA 图像缓存按尺寸分配并复用。

#pragma once
#include "v4l2_capture.hpp"

namespace camera_dma {
// One instance per YOLO worker; all images are imported once and reused.
class Preprocessor {
 public:
  explicit Preprocessor(std::string heap) : heap_(std::move(heap)) {}
  void Convert(const Frame& frame) {
    const int width = frame.owner->Width(), height = frame.owner->Height();
    ConvertRaw(frame.image, width, height);
  }
  void ConvertRaw(rga_buffer_t source, int width, int height) {
    Ensure(full_, width, height, RK_FORMAT_BGR_888);
    Blit(source, full_->Image(), {0, 0, width, height}, {0, 0, width, height});
  }
  // 原图贴在正方形左上角，右侧/下方填 114，再缩放到 640；不是居中 letterbox。
  DmaImage& YoloRgb() {
    const int width = full_->Width(), height = full_->Height();
    const int side = std::max(width, height);
    Ensure(square_, side, side, RK_FORMAT_BGR_888);
    Ensure(yolo_, 640, 640, RK_FORMAT_RGB_888);
    // This board's RGA driver rejects color-fill jobs. Initialize constant padding
    // once per source geometry instead; each frame overwrites only the image area.
    // 板端驱动拒绝 color-fill，故仅在几何尺寸变化时由 CPU 初始化整块背景。
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
  // ROI 坐标属于完整 BGR 图，RGA 同时裁剪、缩放和转 RGB，输出固定 224×224。
  DmaImage& ClipRgb(int x, int y, int width, int height) {
    Ensure(clip_, 224, 224, RK_FORMAT_RGB_888);
    Blit(full_->Image(), clip_->Image(), {x, y, width, height}, {0, 0, 224, 224});
    return *clip_;
  }
 private:
  void Ensure(std::unique_ptr<DmaImage>& image, int width, int height, int format) {
    if (!image || image->Width() != width || image->Height() != height)
      image = std::make_unique<DmaImage>(heap_, width, height, format);
  }
  std::string heap_;
  int padding_width_ = 0, padding_height_ = 0;
  std::unique_ptr<DmaImage> full_, square_, yolo_, clip_;
};

// Exercise real DMA heap allocation, FD import, color conversion, resize and CPU cache sync.
// 用已知颜色验证真实设备操作及 padding/奇数 ROI/NV12，不能由无硬件单元测试替代。
inline void SelfTest(const std::string& heap) {
  DmaBuffer yuyv(heap, 640 * 480 * 2);
  {
    CpuAccess write(yuyv, DMA_BUF_SYNC_WRITE);
    auto* pixels = static_cast<unsigned char*>(yuyv.Data());
    for (int i = 0; i < 640 * 480 * 2; i += 4) {
      pixels[i] = 82; pixels[i + 1] = 90; pixels[i + 2] = 82; pixels[i + 3] = 240;
    }
    write.Finish();
  }
  rga_buffer_handle_t handle = importbuffer_fd(yuyv.Fd(), static_cast<int>(yuyv.Size()));
  if (!handle) throw std::runtime_error("self-test input FD import failed");
  try {
    auto source = wrapbuffer_handle(handle, 640, 480, RK_FORMAT_YUYV_422);
    source.color_space_mode = IM_YUV_TO_RGB_BT601_LIMIT;
    DmaImage rgb(heap, 224, 224, RK_FORMAT_RGB_888);
    Blit(source, rgb.Image(), {0, 0, 640, 480}, {0, 0, 224, 224});
    CpuAccess read(rgb.Buffer(), DMA_BUF_SYNC_READ);
    const auto* pixel = static_cast<const unsigned char*>(rgb.Buffer().Data());
    if (pixel[0] < 240 || pixel[1] > 15 || pixel[2] > 15)
      throw std::runtime_error("DMA/RGA self-test color mismatch");
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
          std::abs(static_cast<int>(padding[2]) - 114) > 2)
        throw std::runtime_error("DMA/RGA letterbox self-test mismatch");
      access.Finish();
    }
    // Odd RGB ROI coordinates ensure no accidental YUV chroma-alignment restriction.
    auto& cropped = prep.ClipRgb(11, 13, 101, 99);
    {
      CpuAccess access(cropped.Buffer(), DMA_BUF_SYNC_READ);
      const auto* base = static_cast<const unsigned char*>(cropped.Buffer().Data());
      if (base[0] < 240 || base[1] > 15 || base[2] > 15)
        throw std::runtime_error("DMA/RGA ROI self-test mismatch");
      access.Finish();
    }
  } catch (...) { releasebuffer_handle(handle); throw; }
  releasebuffer_handle(handle);
  DmaBuffer nv12(heap, 640 * 480 * 3 / 2);
  {
    CpuAccess write(nv12, DMA_BUF_SYNC_WRITE);
    std::memset(nv12.Data(), 128, nv12.Size());
    write.Finish();
  }
  handle = importbuffer_fd(nv12.Fd(), static_cast<int>(nv12.Size()));
  if (!handle) throw std::runtime_error("self-test NV12 FD import failed");
  try {
    auto source = wrapbuffer_handle(handle, 640, 480, RK_FORMAT_YCbCr_420_SP);
    source.color_space_mode = IM_YUV_TO_RGB_BT601_LIMIT;
    Preprocessor prep(heap);
    prep.ConvertRaw(source, 640, 480);
    auto& rgb = prep.ClipRgb(11, 13, 101, 99);
    CpuAccess read(rgb.Buffer(), DMA_BUF_SYNC_READ);
    const auto* pixel = static_cast<const unsigned char*>(rgb.Buffer().Data());
    for (int c = 0; c < 3; ++c)
      if (std::abs(static_cast<int>(pixel[c]) - 130) > 5)
        throw std::runtime_error("DMA/RGA NV12 color self-test mismatch");
    read.Finish();
  } catch (...) { releasebuffer_handle(handle); throw; }
  releasebuffer_handle(handle);
}


}
