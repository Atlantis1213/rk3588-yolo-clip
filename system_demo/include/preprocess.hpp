// 宿主图像预处理和 DMA RGB 张量打包。模型输入顺序均为 RGB、NHWC、FP16。
// 文件输入的 RGA 虚拟地址辅助函数可回退 OpenCV；摄像头 FD 路径由 camera_dma::Preprocessor 实现。

#pragma once
#include "types.hpp"
#include "utils.hpp"
#include "normalization.hpp"
#include "copy_metrics.hpp"

namespace yolo_clip {

// CPU 基线显式复制 raw 帧后由 OpenCV 转色，这笔复制单独记入 CopyMetrics。
class CameraCpuPreprocessor {
 public:
  cv::Mat Convert(const camera_dma::Frame& frame, double& elapsed_ms) {
    const auto start = Clock::now();
    if (!frame.data) throw std::runtime_error("CPU camera buffer is not mapped");
    raw_.resize(frame.bytes);
    dma_buf_sync sync{};
    sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ;
    camera_dma::Check(camera_dma::Ioctl(frame.dma_fd, DMA_BUF_IOCTL_SYNC, &sync) == 0,
                      "start camera CPU access");
    std::memcpy(raw_.data(), frame.data, frame.bytes);
    sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ;
    camera_dma::Check(camera_dma::Ioctl(frame.dma_fd, DMA_BUF_IOCTL_SYNC, &sync) == 0,
                      "finish camera CPU access");
    copy_metrics.Raw(frame.bytes);
    const int width = frame.owner->Width(), height = frame.owner->Height();
    if (frame.fourcc == V4L2_PIX_FMT_YUYV) {
      cv::Mat yuyv(height, width, CV_8UC2, raw_.data(), frame.row_bytes);
      cv::cvtColor(yuyv, bgr_, cv::COLOR_YUV2BGR_YUYV);
    } else {
      cv::Mat nv12(height * 3 / 2, width, CV_8UC1, raw_.data(), frame.row_bytes);
      cv::cvtColor(nv12, bgr_, cv::COLOR_YUV2BGR_NV12);
    }
    elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    return bgr_;
  }
 private:
  std::vector<uint8_t> raw_;
  cv::Mat bgr_;
};

// ROI 的 cv::Mat 可能不连续；先得到连续缓冲后才按紧凑图像包装 RGA 虚拟地址。
inline bool ResizeWithRga(const cv::Mat& src, cv::Mat& dst) {
  if (src.empty() || dst.empty()) {
    return false;
  }
  cv::Mat src_cont = src.isContinuous() ? src : src.clone();
  cv::Mat dst_cont = dst.isContinuous() ? dst : dst.clone();
  rga_buffer_t src_buf =
      wrapbuffer_virtualaddr(src_cont.data, src_cont.cols, src_cont.rows, RK_FORMAT_BGR_888);
  rga_buffer_t dst_buf =
      wrapbuffer_virtualaddr(dst_cont.data, dst_cont.cols, dst_cont.rows, RK_FORMAT_BGR_888);
  im_rect src_rect;
  im_rect dst_rect;
  std::memset(&src_rect, 0, sizeof(src_rect));
  std::memset(&dst_rect, 0, sizeof(dst_rect));
  const int ret = imcheck(src_buf, dst_buf, src_rect, dst_rect);
  if (ret != IM_STATUS_NOERROR) {
    return false;
  }
  const IM_STATUS status = imresize(src_buf, dst_buf);
  if (status != IM_STATUS_SUCCESS) {
    return false;
  }
  if (!dst.isContinuous()) {
    dst_cont.copyTo(dst);
  }
  return true;
}

inline bool ConvertColorWithRga(const cv::Mat& src, cv::Mat& dst, int src_format, int dst_format) {
  if (src.empty() || dst.empty()) {
    return false;
  }
  cv::Mat src_cont = src.isContinuous() ? src : src.clone();
  cv::Mat dst_cont = dst.isContinuous() ? dst : dst.clone();
  rga_buffer_t src_buf =
      wrapbuffer_virtualaddr(src_cont.data, src_cont.cols, src_cont.rows, src_format);
  rga_buffer_t dst_buf =
      wrapbuffer_virtualaddr(dst_cont.data, dst_cont.cols, dst_cont.rows, dst_format);
  const IM_STATUS status = imcvtcolor(src_buf, dst_buf, src_format, dst_format);
  if (status != IM_STATUS_SUCCESS) {
    return false;
  }
  if (!dst.isContinuous()) {
    dst_cont.copyTo(dst);
  }
  return true;
}

// RGA 写完 RGB 后，CPU 在同步读作用域中归一化；输出地址可直接指向 RKNN 导入的 DMA。
inline void PackRgbFp16(camera_dma::DmaImage& rgb, uint8_t* destination, bool clip,
                        normalization::Mode mode = normalization::Mode::Scalar) {
  camera_dma::CpuAccess read(rgb.Buffer(), DMA_BUF_SYNC_READ);
  auto* output = reinterpret_cast<uint16_t*>(destination);
  const auto* base = static_cast<const uint8_t*>(rgb.Buffer().Data());
  normalization::Run(base, rgb.Width(), rgb.Height(), rgb.RowBytes(), output, clip, mode);
  copy_metrics.Normalized(static_cast<size_t>(rgb.Width()) * rgb.Height() * 3 * sizeof(uint16_t));
  read.Finish();
}

// 沿用本项目部署契约：直接 resize 到 224×224，不是官方 CLIP 常见的中心裁剪。
class ClipPreprocessor {
 public:
  explicit ClipPreprocessor(bool use_rga, normalization::Mode mode = normalization::Mode::Scalar)
      : use_rga_(use_rga), mode_(mode) {}

  static constexpr size_t InputBytes() { return 1 * 224 * 224 * 3 * sizeof(uint16_t); }

  void PrepareInput(const cv::Mat& image_bgr, uint8_t* dst_bytes, double& preprocess_ms) {
    const auto prep_start = Clock::now();
    if (resized_.rows != 224 || resized_.cols != 224) {
      resized_.create(224, 224, CV_8UC3);
    }
    bool rga_ok = false;
    if (use_rga_) {
      rga_ok = ResizeWithRga(image_bgr, resized_);
    }
    if (!rga_ok) {
      cv::resize(image_bgr, resized_, cv::Size(224, 224), 0, 0, cv::INTER_LINEAR);
    }
    if (rgb_.rows != 224 || rgb_.cols != 224) {
      rgb_.create(224, 224, CV_8UC3);
    }
    bool color_ok = false;
    if (use_rga_) {
      color_ok = ConvertColorWithRga(resized_, rgb_, RK_FORMAT_BGR_888, RK_FORMAT_RGB_888);
    }
    if (!color_ok) {
      cv::cvtColor(resized_, rgb_, cv::COLOR_BGR2RGB);
    }

    auto* dst = reinterpret_cast<uint16_t*>(dst_bytes);
    // 归一化直接写目标张量；按源 step 读行，输出无行 padding。
    normalization::Run(rgb_.data, 224, 224, rgb_.step, dst, true, mode_);
    copy_metrics.Normalized(InputBytes());
    preprocess_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - prep_start).count();
  }

 private:
  bool use_rga_ = false;
  normalization::Mode mode_;
  cv::Mat resized_;
  cv::Mat rgb_;
};


}  // namespace yolo_clip
