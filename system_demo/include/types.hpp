// 跨模块的数据协议：Options 是配置，FrameTask/ClipTask 是两级队列传递的任务。
// 时间使用 steady_clock；run_ms 和 gate_wait_ms 均以毫秒计。

#pragma once
#include "common.hpp"
#include "buffer_pool.hpp"

namespace yolo_clip {

struct Options {
  std::string yolo_model_path;
  std::string clip_model_path;
  std::string image_dir;
  std::string video_path;
  std::string camera_device;
  // 采集格式与预处理路径相互独立；dma 路径在参数检查阶段自动启用原生输入。
  std::string camera_format = "YUYV";
  std::string camera_path = "dma";
  std::string normalization_mode = "lut";
  std::string dma_heap = "/dev/dma_heap/system";
  int camera_width = 640;
  int camera_height = 480;
  int camera_buffers = 12;
  int opencv_threads = 1;
  bool zero_copy_input = false;
  std::string label_tsv_path;
  std::string text_feature_path;
  int duration_sec = 10;
  double input_fps = 30.0;
  // 队列容量限制等待任务数；clip_buffers 则限制可同时持有的 CLIP 输入总数。
  int frame_queue = 4;
  int clip_queue = 8;
  int clip_buffers = 8;
  int yolo_threads = 2;
  int clip_threads = 4;
  int topk = 5;
  // YOLO 检测阈值和 CLIP 余弦拒识阈值语义不同，不能互相替代。
  float yolo_conf_threshold = 0.05f;
  float reject_threshold = 0.22f;
  std::string preprocess_backend = "opencv";
  // 这些 core 是 NPU 核掩码；CPU 亲和性由 affinity_profile 另行控制。
  std::string clip_core_mode = "round_robin";
  std::string yolo_core_mode = "legacy";
  std::string npu_gate = "none";
  std::string affinity_profile = "none";
  bool use_io_mem = false;
  bool use_affinity = false;
  bool fallback_full_image = false;
  bool serial = false;
};

// 原始输入任务：文件/CPU 图像使用 frame，DMA 采集使用 camera_frame。
// shared_ptr 保证资源跨线程存活；摄像头 CPU 路径可能同时持有这两个字段。
struct FrameTask {
  int seq = 0;
  int image_index = 0;
  std::string source_name;
  std::shared_ptr<cv::Mat> frame;
  std::shared_ptr<camera_dma::Frame> camera_frame;
  Clock::time_point capture_time;
};

// 一个可复用输入张量槽；Data 统一宿主内存和 DMA 映射地址。
struct ClipSlot {
  std::vector<uint8_t> input_bytes;
  std::shared_ptr<camera_dma::DmaBuffer> dma;
  uint8_t* Data() { return dma ? static_cast<uint8_t*>(dma->Data()) : input_bytes.data(); }
  bool valid = false;
};
using ClipSlotPool = BufferPool<ClipSlot>;

// 已完成 ROI 预处理的分类任务，不再需要整张原图；slot 是不可复制的独占租约。
struct ClipTask {
  int seq = 0;
  int image_index = 0;
  std::string source_name;
  ClipSlotPool::Lease slot;
  Clock::time_point capture_time;
  Clock::time_point enqueue_time;
  bool detected = false;
};

// 检测失败时 roi 被设为整图，供上层按 fallback_full_image 决定是否继续分类。
struct YoloResult {
  bool detected = false;
  cv::Rect roi;
  double run_ms = 0.0;
  double gate_wait_ms = 0.0;
};

struct ClipResult {
  double run_ms = 0.0;
  double gate_wait_ms = 0.0;
  uint64_t checksum = 0;
};

// scores 为归一化特征点积所得的余弦相似度，并非 softmax 概率。
struct ClassificationResult {
  std::vector<std::string> topk_labels;
  std::vector<float> scores;
  std::string top1_label;
  std::string mapped_category;
  bool is_unknown = false;
};

struct SamplePrediction {
  int image_index = -1;
  std::string source_name;
  std::vector<std::string> topk_labels;
  std::vector<float> topk_scores;
  std::string top1_label;
  std::string mapped_category;
  double score = 0.0;
  bool is_unknown = false;
};


}  // namespace yolo_clip
