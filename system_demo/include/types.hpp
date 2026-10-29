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
  int frame_queue = 4;
  int clip_queue = 8;
  int clip_buffers = 8;
  int yolo_threads = 2;
  int clip_threads = 4;
  int topk = 5;
  float yolo_conf_threshold = 0.05f;
  float reject_threshold = 0.22f;
  std::string preprocess_backend = "opencv";
  std::string clip_core_mode = "round_robin";
  std::string yolo_core_mode = "legacy";
  std::string npu_gate = "none";
  std::string affinity_profile = "none";
  bool use_io_mem = false;
  bool use_affinity = false;
  bool fallback_full_image = false;
  bool serial = false;
};

struct FrameTask {
  int seq = 0;
  int image_index = 0;
  std::string source_name;
  std::shared_ptr<cv::Mat> frame;
  std::shared_ptr<camera_dma::Frame> camera_frame;
  Clock::time_point capture_time;
};

struct ClipSlot {
  std::vector<uint8_t> input_bytes;
  std::shared_ptr<camera_dma::DmaBuffer> dma;
  uint8_t* Data() { return dma ? static_cast<uint8_t*>(dma->Data()) : input_bytes.data(); }
  bool valid = false;
};
using ClipSlotPool = BufferPool<ClipSlot>;

struct ClipTask {
  int seq = 0;
  int image_index = 0;
  std::string source_name;
  ClipSlotPool::Lease slot;
  Clock::time_point capture_time;
  Clock::time_point enqueue_time;
  bool detected = false;
};

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
