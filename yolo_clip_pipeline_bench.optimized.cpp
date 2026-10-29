#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <pthread.h>
#include <sched.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <im2d.h>
#include <rga/im2d_buffer.h>

#include "rknn_api.h"
#include "camera_dma.hpp"

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  std::string yolo_model_path;
  std::string clip_model_path;
  std::string image_dir;
  std::string video_path;
  std::string camera_device;
  std::string camera_format = "YUYV";
  std::string dma_heap = "/dev/dma_heap/system";
  int camera_width = 640;
  int camera_height = 480;
  int camera_buffers = 12;
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

struct ClipTask {
  int seq = 0;
  int image_index = 0;
  std::string source_name;
  int slot_id = -1;
  Clock::time_point capture_time;
  Clock::time_point enqueue_time;
  bool detected = false;
};

struct ClipSlot {
  std::vector<uint8_t> input_bytes;
  std::shared_ptr<camera_dma::DmaBuffer> dma;
  uint8_t* Data() { return dma ? static_cast<uint8_t*>(dma->Data()) : input_bytes.data(); }
  bool valid = false;
};

struct YoloResult {
  bool detected = false;
  cv::Rect roi;
  double run_ms = 0.0;
};

struct ClipResult {
  double run_ms = 0.0;
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

struct Stats {
  std::atomic<int> generated{0};
  std::atomic<int> frame_dropped{0};
  std::atomic<int> clip_queue_dropped{0};
  std::atomic<int> clip_slot_dropped{0};
  std::atomic<int> completed{0};
  std::atomic<int> detected{0};
  std::atomic<int> fallback_full_image{0};
  std::atomic<int> skipped_no_object{0};
  std::atomic<int> yolo_runs{0};
  std::atomic<int> clip_runs{0};
  std::atomic<int> yolo_preprocess_runs{0};
  std::atomic<int> clip_preprocess_runs{0};

  mutable std::mutex accum_mutex;
  double yolo_run_ms_sum = 0.0;
  double clip_run_ms_sum = 0.0;
  double yolo_preprocess_ms_sum = 0.0;
  double clip_preprocess_ms_sum = 0.0;
  double yolo_queue_wait_ms_sum = 0.0;
  double clip_queue_wait_ms_sum = 0.0;
  double latency_ms_sum = 0.0;
  double latency_ms_min = 1e30;
  double latency_ms_max = 0.0;
  uint64_t checksum_xor = 0;
  std::vector<double> latency_ms_samples;
  std::map<int, SamplePrediction> sample_predictions;

  void AddYoloRun(double ms) {
    yolo_runs.fetch_add(1);
    std::lock_guard<std::mutex> lock(accum_mutex);
    yolo_run_ms_sum += ms;
  }

  void AddClipRun(double ms) {
    clip_runs.fetch_add(1);
    std::lock_guard<std::mutex> lock(accum_mutex);
    clip_run_ms_sum += ms;
  }

  void AddYoloPreprocess(double ms) {
    yolo_preprocess_runs.fetch_add(1);
    std::lock_guard<std::mutex> lock(accum_mutex);
    yolo_preprocess_ms_sum += ms;
  }

  void AddClipPreprocess(double ms) {
    clip_preprocess_runs.fetch_add(1);
    std::lock_guard<std::mutex> lock(accum_mutex);
    clip_preprocess_ms_sum += ms;
  }

  void AddYoloQueueWait(double ms) {
    std::lock_guard<std::mutex> lock(accum_mutex);
    yolo_queue_wait_ms_sum += ms;
  }

  void AddClipQueueWait(double ms) {
    std::lock_guard<std::mutex> lock(accum_mutex);
    clip_queue_wait_ms_sum += ms;
  }

  void AddCompleted(bool was_detected, double latency_ms, uint64_t checksum, int image_index,
                    const std::string& source_name, const ClassificationResult& cls) {
    completed.fetch_add(1);
    if (was_detected) {
      detected.fetch_add(1);
    } else {
      fallback_full_image.fetch_add(1);
    }
    std::lock_guard<std::mutex> lock(accum_mutex);
    latency_ms_sum += latency_ms;
    latency_ms_samples.push_back(latency_ms);
    latency_ms_min = std::min(latency_ms_min, latency_ms);
    latency_ms_max = std::max(latency_ms_max, latency_ms);
    checksum_xor ^= checksum;
    if (sample_predictions.find(image_index) == sample_predictions.end()) {
      SamplePrediction pred;
      pred.image_index = image_index;
      pred.source_name = source_name;
      pred.topk_labels = cls.topk_labels;
      pred.topk_scores = cls.scores;
      pred.top1_label = cls.top1_label;
      pred.mapped_category = cls.mapped_category;
      pred.score = cls.scores.empty() ? 0.0 : cls.scores[0];
      pred.is_unknown = cls.is_unknown;
      sample_predictions.emplace(image_index, std::move(pred));
    }
  }
};

double Percentile(std::vector<double> samples, double percentile) {
  if (samples.empty()) {
    return 0.0;
  }
  std::sort(samples.begin(), samples.end());
  const size_t index = static_cast<size_t>(std::ceil(percentile * (samples.size() - 1)));
  return samples[index];
}

uint16_t Float32ToFloat16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  const uint32_t mantissa = bits & 0x007fffffu;
  int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;

  if (exponent <= 0) {
    if (exponent < -10) {
      return static_cast<uint16_t>(sign);
    }
    const uint32_t shifted = (mantissa | 0x00800000u) >> (1 - exponent);
    return static_cast<uint16_t>(sign | ((shifted + 0x00001000u) >> 13));
  }
  if (exponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00u);
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) |
                               ((mantissa + 0x00001000u) >> 13));
}

uint64_t ChecksumFloat(const float* data, size_t count) {
  uint64_t sum = 0;
  for (size_t i = 0; i < count; ++i) {
    uint32_t bits;
    std::memcpy(&bits, &data[i], sizeof(bits));
    sum = sum * 131 + bits;
  }
  return sum;
}

float Dot(const float* a, const float* b, size_t count) {
  float sum = 0.0f;
  for (size_t i = 0; i < count; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

float L2Norm(const float* data, size_t count) {
  return std::sqrt(std::max(1e-12f, Dot(data, data, count)));
}

std::string JsonEscape(const std::string& value) {
  std::ostringstream oss;
  for (char ch : value) {
    switch (ch) {
      case '\\':
        oss << "\\\\";
        break;
      case '"':
        oss << "\\\"";
        break;
      case '\n':
        oss << "\\n";
        break;
      case '\r':
        oss << "\\r";
        break;
      case '\t':
        oss << "\\t";
        break;
      default:
        oss << ch;
        break;
    }
  }
  return oss.str();
}

std::string Basename(const std::string& path) {
  const auto pos = path.find_last_of("/\\");
  return pos == std::string::npos ? path : path.substr(pos + 1);
}

void PinThreadToCore(int core_id) {
  if (core_id < 0) {
    return;
  }
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

int ProducerCore(const Options& opt) {
  if (opt.affinity_profile == "legacy") return 7;
  if (opt.affinity_profile == "segregated") return 4;
  return -1;
}

int YoloWorkerCore(const Options& opt, int worker_id) {
  if (opt.affinity_profile == "legacy") return worker_id % 4;
  if (opt.affinity_profile == "segregated") return 5 + (worker_id % 3);
  return -1;
}

int ClipWorkerCore(const Options& opt, int worker_id) {
  if (opt.affinity_profile == "legacy") return 4 + (worker_id % 4);
  if (opt.affinity_profile == "segregated") return worker_id % 4;
  return -1;
}

bool ResizeWithRga(const cv::Mat& src, cv::Mat& dst) {
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

bool ConvertColorWithRga(const cv::Mat& src, cv::Mat& dst, int src_format, int dst_format) {
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

template <typename T>
class DropOldestQueue {
 public:
  explicit DropOldestQueue(size_t capacity) : capacity_(capacity) {}

  template <typename DropFn>
  void Push(T item, DropFn on_drop) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
      return;
    }
    if (queue_.size() >= capacity_) {
      T dropped = std::move(queue_.front());
      queue_.pop();
      on_drop(dropped);
    }
    queue_.push(std::move(item));
    cv_.notify_one();
  }

  bool Pop(T& item) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return closed_ || !queue_.empty(); });
    if (queue_.empty()) {
      return false;
    }
    item = std::move(queue_.front());
    queue_.pop();
    return true;
  }

  void Close() {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    cv_.notify_all();
  }

 private:
  size_t capacity_;
  bool closed_ = false;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::queue<T> queue_;
};

class ClipSlotPool {
 public:
  ClipSlotPool(int count, size_t bytes_per_slot, const std::string& heap = "") : slots_(count) {
    for (int i = 0; i < count; ++i) {
      if (heap.empty()) slots_[i].input_bytes.resize(bytes_per_slot);
      else slots_[i].dma = std::make_shared<camera_dma::DmaBuffer>(heap, bytes_per_slot);
      free_ids_.push(i);
    }
  }

  bool TryAcquire(int& slot_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (free_ids_.empty()) {
      return false;
    }
    slot_id = free_ids_.front();
    free_ids_.pop();
    return true;
  }

  void Release(int slot_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    slots_[slot_id].valid = false;
    free_ids_.push(slot_id);
  }

  ClipSlot& Slot(int slot_id) { return slots_[slot_id]; }

 private:
  std::mutex mutex_;
  std::queue<int> free_ids_;
  std::vector<ClipSlot> slots_;
};

std::vector<uint8_t> ReadBinaryFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("failed to open file: " + path);
  }
  file.seekg(0, std::ios::end);
  const auto size = static_cast<size_t>(file.tellg());
  file.seekg(0, std::ios::beg);
  std::vector<uint8_t> bytes(size);
  if (size > 0) {
    file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
  }
  return bytes;
}

std::vector<std::string> LoadImagePaths(const std::string& dir) {
  DIR* handle = opendir(dir.c_str());
  if (!handle) {
    throw std::runtime_error("failed to open image dir: " + dir);
  }
  std::vector<std::string> paths;
  while (dirent* entry = readdir(handle)) {
    const std::string name = entry->d_name;
    if (name == "." || name == "..") {
      continue;
    }
    const auto dot = name.find_last_of('.');
    if (dot == std::string::npos) {
      continue;
    }
    std::string ext = name.substr(dot + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp") {
      paths.push_back(dir + "/" + name);
    }
  }
  closedir(handle);
  std::sort(paths.begin(), paths.end());
  return paths;
}

std::vector<cv::Mat> LoadImages(const std::string& dir) {
  const auto paths = LoadImagePaths(dir);
  std::vector<cv::Mat> images;
  for (const auto& path : paths) {
    cv::Mat image = cv::imread(path);
    if (!image.empty()) {
      images.push_back(image);
    }
  }
  if (images.empty()) {
    throw std::runtime_error("no images loaded from " + dir);
  }
  return images;
}

cv::VideoCapture OpenVideo(const std::string& path) {
  cv::VideoCapture cap;
  cap.open(path, cv::CAP_FFMPEG);
  if (cap.isOpened()) {
    return cap;
  }
  cap.open(path, cv::CAP_GSTREAMER);
  if (cap.isOpened()) {
    return cap;
  }
  cap.open("file://" + path);
  if (!cap.isOpened()) {
    throw std::runtime_error("failed to open video: " + path);
  }
  return cap;
}

class TextBank {
 public:
  void Load(const std::string& label_tsv_path, const std::string& feature_bin_path) {
    std::ifstream label_file(label_tsv_path);
    if (!label_file) {
      throw std::runtime_error("failed to open label tsv: " + label_tsv_path);
    }
    labels_.clear();
    categories_.clear();
    std::string line;
    while (std::getline(label_file, line)) {
      if (line.empty()) {
        continue;
      }
      const auto tab = line.find('\t');
      if (tab == std::string::npos) {
        throw std::runtime_error("invalid label tsv line: " + line);
      }
      labels_.push_back(line.substr(0, tab));
      categories_.push_back(line.substr(tab + 1));
    }
    if (labels_.empty()) {
      throw std::runtime_error("empty label bank");
    }

    std::vector<uint8_t> feature_bytes = ReadBinaryFile(feature_bin_path);
    if (feature_bytes.size() % sizeof(float) != 0) {
      throw std::runtime_error("invalid feature file size");
    }
    const size_t total_floats = feature_bytes.size() / sizeof(float);
    if (total_floats % labels_.size() != 0) {
      throw std::runtime_error("feature rows do not match label count");
    }
    dim_ = total_floats / labels_.size();
    features_.resize(total_floats);
    std::memcpy(features_.data(), feature_bytes.data(), feature_bytes.size());

    for (size_t row = 0; row < labels_.size(); ++row) {
      float* ptr = features_.data() + row * dim_;
      const float norm = L2Norm(ptr, dim_);
      for (size_t i = 0; i < dim_; ++i) {
        ptr[i] /= norm;
      }
    }
  }

  ClassificationResult Classify(const float* embedding, size_t count, int topk,
                                float reject_threshold) const {
    if (dim_ == 0 || count != dim_) {
      throw std::runtime_error("embedding dim mismatch");
    }

    std::vector<float> normalized(count);
    std::memcpy(normalized.data(), embedding, count * sizeof(float));
    const float norm = L2Norm(normalized.data(), count);
    for (size_t i = 0; i < count; ++i) {
      normalized[i] /= norm;
    }

    std::vector<std::pair<float, int>> scored;
    scored.reserve(labels_.size());
    for (size_t row = 0; row < labels_.size(); ++row) {
      const float* text = features_.data() + row * dim_;
      scored.emplace_back(Dot(text, normalized.data(), dim_), static_cast<int>(row));
    }
    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    ClassificationResult result;
    const int k = std::min(topk, static_cast<int>(scored.size()));
    result.topk_labels.reserve(k);
    result.scores.reserve(k);
    for (int i = 0; i < k; ++i) {
      result.topk_labels.push_back(labels_[scored[i].second]);
      result.scores.push_back(scored[i].first);
    }
    if (k > 0) {
      const int idx = scored[0].second;
      result.top1_label = labels_[idx];
      result.mapped_category = categories_[idx];
      result.is_unknown = scored[0].first < reject_threshold;
      if (result.is_unknown) {
        result.top1_label = "unknown";
        result.mapped_category = "unknown";
      }
    }
    return result;
  }

 private:
  size_t dim_ = 0;
  std::vector<std::string> labels_;
  std::vector<std::string> categories_;
  std::vector<float> features_;
};

rknn_core_mask YoloCoreMask(int worker_id) {
  return worker_id % 2 == 0 ? RKNN_NPU_CORE_0 : RKNN_NPU_CORE_1;
}

rknn_core_mask ClipCoreMask(int worker_id, const std::string& mode) {
  if (mode == "all") {
    return RKNN_NPU_CORE_0_1_2;
  }
  if (mode == "auto") {
    return RKNN_NPU_CORE_AUTO;
  }
  switch (worker_id % 4) {
    case 0:
      return RKNN_NPU_CORE_2;
    case 1:
      return RKNN_NPU_CORE_0;
    case 2:
      return RKNN_NPU_CORE_1;
    default:
      return RKNN_NPU_CORE_2;
  }
}

class BaseRunner {
 public:
  BaseRunner() = default;
  ~BaseRunner() {
    for (auto& entry : imported_inputs_) rknn_destroy_mem(ctx_, entry.second.memory);
    imported_inputs_.clear();
    if (input_mem_ != nullptr) {
      rknn_destroy_mem(ctx_, input_mem_);
      input_mem_ = nullptr;
    }
    if (initialized_) {
      rknn_destroy(ctx_);
    }
  }

  BaseRunner(const BaseRunner&) = delete;
  BaseRunner& operator=(const BaseRunner&) = delete;

  void InitFromModel(const std::vector<uint8_t>& model_bytes, rknn_core_mask core_mask) {
    int ret = rknn_init(&ctx_, const_cast<uint8_t*>(model_bytes.data()),
                        static_cast<uint32_t>(model_bytes.size()), 0, nullptr);
    if (ret != RKNN_SUCC) {
      throw std::runtime_error("rknn_init failed: " + std::to_string(ret));
    }
    initialized_ = true;
    PostInit(core_mask);
  }

  void InitFromContext(rknn_context* root_ctx, rknn_core_mask core_mask) {
    int ret = rknn_dup_context(root_ctx, &ctx_);
    if (ret != RKNN_SUCC) {
      throw std::runtime_error("rknn_dup_context failed: " + std::to_string(ret));
    }
    initialized_ = true;
    PostInit(core_mask);
  }

  rknn_context* MutableContext() { return &ctx_; }
  void EnableIoMem(bool enabled) { use_io_mem_ = enabled; }
  void EnableDirectInput(const std::string& heap) { direct_heap_ = heap; }
  size_t InputAllocationBytes(size_t packed_bytes) const {
    if (direct_heap_.empty()) return packed_bytes;
    if (native_input_attr_.n_elems * sizeof(uint16_t) != packed_bytes)
      throw std::runtime_error("direct input tensor size does not match model");
    return std::max(packed_bytes, static_cast<size_t>(native_input_attr_.size_with_stride));
  }
  void RunDma(const std::shared_ptr<camera_dma::DmaBuffer>& buffer, double& run_ms) {
    auto found = imported_inputs_.find(buffer->Fd());
    if (found == imported_inputs_.end()) {
      rknn_tensor_mem* mem = rknn_create_mem_from_fd(ctx_, buffer->Fd(), buffer->Data(),
                                                  buffer->Size(), 0);
      if (!mem) throw std::runtime_error("rknn_create_mem_from_fd failed");
      try {
        found = imported_inputs_.emplace(buffer->Fd(), ImportedInput{buffer, mem}).first;
      } catch (...) { rknn_destroy_mem(ctx_, mem); throw; }
    }
    rknn_tensor_attr attr = native_input_attr_;
    attr.pass_through = 1;
    const int ret = rknn_set_io_mem(ctx_, found->second.memory, &attr);
    if (ret != RKNN_SUCC)
      throw std::runtime_error("bind native DMA input failed: " + std::to_string(ret));
    RunRaw(nullptr, 0, RKNN_TENSOR_FLOAT16, RKNN_TENSOR_NHWC, run_ms, true);
  }

 protected:
  void PostInit(rknn_core_mask core_mask) {
    int ret = rknn_set_core_mask(ctx_, core_mask);
    if (ret != RKNN_SUCC) {
      throw std::runtime_error("rknn_set_core_mask failed: " + std::to_string(ret));
    }

    rknn_input_output_num io_num;
    std::memset(&io_num, 0, sizeof(io_num));
    ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
      throw std::runtime_error("query io num failed: " + std::to_string(ret));
    }
    if (io_num.n_input != 1 || io_num.n_output != 1) {
      throw std::runtime_error("unexpected tensor count");
    }

    std::memset(&input_attr_, 0, sizeof(input_attr_));
    input_attr_.index = 0;
    ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_attr_, sizeof(input_attr_));
    if (ret != RKNN_SUCC) {
      throw std::runtime_error("query input attr failed: " + std::to_string(ret));
    }

    std::memset(&output_attr_, 0, sizeof(output_attr_));
    output_attr_.index = 0;
    ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attr_, sizeof(output_attr_));
    if (ret != RKNN_SUCC) {
      throw std::runtime_error("query output attr failed: " + std::to_string(ret));
    }

    output_float_.resize(output_attr_.n_elems);
    initialized_ = true;
    if (!direct_heap_.empty()) {
      native_input_attr_ = {};
      native_input_attr_.index = 0;
      ret = rknn_query(ctx_, RKNN_QUERY_NATIVE_NHWC_INPUT_ATTR, &native_input_attr_,
                       sizeof(native_input_attr_));
      if (ret != RKNN_SUCC || native_input_attr_.type != RKNN_TENSOR_FLOAT16 ||
          native_input_attr_.fmt != RKNN_TENSOR_NHWC || native_input_attr_.n_dims != 4 ||
          native_input_attr_.dims[0] != 1 || native_input_attr_.dims[3] != 3 ||
          (native_input_attr_.w_stride != 0 &&
           native_input_attr_.w_stride != native_input_attr_.dims[2]))
        throw std::runtime_error("zero-copy requires packed native FP16 NHWC input; model layout unsupported");
    }
  }

  void RunRaw(const uint8_t* input_buf, uint32_t input_size, rknn_tensor_type type,
              rknn_tensor_format fmt, double& run_ms, bool already_bound = false) {
    int ret = RKNN_SUCC;
    if (already_bound) {
      // CPU normalization already wrote directly into the imported native tensor.
    } else if (use_io_mem_) {
      EnsureInputMem(input_size);
      std::memcpy(input_mem_->virt_addr, input_buf, input_size);
    } else {
      rknn_input input;
      std::memset(&input, 0, sizeof(input));
      input.index = 0;
      input.buf = const_cast<uint8_t*>(input_buf);
      input.size = input_size;
      input.pass_through = 0;
      input.type = type;
      input.fmt = fmt;

      ret = rknn_inputs_set(ctx_, 1, &input);
      if (ret != RKNN_SUCC) {
        throw std::runtime_error("rknn_inputs_set failed: " + std::to_string(ret));
      }
    }

    rknn_output output;
    std::memset(&output, 0, sizeof(output));
    output.want_float = 1;
    output.is_prealloc = 1;
    output.index = 0;
    output.buf = output_float_.data();
    output.size = static_cast<uint32_t>(output_float_.size() * sizeof(float));

    auto start = Clock::now();
    ret = rknn_run(ctx_, nullptr);
    if (ret != RKNN_SUCC) {
      throw std::runtime_error("rknn_run failed: " + std::to_string(ret));
    }
    ret = rknn_outputs_get(ctx_, 1, &output, nullptr);
    if (ret != RKNN_SUCC) {
      throw std::runtime_error("rknn_outputs_get failed: " + std::to_string(ret));
    }
    auto end = Clock::now();
    run_ms = std::chrono::duration<double, std::milli>(end - start).count();

    ret = rknn_outputs_release(ctx_, 1, &output);
    if (ret != RKNN_SUCC) {
      throw std::runtime_error("rknn_outputs_release failed: " + std::to_string(ret));
    }
  }

  void EnsureInputMem(uint32_t input_size) {
    if (input_mem_ != nullptr && input_mem_->size >= input_size) {
      return;
    }
    if (input_mem_ != nullptr) {
      rknn_destroy_mem(ctx_, input_mem_);
      input_mem_ = nullptr;
    }
    input_mem_ = rknn_create_mem(ctx_, input_size);
    if (input_mem_ == nullptr) {
      throw std::runtime_error("rknn_create_mem failed");
    }
    rknn_tensor_attr attr = input_attr_;
    attr.type = RKNN_TENSOR_FLOAT16;
    attr.fmt = RKNN_TENSOR_NHWC;
    const int ret = rknn_set_io_mem(ctx_, input_mem_, &attr);
    if (ret != RKNN_SUCC) {
      throw std::runtime_error("rknn_set_io_mem failed: " + std::to_string(ret));
    }
  }

  rknn_context ctx_ = 0;
  rknn_tensor_attr input_attr_;
  rknn_tensor_attr output_attr_;
  std::vector<float> output_float_;
  rknn_tensor_mem* input_mem_ = nullptr;
  bool use_io_mem_ = false;
  bool initialized_ = false;
  std::string direct_heap_;
  rknn_tensor_attr native_input_attr_{};
  struct ImportedInput {
    std::shared_ptr<camera_dma::DmaBuffer> buffer;
    rknn_tensor_mem* memory;
  };
  std::map<int, ImportedInput> imported_inputs_;
  std::shared_ptr<camera_dma::DmaBuffer> direct_input_;
};

void PackRgbFp16(camera_dma::DmaImage& rgb, uint8_t* destination, bool clip) {
  camera_dma::CpuAccess read(rgb.Buffer(), DMA_BUF_SYNC_READ);
  static const float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
  static const float stdv[3] = {0.26862954f, 0.26130258f, 0.27577711f};
  auto* output = reinterpret_cast<uint16_t*>(destination);
  const auto* base = static_cast<const uint8_t*>(rgb.Buffer().Data());
  for (int y = 0; y < rgb.Height(); ++y) {
    const auto* row = base + y * rgb.RowBytes();
    for (int x = 0; x < rgb.Width(); ++x) {
      for (int c = 0; c < 3; ++c) {
        float value = row[x * 3 + c] / 255.0f;
        if (clip) value = (value - mean[c]) / stdv[c];
        *output++ = Float32ToFloat16(value);
      }
    }
  }
  read.Finish();
}

class YoloRunner : public BaseRunner {
 public:
  explicit YoloRunner(bool use_rga) : use_rga_(use_rga) {}

  YoloResult Run(const cv::Mat& image_bgr, float conf_threshold, double& preprocess_ms) {
    PrepareInput(image_bgr, preprocess_ms);
    return Infer(image_bgr.cols, image_bgr.rows, conf_threshold);
  }

  YoloResult RunCamera(const camera_dma::Frame& frame, camera_dma::Preprocessor& prep,
                      float conf_threshold, double& preprocess_ms) {
    const auto start = Clock::now();
    const int width = frame.owner->Width(), height = frame.owner->Height();
    letterbox_scale_ = 640.0f / std::max(width, height);
    prep.Convert(frame);
    auto& rgb = prep.YoloRgb();
    uint8_t* destination = InputStorage();
    std::unique_ptr<camera_dma::CpuAccess> write;
    if (direct_input_) write = std::make_unique<camera_dma::CpuAccess>(*direct_input_, DMA_BUF_SYNC_WRITE);
    PackRgbFp16(rgb, destination, false);
    if (write) write->Finish();
    preprocess_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    return Infer(width, height, conf_threshold);
  }

 private:
  uint8_t* InputStorage() {
    const size_t bytes = 640 * 640 * 3 * sizeof(uint16_t);
    if (!direct_heap_.empty()) {
      if (!direct_input_) direct_input_ = std::make_shared<camera_dma::DmaBuffer>(
          direct_heap_, InputAllocationBytes(bytes));
      return static_cast<uint8_t*>(direct_input_->Data());
    }
    input_bytes_.resize(bytes);
    return input_bytes_.data();
  }

  YoloResult Infer(int width, int height, float conf_threshold) {
    double run_ms = 0.0;
    if (direct_input_) RunDma(direct_input_, run_ms);
    else RunRaw(input_bytes_.data(), static_cast<uint32_t>(input_bytes_.size()),
                RKNN_TENSOR_FLOAT16, RKNN_TENSOR_NHWC, run_ms);

    YoloResult result;
    result.run_ms = run_ms;
    const float* out = output_float_.data();
    const int count = static_cast<int>(output_attr_.n_elems / 5);
    float best_conf = -1.0f;
    int best_idx = -1;
    for (int i = 0; i < count; ++i) {
      const float conf = out[4 * count + i];
      if (conf > best_conf) {
        best_conf = conf;
        best_idx = i;
      }
    }
    if (best_idx < 0 || best_conf < conf_threshold) {
      result.detected = false;
      result.roi = cv::Rect(0, 0, width, height);
      return result;
    }

    const float cx = out[0 * count + best_idx];
    const float cy = out[1 * count + best_idx];
    const float w = out[2 * count + best_idx];
    const float h = out[3 * count + best_idx];
    const float x1 = std::max(0.0f, (cx - w * 0.5f) / letterbox_scale_);
    const float y1 = std::max(0.0f, (cy - h * 0.5f) / letterbox_scale_);
    const float x2 = std::min(static_cast<float>(width - 1), (cx + w * 0.5f) / letterbox_scale_);
    const float y2 = std::min(static_cast<float>(height - 1), (cy + h * 0.5f) / letterbox_scale_);
    cv::Rect rect(cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
                  cv::Point(static_cast<int>(x2), static_cast<int>(y2)));
    if (rect.width <= 1 || rect.height <= 1) {
      result.detected = false;
      result.roi = cv::Rect(0, 0, width, height);
      return result;
    }

    result.detected = true;
    result.roi = rect & cv::Rect(0, 0, width, height);
    return result;
  }

  void PrepareInput(const cv::Mat& image_bgr, double& preprocess_ms) {
    const auto prep_start = Clock::now();
    const int src_h = image_bgr.rows;
    const int src_w = image_bgr.cols;
    const int side = std::max(src_h, src_w);
    letterbox_scale_ = 640.0f / static_cast<float>(side);

    if (canvas_.rows != side || canvas_.cols != side) {
      canvas_.create(side, side, CV_8UC3);
    }
    canvas_.setTo(cv::Scalar(114, 114, 114));
    image_bgr.copyTo(canvas_(cv::Rect(0, 0, src_w, src_h)));

    if (resized_.rows != 640 || resized_.cols != 640) {
      resized_.create(640, 640, CV_8UC3);
    }
    bool rga_ok = false;
    if (use_rga_) {
      rga_ok = ResizeWithRga(canvas_, resized_);
    }
    if (!rga_ok) {
      cv::resize(canvas_, resized_, cv::Size(640, 640), 0, 0, cv::INTER_LINEAR);
    }
    if (rgb_.rows != 640 || rgb_.cols != 640) {
      rgb_.create(640, 640, CV_8UC3);
    }
    bool color_ok = false;
    if (use_rga_) {
      color_ok = ConvertColorWithRga(resized_, rgb_, RK_FORMAT_BGR_888, RK_FORMAT_RGB_888);
    }
    if (!color_ok) {
      cv::cvtColor(resized_, rgb_, cv::COLOR_BGR2RGB);
    }

    auto* dst = reinterpret_cast<uint16_t*>(InputStorage());
    std::unique_ptr<camera_dma::CpuAccess> write;
    if (direct_input_) write = std::make_unique<camera_dma::CpuAccess>(*direct_input_, DMA_BUF_SYNC_WRITE);
    for (int y = 0; y < 640; ++y) {
      for (int x = 0; x < 640; ++x) {
        for (int c = 0; c < 3; ++c) {
          const float value = rgb_.at<cv::Vec3b>(y, x)[c] / 255.0f;
          *dst++ = Float32ToFloat16(value);
        }
      }
    }
    if (write) write->Finish();
    preprocess_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - prep_start).count();
  }

  bool use_rga_ = false;
  float letterbox_scale_ = 1.0f;
  cv::Mat canvas_;
  cv::Mat resized_;
  cv::Mat rgb_;
  std::vector<uint8_t> input_bytes_;
};

// Preprocessing owns mutable OpenCV buffers, so every producing worker needs its own instance.
class ClipPreprocessor {
 public:
  explicit ClipPreprocessor(bool use_rga) : use_rga_(use_rga) {}

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

    static const float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    static const float stdv[3] = {0.26862954f, 0.26130258f, 0.27577711f};

    auto* dst = reinterpret_cast<uint16_t*>(dst_bytes);
    for (int y = 0; y < 224; ++y) {
      for (int x = 0; x < 224; ++x) {
        const cv::Vec3b pixel = rgb_.at<cv::Vec3b>(y, x);
        for (int c = 0; c < 3; ++c) {
          const float normalized = (pixel[c] / 255.0f - mean[c]) / stdv[c];
          *dst++ = Float32ToFloat16(normalized);
        }
      }
    }
    preprocess_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - prep_start).count();
  }

 private:
  bool use_rga_ = false;
  cv::Mat resized_;
  cv::Mat rgb_;
};

class ClipRunner : public BaseRunner {
 public:
  size_t InputBytes() const { return ClipPreprocessor::InputBytes(); }
  size_t OutputCount() const { return output_float_.size(); }
  const float* OutputData() const { return output_float_.data(); }

  ClipResult RunPrepared(const uint8_t* prepared_bytes) {
    double run_ms = 0.0;
    RunRaw(prepared_bytes, static_cast<uint32_t>(InputBytes()), RKNN_TENSOR_FLOAT16,
           RKNN_TENSOR_NHWC, run_ms);
    ClipResult result;
    result.run_ms = run_ms;
    result.checksum = ChecksumFloat(output_float_.data(), output_float_.size());
    return result;
  }

  ClipResult RunPreparedDma(const std::shared_ptr<camera_dma::DmaBuffer>& buffer) {
    ClipResult result;
    RunDma(buffer, result.run_ms);
    result.checksum = ChecksumFloat(output_float_.data(), output_float_.size());
    return result;
  }

};

Options ParseArgs(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + name);
      }
      ++i;
      return argv[i];
    };
    if (arg == "--yolo") opt.yolo_model_path = next(arg);
    else if (arg == "--clip") opt.clip_model_path = next(arg);
    else if (arg == "--images") opt.image_dir = next(arg);
    else if (arg == "--video") opt.video_path = next(arg);
    else if (arg == "--camera") opt.camera_device = next(arg);
    else if (arg == "--camera-format") opt.camera_format = next(arg);
    else if (arg == "--camera-width") opt.camera_width = std::stoi(next(arg));
    else if (arg == "--camera-height") opt.camera_height = std::stoi(next(arg));
    else if (arg == "--camera-buffers") opt.camera_buffers = std::stoi(next(arg));
    else if (arg == "--dma-heap") opt.dma_heap = next(arg);
    else if (arg == "--zero-copy-input") opt.zero_copy_input = true;
    else if (arg == "--label-tsv") opt.label_tsv_path = next(arg);
    else if (arg == "--text-features") opt.text_feature_path = next(arg);
    else if (arg == "--duration") opt.duration_sec = std::stoi(next(arg));
    else if (arg == "--fps") opt.input_fps = std::stod(next(arg));
    else if (arg == "--frame-queue") opt.frame_queue = std::stoi(next(arg));
    else if (arg == "--clip-queue") opt.clip_queue = std::stoi(next(arg));
    else if (arg == "--clip-buffers") opt.clip_buffers = std::stoi(next(arg));
    else if (arg == "--yolo-threads") opt.yolo_threads = std::stoi(next(arg));
    else if (arg == "--clip-threads") opt.clip_threads = std::stoi(next(arg));
    else if (arg == "--clip-core-mode") opt.clip_core_mode = next(arg);
    else if (arg == "--topk") opt.topk = std::stoi(next(arg));
    else if (arg == "--reject") opt.reject_threshold = std::stof(next(arg));
    else if (arg == "--preprocess") opt.preprocess_backend = next(arg);
    else if (arg == "--io-mem") opt.use_io_mem = true;
    else if (arg == "--affinity") {
      opt.use_affinity = true;
      opt.affinity_profile = "legacy";
    }
    else if (arg == "--affinity-profile") {
      opt.affinity_profile = next(arg);
      opt.use_affinity = opt.affinity_profile != "none";
    }
    else if (arg == "--fallback-full-image") opt.fallback_full_image = true;
    else if (arg == "--conf") opt.yolo_conf_threshold = std::stof(next(arg));
    else if (arg == "--serial") opt.serial = true;
    else throw std::runtime_error("unknown arg: " + arg);
  }
  const bool has_image_dir = !opt.image_dir.empty();
  const bool has_video = !opt.video_path.empty();
  const bool has_camera = !opt.camera_device.empty();
  if (static_cast<int>(has_image_dir) + has_video + has_camera != 1) {
    throw std::runtime_error("provide exactly one of --images DIR, --video FILE, or --camera /dev/videoN");
  }
  if (opt.yolo_model_path.empty() || opt.clip_model_path.empty() ||
      opt.label_tsv_path.empty() || opt.text_feature_path.empty()) {
    throw std::runtime_error(
        "usage: yolo_clip_pipeline_bench --yolo YOLO.rknn --clip CLIP.rknn (--images DIR | --video FILE | --camera DEVICE) "
        "--label-tsv LABELS.tsv --text-features FEATURES.f32 "
        "[--duration 10] [--fps 30] [--serial] [--yolo-threads 2] [--clip-threads 4] "
        "[--frame-queue 4] [--clip-queue 8] [--clip-buffers 8] [--topk 5] "
        "[--clip-core-mode all|round_robin|auto] "
        "[--reject 0.22] [--preprocess opencv|rga] [--io-mem] "
        "[--affinity-profile none|legacy|segregated] "
        "[--fallback-full-image] [--conf 0.05] [--zero-copy-input] [--dma-heap /dev/dma_heap/system] "
        "[--camera-width 640] [--camera-height 480] [--camera-format YUYV|NV12] [--camera-buffers 12]");
  }
  if (opt.duration_sec <= 0 || !std::isfinite(opt.input_fps) || opt.input_fps <= 0.0 ||
      opt.frame_queue <= 0 || opt.clip_queue <= 0 || opt.clip_buffers <= 0 ||
      opt.yolo_threads <= 0 || opt.clip_threads <= 0 || opt.topk <= 0) {
    throw std::runtime_error("duration, fps, queue sizes, buffer count, thread counts, and topk must be positive");
  }
  if (opt.preprocess_backend != "opencv" && opt.preprocess_backend != "rga") {
    throw std::runtime_error("--preprocess must be opencv or rga");
  }
  if (opt.clip_core_mode != "all" && opt.clip_core_mode != "round_robin" &&
      opt.clip_core_mode != "auto") {
    throw std::runtime_error("--clip-core-mode must be all, round_robin, or auto");
  }
  if (opt.affinity_profile != "none" && opt.affinity_profile != "legacy" &&
      opt.affinity_profile != "segregated") {
    throw std::runtime_error("--affinity-profile must be none, legacy, or segregated");
  }
  if (opt.camera_width <= 0 || opt.camera_width > 8192 || opt.camera_height <= 0 ||
      opt.camera_height > 8192 || opt.camera_buffers < 3 || opt.camera_buffers > 64 ||
      (opt.camera_format != "YUYV" && opt.camera_format != "NV12") || opt.input_fps > 1000.0)
    throw std::runtime_error("invalid camera size, buffer count, format, or FPS");
  if (has_camera) {
    if (opt.input_fps < 0.001) throw std::runtime_error("camera FPS must be >= 0.001");
    const int minimum = opt.frame_queue + (opt.serial ? 1 : opt.yolo_threads) + 2;
    if (opt.camera_buffers < minimum)
      throw std::runtime_error("--camera-buffers must be >= frame-queue + YOLO workers + 2");
    opt.preprocess_backend = "rga";
    opt.zero_copy_input = true;
  }
  return opt;
}

void PrintReport(const Options& opt, const Stats& stats, const std::vector<std::string>& image_paths,
                 double wall_ms) {
  const int completed = stats.completed.load();
  const int skipped_no_object = stats.skipped_no_object.load();
  const int processed = completed + skipped_no_object;
  const int yolo_runs = stats.yolo_runs.load();
  const int clip_runs = stats.clip_runs.load();

  double yolo_mean = 0.0;
  double clip_mean = 0.0;
  double yolo_pre_mean = 0.0;
  double clip_pre_mean = 0.0;
  double yolo_queue_wait_mean = 0.0;
  double clip_queue_wait_mean = 0.0;
  double latency_mean = 0.0;
  double latency_min = 0.0;
  double latency_max = 0.0;
  double latency_p50 = 0.0;
  double latency_p95 = 0.0;
  uint64_t checksum = 0;
  std::map<int, SamplePrediction> sample_predictions;
  {
    std::lock_guard<std::mutex> lock(stats.accum_mutex);
    if (yolo_runs > 0) yolo_mean = stats.yolo_run_ms_sum / yolo_runs;
    if (clip_runs > 0) clip_mean = stats.clip_run_ms_sum / clip_runs;
    if (stats.yolo_preprocess_runs.load() > 0) {
      yolo_pre_mean = stats.yolo_preprocess_ms_sum / stats.yolo_preprocess_runs.load();
    }
    if (stats.clip_preprocess_runs.load() > 0) {
      clip_pre_mean = stats.clip_preprocess_ms_sum / stats.clip_preprocess_runs.load();
    }
    if (yolo_runs > 0) yolo_queue_wait_mean = stats.yolo_queue_wait_ms_sum / yolo_runs;
    if (clip_runs > 0) clip_queue_wait_mean = stats.clip_queue_wait_ms_sum / clip_runs;
    if (completed > 0) {
      latency_mean = stats.latency_ms_sum / completed;
      latency_min = stats.latency_ms_min;
      latency_max = stats.latency_ms_max;
      latency_p50 = Percentile(stats.latency_ms_samples, 0.50);
      latency_p95 = Percentile(stats.latency_ms_samples, 0.95);
    }
    checksum = stats.checksum_xor;
    sample_predictions = stats.sample_predictions;
  }

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3);
  oss << "{\n";
  oss << "  \"mode\": \"" << (opt.serial ? "serial" : "two_stage_pool") << "\",\n";
  oss << "  \"preprocess_backend\": \"" << opt.preprocess_backend << "\",\n";
  oss << "  \"clip_core_mode\": \"" << opt.clip_core_mode << "\",\n";
  oss << "  \"use_io_mem\": " << (opt.use_io_mem ? "true" : "false") << ",\n";
  oss << "  \"zero_copy_input\": " << (opt.zero_copy_input ? "true" : "false") << ",\n";
  oss << "  \"camera_device\": \"" << JsonEscape(opt.camera_device) << "\",\n";
  oss << "  \"camera_format\": \"" << opt.camera_format << "\",\n";
  oss << "  \"input_copy_mode\": \"" << (opt.zero_copy_input ? "native_dma_fd" : "host_buffer") << "\",\n";
  oss << "  \"use_affinity\": " << (opt.use_affinity ? "true" : "false") << ",\n";
  oss << "  \"affinity_profile\": \"" << opt.affinity_profile << "\",\n";
  oss << "  \"input_fps\": " << opt.input_fps << ",\n";
  oss << "  \"duration_sec\": " << opt.duration_sec << ",\n";
  oss << "  \"generated_frames\": " << stats.generated.load() << ",\n";
  oss << "  \"frame_queue_dropped\": " << stats.frame_dropped.load() << ",\n";
  oss << "  \"clip_queue_dropped\": " << stats.clip_queue_dropped.load() << ",\n";
  oss << "  \"clip_slot_dropped\": " << stats.clip_slot_dropped.load() << ",\n";
  oss << "  \"completed\": " << completed << ",\n";
  oss << "  \"completed_fps\": " << (completed * 1000.0 / wall_ms) << ",\n";
  oss << "  \"processed\": " << processed << ",\n";
  oss << "  \"processed_fps\": " << (processed * 1000.0 / wall_ms) << ",\n";
  oss << "  \"detected\": " << stats.detected.load() << ",\n";
  oss << "  \"fallback_full_image\": " << stats.fallback_full_image.load() << ",\n";
  oss << "  \"skipped_no_object\": " << skipped_no_object << ",\n";
  oss << "  \"yolo_threads\": " << (opt.serial ? 1 : opt.yolo_threads) << ",\n";
  oss << "  \"clip_threads\": " << (opt.serial ? 1 : opt.clip_threads) << ",\n";
  oss << "  \"clip_buffers\": " << (opt.serial ? 1 : opt.clip_buffers) << ",\n";
  oss << "  \"yolo_preprocess_ms_mean\": " << yolo_pre_mean << ",\n";
  oss << "  \"yolo_queue_wait_ms_mean\": " << yolo_queue_wait_mean << ",\n";
  oss << "  \"yolo_run_ms_mean\": " << yolo_mean << ",\n";
  oss << "  \"clip_preprocess_ms_mean\": " << clip_pre_mean << ",\n";
  oss << "  \"clip_queue_wait_ms_mean\": " << clip_queue_wait_mean << ",\n";
  oss << "  \"clip_run_ms_mean\": " << clip_mean << ",\n";
  oss << "  \"latency_ms_mean\": " << latency_mean << ",\n";
  oss << "  \"latency_ms_min\": " << latency_min << ",\n";
  oss << "  \"latency_ms_p50\": " << latency_p50 << ",\n";
  oss << "  \"latency_ms_p95\": " << latency_p95 << ",\n";
  oss << "  \"latency_ms_max\": " << latency_max << ",\n";
  oss << "  \"wall_ms\": " << wall_ms << ",\n";
  oss << "  \"checksum_xor\": " << checksum << ",\n";
  oss << "  \"sample_predictions\": [\n";
  bool first = true;
  for (const auto& kv : sample_predictions) {
    if (!first) {
      oss << ",\n";
    }
    first = false;
    const auto& pred = kv.second;
    const std::string image_name = !pred.source_name.empty()
                                       ? pred.source_name
                                       : (pred.image_index >= 0 &&
                                                  pred.image_index < static_cast<int>(image_paths.size())
                                              ? Basename(image_paths[pred.image_index])
                                              : std::to_string(pred.image_index));
    oss << "    {\"image_index\": " << pred.image_index << ", \"image\": \""
        << JsonEscape(image_name) << "\", \"top1_label\": \"" << JsonEscape(pred.top1_label)
        << "\", \"mapped_category\": \"" << JsonEscape(pred.mapped_category)
        << "\", \"score\": " << pred.score << ", \"is_unknown\": "
        << (pred.is_unknown ? "true" : "false") << ", \"topk_labels\": [";
    for (size_t i = 0; i < pred.topk_labels.size(); ++i) {
      if (i > 0) oss << ", ";
      oss << "\"" << JsonEscape(pred.topk_labels[i]) << "\"";
    }
    oss << "], \"topk_scores\": [";
    for (size_t i = 0; i < pred.topk_scores.size(); ++i) {
      if (i > 0) oss << ", ";
      oss << pred.topk_scores[i];
    }
    oss << "]}";
  }
  oss << "\n  ]\n";
  oss << "}\n";
  std::cout << oss.str();
}

class ThreadFailure {
 public:
  std::atomic<bool> stopped{false};
  template <typename Work, typename Wake>
  void Guard(Work work, Wake wake) noexcept {
    try { work(); }
    catch (...) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!error_) error_ = std::current_exception();
      }
      stopped.store(true);
      wake();
    }
  }
  void Rethrow() {
    if (error_) std::rethrow_exception(error_);
  }
 private:
  std::mutex mutex_;
  std::exception_ptr error_;
};

std::shared_ptr<camera_dma::V4l2Capture> OpenCamera(const Options& opt) {
  if (opt.camera_device.empty()) return nullptr;
  auto camera = camera_dma::V4l2Capture::Open(opt.camera_device, opt.camera_width,
      opt.camera_height, opt.camera_format, opt.camera_buffers, opt.input_fps,
      opt.frame_queue + (opt.serial ? 1 : opt.yolo_threads) + 2);
  std::cerr << "V4L2: " << camera->Description() << '\n';
  return camera;
}

void ProduceCamera(const Options& opt, const std::shared_ptr<camera_dma::V4l2Capture>& camera,
                   DropOldestQueue<FrameTask>& queue, Stats& stats, Clock::time_point start,
                   const ThreadFailure& failure) {
  const auto deadline = start + std::chrono::seconds(opt.duration_sec);
  int sequence = 0;
  while (!failure.stopped.load() && Clock::now() < deadline) {
    auto frame = camera->Acquire();
    if (!frame) continue;
    FrameTask task;
    task.seq = sequence++;
    task.image_index = -1;
    task.source_name = "camera_frame_" + std::to_string(frame->sequence);
    task.capture_time = Clock::now();
    task.camera_frame = std::move(frame);
    stats.generated.fetch_add(1);
    queue.Push(std::move(task), [&](const FrameTask&) { stats.frame_dropped.fetch_add(1); });
  }
  if (!failure.stopped.load() && sequence == 0)
    throw std::runtime_error("camera produced no valid frames within duration");
}

void PrepareClipTask(const FrameTask& task, const cv::Rect& roi, ClipPreprocessor& cpu,
                     camera_dma::Preprocessor& hardware, uint8_t* destination,
                     const std::shared_ptr<camera_dma::DmaBuffer>& dma, double& elapsed_ms) {
  const auto start = Clock::now();
  std::unique_ptr<camera_dma::CpuAccess> write;
  if (dma) write = std::make_unique<camera_dma::CpuAccess>(*dma, DMA_BUF_SYNC_WRITE);
  if (task.camera_frame) {
    auto& rgb = hardware.ClipRgb(roi.x, roi.y, roi.width, roi.height);
    PackRgbFp16(rgb, destination, true);
  } else {
    cpu.PrepareInput((*task.frame)(roi), destination, elapsed_ms);
  }
  if (write) write->Finish();
  elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void RunSerial(const Options& opt, const std::vector<cv::Mat>& images,
               const std::vector<std::string>& image_paths, const TextBank& text_bank,
               const std::vector<uint8_t>& yolo_model, const std::vector<uint8_t>& clip_model) {
  Stats stats;
  DropOldestQueue<FrameTask> frame_queue(static_cast<size_t>(opt.frame_queue));
  const bool use_rga = opt.preprocess_backend == "rga";

  auto camera = OpenCamera(opt);
  ThreadFailure failure;
  auto wake = [&]() { frame_queue.Close(); };
  camera_dma::Preprocessor hardware(opt.dma_heap);

  YoloRunner yolo(use_rga);
  yolo.EnableIoMem(opt.use_io_mem);
  if (opt.zero_copy_input) yolo.EnableDirectInput(opt.dma_heap);
  yolo.InitFromModel(yolo_model, RKNN_NPU_CORE_AUTO);
  ClipRunner clip;
  clip.EnableIoMem(opt.use_io_mem);
  if (opt.zero_copy_input) clip.EnableDirectInput(opt.dma_heap);
  clip.InitFromModel(clip_model, RKNN_NPU_CORE_AUTO);
  ClipPreprocessor clip_preprocessor(use_rga);
  std::vector<uint8_t> clip_input(clip.InputBytes());
  std::shared_ptr<camera_dma::DmaBuffer> clip_dma;
  if (opt.zero_copy_input) clip_dma = std::make_shared<camera_dma::DmaBuffer>(
      opt.dma_heap, clip.InputAllocationBytes(clip.InputBytes()));

  if (camera) camera->Start();
  const auto wall_start = Clock::now();
  std::thread producer([&]() {
    failure.Guard([&]() {
    PinThreadToCore(ProducerCore(opt));
    if (camera) {
      ProduceCamera(opt, camera, frame_queue, stats, wall_start, failure);
      return;
    }
    const int total_frames = static_cast<int>(opt.duration_sec * opt.input_fps);
    const auto interval = std::chrono::duration<double>(1.0 / opt.input_fps);
    cv::VideoCapture cap;
    if (!opt.video_path.empty()) {
      cap = OpenVideo(opt.video_path);
    }
    for (int i = 0; i < total_frames && !failure.stopped.load(); ++i) {
      const auto target_time = wall_start + std::chrono::duration_cast<Clock::duration>(interval * i);
      std::this_thread::sleep_until(target_time);
      FrameTask task;
      task.seq = i;
      if (!opt.video_path.empty()) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
          cap.release();
          cap = OpenVideo(opt.video_path);
          if (!cap.read(frame) || frame.empty()) {
            throw std::runtime_error("failed to loop video input");
          }
        }
        task.image_index = -1;
        task.source_name = "video_frame";
        task.frame = std::make_shared<cv::Mat>(frame.clone());
      } else {
        task.image_index = i % static_cast<int>(images.size());
        task.source_name = Basename(image_paths[task.image_index]);
        task.frame = std::make_shared<cv::Mat>(images[task.image_index]);
      }
      task.capture_time = Clock::now();
      stats.generated.fetch_add(1);
      frame_queue.Push(std::move(task), [&](const FrameTask&) { stats.frame_dropped.fetch_add(1); });
    }
    }, wake);
    frame_queue.Close();
  });

  std::thread worker([&]() {
    failure.Guard([&]() {
    PinThreadToCore(YoloWorkerCore(opt, 0));
    FrameTask task;
    while (!failure.stopped.load() && frame_queue.Pop(task)) {
      stats.AddYoloQueueWait(
          std::chrono::duration<double, std::milli>(Clock::now() - task.capture_time).count());
      double yolo_pre_ms = 0.0;
      const YoloResult yolo_result = task.camera_frame ?
          yolo.RunCamera(*task.camera_frame, hardware, opt.yolo_conf_threshold, yolo_pre_ms) :
          yolo.Run(*task.frame, opt.yolo_conf_threshold, yolo_pre_ms);
      stats.AddYoloPreprocess(yolo_pre_ms);
      stats.AddYoloRun(yolo_result.run_ms);

      if (!yolo_result.detected && !opt.fallback_full_image) {
        stats.skipped_no_object.fetch_add(1);
        continue;
      }

      double clip_pre_ms = 0.0;
      PrepareClipTask(task, yolo_result.roi, clip_preprocessor, hardware,
          clip_dma ? static_cast<uint8_t*>(clip_dma->Data()) : clip_input.data(), clip_dma, clip_pre_ms);
      task.camera_frame.reset();
      stats.AddClipPreprocess(clip_pre_ms);
      const ClipResult clip_result = clip_dma ? clip.RunPreparedDma(clip_dma) :
                                               clip.RunPrepared(clip_input.data());
      stats.AddClipRun(clip_result.run_ms);
      const ClassificationResult cls =
          text_bank.Classify(clip.OutputData(), clip.OutputCount(), opt.topk, opt.reject_threshold);

      const auto now = Clock::now();
      const double latency_ms =
          std::chrono::duration<double, std::milli>(now - task.capture_time).count();
      stats.AddCompleted(yolo_result.detected, latency_ms, clip_result.checksum, task.image_index,
                         task.source_name, cls);
    }
    }, wake);
  });

  producer.join();
  worker.join();
  failure.Rethrow();
  const auto wall_end = Clock::now();
  PrintReport(opt, stats, image_paths,
              std::chrono::duration<double, std::milli>(wall_end - wall_start).count());
}

void RunPool(const Options& opt, const std::vector<cv::Mat>& images,
             const std::vector<std::string>& image_paths, const TextBank& text_bank,
             const std::vector<uint8_t>& yolo_model,
             const std::vector<uint8_t>& clip_model) {
  Stats stats;
  DropOldestQueue<FrameTask> frame_queue(static_cast<size_t>(opt.frame_queue));
  DropOldestQueue<ClipTask> clip_queue(static_cast<size_t>(opt.clip_queue));
  const bool use_rga = opt.preprocess_backend == "rga";
  auto camera = OpenCamera(opt);
  ThreadFailure failure;
  auto wake = [&]() { frame_queue.Close(); clip_queue.Close(); };

  std::vector<std::unique_ptr<YoloRunner>> yolo_workers;
  yolo_workers.reserve(opt.yolo_threads);
  auto yolo_root = std::make_unique<YoloRunner>(use_rga);
  yolo_root->EnableIoMem(opt.use_io_mem);
  if (opt.zero_copy_input) yolo_root->EnableDirectInput(opt.dma_heap);
  yolo_root->InitFromModel(yolo_model, YoloCoreMask(0));
  yolo_workers.push_back(std::move(yolo_root));
  for (int i = 1; i < opt.yolo_threads; ++i) {
    auto runner = std::make_unique<YoloRunner>(use_rga);
    runner->EnableIoMem(opt.use_io_mem);
    if (opt.zero_copy_input) runner->EnableDirectInput(opt.dma_heap);
    runner->InitFromContext(yolo_workers[0]->MutableContext(), YoloCoreMask(i));
    yolo_workers.push_back(std::move(runner));
  }

  std::vector<std::unique_ptr<ClipRunner>> clip_workers;
  clip_workers.reserve(opt.clip_threads);
  auto clip_root = std::make_unique<ClipRunner>();
  clip_root->EnableIoMem(opt.use_io_mem);
  if (opt.zero_copy_input) clip_root->EnableDirectInput(opt.dma_heap);
  clip_root->InitFromModel(clip_model, ClipCoreMask(0, opt.clip_core_mode));
  const size_t clip_input_bytes = clip_root->InputAllocationBytes(clip_root->InputBytes());
  clip_workers.push_back(std::move(clip_root));
  for (int i = 1; i < opt.clip_threads; ++i) {
    auto runner = std::make_unique<ClipRunner>();
    runner->EnableIoMem(opt.use_io_mem);
    if (opt.zero_copy_input) runner->EnableDirectInput(opt.dma_heap);
    runner->InitFromContext(clip_workers[0]->MutableContext(), ClipCoreMask(i, opt.clip_core_mode));
    clip_workers.push_back(std::move(runner));
  }

  ClipSlotPool clip_slots(opt.clip_buffers, clip_input_bytes,
                         opt.zero_copy_input ? opt.dma_heap : "");
  std::vector<std::unique_ptr<ClipPreprocessor>> clip_preprocessors;
  clip_preprocessors.reserve(opt.yolo_threads);
  std::vector<std::unique_ptr<camera_dma::Preprocessor>> hardware_preprocessors;
  for (int i = 0; i < opt.yolo_threads; ++i) {
    clip_preprocessors.push_back(std::make_unique<ClipPreprocessor>(use_rga));
    hardware_preprocessors.push_back(std::make_unique<camera_dma::Preprocessor>(opt.dma_heap));
  }
  std::atomic<int> yolo_alive{opt.yolo_threads};
  std::atomic<int> clip_alive{opt.clip_threads};

  if (camera) camera->Start();
  const auto wall_start = Clock::now();
  std::thread producer([&]() {
    failure.Guard([&]() {
    PinThreadToCore(ProducerCore(opt));
    if (camera) {
      ProduceCamera(opt, camera, frame_queue, stats, wall_start, failure);
      return;
    }
    const int total_frames = static_cast<int>(opt.duration_sec * opt.input_fps);
    const auto interval = std::chrono::duration<double>(1.0 / opt.input_fps);
    cv::VideoCapture cap;
    if (!opt.video_path.empty()) {
      cap = OpenVideo(opt.video_path);
    }
    for (int i = 0; i < total_frames && !failure.stopped.load(); ++i) {
      const auto target_time = wall_start + std::chrono::duration_cast<Clock::duration>(interval * i);
      std::this_thread::sleep_until(target_time);
      FrameTask task;
      task.seq = i;
      if (!opt.video_path.empty()) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
          cap.release();
          cap = OpenVideo(opt.video_path);
          if (!cap.read(frame) || frame.empty()) {
            throw std::runtime_error("failed to loop video input");
          }
        }
        task.image_index = -1;
        task.source_name = "video_frame";
        task.frame = std::make_shared<cv::Mat>(frame.clone());
      } else {
        task.image_index = i % static_cast<int>(images.size());
        task.source_name = Basename(image_paths[task.image_index]);
        task.frame = std::make_shared<cv::Mat>(images[task.image_index]);
      }
      task.capture_time = Clock::now();
      stats.generated.fetch_add(1);
      frame_queue.Push(std::move(task), [&](const FrameTask&) { stats.frame_dropped.fetch_add(1); });
    }
    }, wake);
    frame_queue.Close();
  });

  std::vector<std::thread> yolo_threads;
  for (int i = 0; i < opt.yolo_threads; ++i) {
    yolo_threads.emplace_back([&, i]() {
      failure.Guard([&]() {
      PinThreadToCore(YoloWorkerCore(opt, i));
      FrameTask task;
      while (!failure.stopped.load() && frame_queue.Pop(task)) {
        stats.AddYoloQueueWait(
            std::chrono::duration<double, std::milli>(Clock::now() - task.capture_time).count());
        double yolo_pre_ms = 0.0;
        const YoloResult yolo_result = task.camera_frame ?
            yolo_workers[i]->RunCamera(*task.camera_frame, *hardware_preprocessors[i],
                                      opt.yolo_conf_threshold, yolo_pre_ms) :
            yolo_workers[i]->Run(*task.frame, opt.yolo_conf_threshold, yolo_pre_ms);
        stats.AddYoloPreprocess(yolo_pre_ms);
        stats.AddYoloRun(yolo_result.run_ms);

        if (!yolo_result.detected && !opt.fallback_full_image) {
          stats.skipped_no_object.fetch_add(1);
          continue;
        }

        int slot_id = -1;
        if (!clip_slots.TryAcquire(slot_id)) {
          stats.clip_slot_dropped.fetch_add(1);
          continue;
        }

        double clip_pre_ms = 0.0;
        auto& slot = clip_slots.Slot(slot_id);
        PrepareClipTask(task, yolo_result.roi, *clip_preprocessors[i], *hardware_preprocessors[i],
                        slot.Data(), slot.dma, clip_pre_ms);
        task.camera_frame.reset();
        stats.AddClipPreprocess(clip_pre_ms);
        clip_slots.Slot(slot_id).valid = true;

        ClipTask clip_task;
        clip_task.seq = task.seq;
        clip_task.image_index = task.image_index;
        clip_task.source_name = task.source_name;
        clip_task.slot_id = slot_id;
        clip_task.capture_time = task.capture_time;
        clip_task.enqueue_time = Clock::now();
        clip_task.detected = yolo_result.detected;
        clip_queue.Push(std::move(clip_task), [&](const ClipTask& dropped) {
          clip_slots.Release(dropped.slot_id);
          stats.clip_queue_dropped.fetch_add(1);
        });
      }
      }, wake);
      if (yolo_alive.fetch_sub(1) == 1) {
        clip_queue.Close();
      }
    });
  }

  std::vector<std::thread> clip_threads;
  for (int i = 0; i < opt.clip_threads; ++i) {
    clip_threads.emplace_back([&, i]() {
      failure.Guard([&]() {
      PinThreadToCore(ClipWorkerCore(opt, i));
      ClipTask task;
      while (!failure.stopped.load() && clip_queue.Pop(task)) {
        stats.AddClipQueueWait(
            std::chrono::duration<double, std::milli>(Clock::now() - task.enqueue_time).count());
        ClipSlot& slot = clip_slots.Slot(task.slot_id);
        const ClipResult clip_result = slot.dma ? clip_workers[i]->RunPreparedDma(slot.dma) :
                                                 clip_workers[i]->RunPrepared(slot.Data());
        stats.AddClipRun(clip_result.run_ms);
        const ClassificationResult cls = text_bank.Classify(
            clip_workers[i]->OutputData(), clip_workers[i]->OutputCount(), opt.topk,
            opt.reject_threshold);
        const auto now = Clock::now();
        const double latency_ms =
            std::chrono::duration<double, std::milli>(now - task.capture_time).count();
        stats.AddCompleted(task.detected, latency_ms, clip_result.checksum, task.image_index,
                           task.source_name, cls);
        clip_slots.Release(task.slot_id);
      }
      }, wake);
      clip_alive.fetch_sub(1);
    });
  }

  producer.join();
  for (auto& t : yolo_threads) t.join();
  for (auto& t : clip_threads) t.join();
  failure.Rethrow();
  const auto wall_end = Clock::now();
  PrintReport(opt, stats, image_paths,
              std::chrono::duration<double, std::milli>(wall_end - wall_start).count());
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc >= 2 && std::string(argv[1]) == "--dma-self-test") {
      if (argc != 2 && !(argc == 4 && std::string(argv[2]) == "--dma-heap"))
        throw std::runtime_error("--dma-self-test [--dma-heap PATH]");
      camera_dma::SelfTest(argc == 4 ? argv[3] : "/dev/dma_heap/system");
      std::cout << "DMA heap + RGA FD import + color/resize/letterbox/ROI + cache sync: PASS\n";
      return 0;
    }
    const Options opt = ParseArgs(argc, argv);
    const auto image_paths = opt.image_dir.empty() ? std::vector<std::string>{} : LoadImagePaths(opt.image_dir);
    const auto images = opt.image_dir.empty() ? std::vector<cv::Mat>{} : LoadImages(opt.image_dir);
    TextBank text_bank;
    text_bank.Load(opt.label_tsv_path, opt.text_feature_path);
    const auto yolo_model = ReadBinaryFile(opt.yolo_model_path);
    const auto clip_model = ReadBinaryFile(opt.clip_model_path);

    if (opt.serial) {
      RunSerial(opt, images, image_paths, text_bank, yolo_model, clip_model);
    } else {
      RunPool(opt, images, image_paths, text_bank, yolo_model, clip_model);
    }
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << std::endl;
    return 1;
  }
}
