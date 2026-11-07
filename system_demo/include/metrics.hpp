// 多线程统计与最终 JSON 报告。计数器用原子变量，累计值/样本集合由 accum_mutex 保护。
// 完整分类吞吐、仅检测处理吞吐、单帧延迟属于不同指标，解读时需分别看分母。

#pragma once
#include "types.hpp"
#include "utils.hpp"
#include "copy_metrics.hpp"

namespace yolo_clip {

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
  double yolo_gate_wait_ms_sum = 0.0;
  double clip_gate_wait_ms_sum = 0.0;
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

  void AddYoloRun(double ms, double gate_ms = 0.0) {
    yolo_runs.fetch_add(1);
    std::lock_guard<std::mutex> lock(accum_mutex);
    yolo_run_ms_sum += ms;
    yolo_gate_wait_ms_sum += gate_ms;
  }

  void AddClipRun(double ms, double gate_ms = 0.0) {
    clip_runs.fetch_add(1);
    std::lock_guard<std::mutex> lock(accum_mutex);
    clip_run_ms_sum += ms;
    clip_gate_wait_ms_sum += gate_ms;
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

  // 只有完成 CLIP 和文本匹配的任务进入此统计；无目标而跳过的任务不算 completed。
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
    // 每个 image_index 只保留首条样例；视频/摄像头都用 -1，所以这里只是样例，不是逐帧日志。
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

inline void PrintReport(const Options& opt, const Stats& stats, const std::vector<std::string>& image_paths,
                 double wall_ms) {
  const int completed = stats.completed.load();
  const int skipped_no_object = stats.skipped_no_object.load();
  // processed 包括无目标而跳过 CLIP 的帧；completed_fps 才表示完整两模型链路的吞吐。
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

  // 调用方在 worker 全部 join 后打印；wall_ms 包含排空队列时间，不能用 1000/FPS 当单请求延迟。
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3);
  oss << "{\n";
  oss << "  \"mode\": \"" << (opt.serial ? "serial" : "two_stage_pool") << "\",\n";
  oss << "  \"preprocess_backend\": \"" << opt.preprocess_backend << "\",\n";
  oss << "  \"clip_core_mode\": \"" << opt.clip_core_mode << "\",\n";
  oss << "  \"yolo_core_mode\": \"" << opt.yolo_core_mode << "\",\n";
  oss << "  \"npu_gate\": \"" << opt.npu_gate << "\",\n";
  oss << "  \"yolo_npu_gate_wait_ms_mean\": " << (yolo_runs ? stats.yolo_gate_wait_ms_sum / yolo_runs : 0) << ",\n";
  oss << "  \"clip_npu_gate_wait_ms_mean\": " << (clip_runs ? stats.clip_gate_wait_ms_sum / clip_runs : 0) << ",\n";
  oss << "  \"use_io_mem\": " << (opt.use_io_mem ? "true" : "false") << ",\n";
  oss << "  \"zero_copy_input\": " << (opt.zero_copy_input ? "true" : "false") << ",\n";
  oss << "  \"camera_device\": \"" << JsonEscape(opt.camera_device) << "\",\n";
  oss << "  \"camera_format\": \"" << opt.camera_format << "\",\n";
  oss << "  \"camera_path\": \"" << opt.camera_path << "\",\n";
  oss << "  \"normalization_mode\": \"" << opt.normalization_mode << "\",\n";
  oss << "  \"opencv_threads\": " << opt.opencv_threads << ",\n";
  oss << "  \"explicit_cpu_copy_count\": " << (copy_metrics.raw_copies.load() + copy_metrics.letterbox_copies.load() + copy_metrics.input_copies.load()) << ",\n";
  oss << "  \"explicit_cpu_copy_bytes\": " << (copy_metrics.raw_bytes.load() + copy_metrics.letterbox_bytes.load() + copy_metrics.input_copy_bytes.load()) << ",\n";
  oss << "  \"raw_copy_bytes\": " << copy_metrics.raw_bytes.load() << ",\n";
  oss << "  \"letterbox_copy_bytes\": " << copy_metrics.letterbox_bytes.load() << ",\n";
  oss << "  \"host_input_submissions\": " << copy_metrics.host_submissions.load() << ",\n";
  oss << "  \"host_input_submission_bytes\": " << copy_metrics.host_submission_bytes.load() << ",\n";
  oss << "  \"normalization_output_bytes\": " << copy_metrics.normalization_bytes.load() << ",\n";
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


}  // namespace yolo_clip
