// 显式内存流量计数：只覆盖应用中已标记的 raw、letterbox 和 RKNN 输入 memcpy。
// HostSubmission 是提交字节数，不保证 SDK 内部复制量；Normalized 是计算结果写入量。

#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace yolo_clip {
// Counts explicit application copies only. SDK-internal traffic is opaque.
struct CopyMetrics {
  std::atomic<uint64_t> raw_copies{0}, raw_bytes{0};
  std::atomic<uint64_t> letterbox_copies{0}, letterbox_bytes{0};
  std::atomic<uint64_t> input_copies{0}, input_copy_bytes{0};
  std::atomic<uint64_t> host_submissions{0}, host_submission_bytes{0};
  std::atomic<uint64_t> normalization_bytes{0};
  void Raw(size_t bytes) { ++raw_copies; raw_bytes += bytes; }
  void Letterbox(size_t bytes) { ++letterbox_copies; letterbox_bytes += bytes; }
  void InputCopy(size_t bytes) { ++input_copies; input_copy_bytes += bytes; }
  void HostSubmission(size_t bytes) { ++host_submissions; host_submission_bytes += bytes; }
  void Normalized(size_t bytes) { normalization_bytes += bytes; }
};
inline CopyMetrics copy_metrics;
}  // namespace yolo_clip
