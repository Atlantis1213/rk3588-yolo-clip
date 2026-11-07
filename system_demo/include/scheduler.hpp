// CPU 线程亲和性与线程异常传播。CPU core_id 与 RKNN 的 NPU core mask 是两套概念。

#pragma once
#include "types.hpp"

namespace yolo_clip {

// 负数表示交给操作系统调度；亲和性是性能配置，不决定推理正确性。
inline void PinThreadToCore(int core_id) {
  if (core_id < 0) {
    return;
  }
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

inline int ProducerCore(const Options& opt) {
  if (opt.affinity_profile == "legacy") return 7;
  if (opt.affinity_profile == "segregated") return 4;
  return -1;
}

inline int YoloWorkerCore(const Options& opt, int worker_id) {
  if (opt.affinity_profile == "legacy") return worker_id % 4;
  if (opt.affinity_profile == "segregated") return 5 + (worker_id % 3);
  return -1;
}

inline int ClipWorkerCore(const Options& opt, int worker_id) {
  if (opt.affinity_profile == "legacy") return 4 + (worker_id % 4);
  if (opt.affinity_profile == "segregated") return worker_id % 4;
  return -1;
}

// 线程函数不能直接把异常抛回创建线程；这里只保存首个异常并广播停止信号。
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
      // 停止标志配合关闭队列使用：仅设置标志无法唤醒正在条件变量上等待的线程。
      stopped.store(true);
      wake();
    }
  }
  // 主线程在所有 worker join 后调用，此时 error_ 已不再被并发写入。
  void Rethrow() {
    if (error_) std::rethrow_exception(error_);
  }
 private:
  std::mutex mutex_;
  std::exception_ptr error_;
};


}  // namespace yolo_clip
