#pragma once
#include "types.hpp"

namespace yolo_clip {

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


}  // namespace yolo_clip
