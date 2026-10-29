#pragma once
#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace yolo_clip {

// Optional application admission control. It does not change kernel priorities.
class NpuArbiter {
 public:
  class Permit {
   public:
    Permit() = default;
    ~Permit() { Reset(); }
    Permit(const Permit&) = delete;
    Permit& operator=(const Permit&) = delete;
    Permit(Permit&& other) noexcept : owner_(other.owner_), mask_(other.mask_) { other.owner_ = nullptr; }
    Permit& operator=(Permit&& other) noexcept {
      if (this != &other) { Reset(); owner_ = other.owner_; mask_ = other.mask_; other.owner_ = nullptr; }
      return *this;
    }
    void Reset() noexcept {
      if (owner_) { owner_->Release(mask_); owner_ = nullptr; }
    }
    uint32_t Mask() const { return mask_; }
   private:
    friend class NpuArbiter;
    Permit(NpuArbiter* owner, uint32_t mask) : owner_(owner), mask_(mask) {}
    NpuArbiter* owner_ = nullptr;
    uint32_t mask_ = 0;
  };
  void Enable(bool enabled) { enabled_ = enabled; }
  Permit Acquire(uint32_t mask) {
    if (!enabled_) return {};
    if (!mask) mask = 7;  // AUTO may use any core; reserve all conservatively.
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return !(active_ & mask) && pending_any_ == 0; });
    active_ |= mask;
    return Permit(this, mask);
  }
  Permit AcquireAnyCore() {
    std::unique_lock<std::mutex> lock(mutex_);
    ++pending_any_;
    cv_.wait(lock, [&] { return active_ != 7; });
    --pending_any_;
    for (int offset = 0; offset < 3; ++offset) {
      const int core = (next_core_ + offset) % 3;
      const uint32_t bit = 1u << core;
      if (!(active_ & bit)) {
        active_ |= bit;
        next_core_ = (core + 1) % 3;
        cv_.notify_all(); // Wake disjoint static waiters after pending-any is consumed.
        return Permit(this, bit);
      }
    }
    std::terminate();
  }
 private:
  void Release(uint32_t mask) noexcept {
    { std::lock_guard<std::mutex> lock(mutex_); active_ &= ~mask; }
    cv_.notify_all();
  }
  bool enabled_ = false;
  uint32_t active_ = 0;
  int pending_any_ = 0, next_core_ = 0;
  std::mutex mutex_;
  std::condition_variable cv_;
};
inline NpuArbiter npu_arbiter;
}  // namespace yolo_clip
