#pragma once
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace yolo_clip {
template <typename T>
class BufferPool {
  struct State {
    explicit State(std::vector<T> initial) : slots(std::move(initial)) {
      free.reserve(slots.size());
      for (size_t i = slots.size(); i > 0; --i) free.push_back(static_cast<int>(i - 1));
    }
    std::mutex mutex;
    std::vector<T> slots;
    std::vector<int> free;
  };
 public:
  // A lease owns the obligation to return a slot, including exception/closed-queue paths.
  class Lease {
   public:
    Lease() = default;
    ~Lease() { Reset(); }
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    Lease(Lease&& other) noexcept : state_(std::move(other.state_)), id_(other.id_) {}
    Lease& operator=(Lease&& other) noexcept {
      if (this != &other) { Reset(); state_ = std::move(other.state_); id_ = other.id_; }
      return *this;
    }
    explicit operator bool() const { return static_cast<bool>(state_); }
    T& Get() const {
      if (!state_) throw std::logic_error("accessing empty buffer lease");
      return state_->slots[id_];
    }
    int Id() const { return state_ ? id_ : -1; }
    void Reset() noexcept {
      auto state = std::move(state_);
      if (state) {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->free.push_back(id_);  // Reserved capacity; no allocation on release.
      }
    }
   private:
    friend class BufferPool;
    Lease(std::shared_ptr<State> state, int id) : state_(std::move(state)), id_(id) {}
    std::shared_ptr<State> state_;
    int id_ = -1;
  };
  explicit BufferPool(std::vector<T> slots) : state_(std::make_shared<State>(std::move(slots))) {}
  Lease TryAcquire() {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (state_->free.empty()) return {};
    int id = state_->free.back(); state_->free.pop_back();
    return Lease(state_, id);
  }
  size_t FreeCount() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->free.size();
  }
 private:
  std::shared_ptr<State> state_;
};
}  // namespace yolo_clip
