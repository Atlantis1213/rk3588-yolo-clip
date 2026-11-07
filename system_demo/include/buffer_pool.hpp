// 固定容量的线程安全缓冲池：一次分配、反复借用，用于 CLIP 输入张量。
// Lease 用 RAII 管理归还责任：离开作用域、丢帧、异常都会自动归还。

#pragma once
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace yolo_clip {
template <typename T>
class BufferPool {
  // 池包装对象和所有租约共享 State，因此即使包装对象先销毁，未归还租约仍可安全访问槽。
  struct State {
    explicit State(std::vector<T> initial) : slots(std::move(initial)) {
      // 归还时只 push 已预留容量的索引，使 noexcept 的释放路径无需再次分配内存。
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
    // 禁止复制，确保一个槽只有一个归还者；移动只转移责任，不复制张量。
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
    // 先移出 state_ 使当前租约失效；重复 Reset 不会重复归还同一个索引。
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
  // 非阻塞借用：没空闲槽就返回空租约，由调用者选择丢帧；锁只保护空闲索引表。
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
