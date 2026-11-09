// 故意保留的历史所有权泄漏复现程序；预期以非零状态退出，不属于应通过的 CTest。
// 关闭队列直接返回，不调用 on_drop；只传整数槽号时，手动归还责任因此丢失。

#include <condition_variable>
#include <mutex>
#include <queue>
#include <iostream>
#include <stdexcept>
#include <utility>

template <typename T>
class DropOldestQueue {
 public:
  explicit DropOldestQueue(size_t capacity) : capacity_(capacity) {
  }

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
    cv_.wait(lock, [&] {
      return closed_ || !queue_.empty();
    });
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

__attribute__((noinline)) void ReproduceOriginalClosedQueueLeak() {
  // Original pipeline transferred only an integer slot id, and released via on_drop.
  DropOldestQueue<int> queue(1);
  queue.Close();
  int free_slots = 1;
  --free_slots;  // Original TryAcquire() ownership obligation.
  queue.Push(0, [&](int) {
    ++free_slots;
  });
  if (free_slots != 1) {
    throw std::runtime_error("original closed Push lost manual slot ownership");
  }
}

int main() {
  try {
    ReproduceOriginalClosedQueueLeak();
  }
  catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
  return 0;
}
