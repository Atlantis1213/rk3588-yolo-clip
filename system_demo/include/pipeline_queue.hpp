// 有界先进先出队列：满时移除最旧的等待任务，以限制实时流水线的积压。
// 只丢弃尚未被 worker 取走的任务，不会取消正在运行的推理。

#pragma once
#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <utility>

namespace yolo_clip {
template <typename T>
class DropOldestQueue {
 public:
  explicit DropOldestQueue(size_t capacity) : capacity_(capacity) {
    if (!capacity) throw std::invalid_argument("queue capacity must be positive");
  }
  // 按值接收使队列拥有任务；关闭后 Push 返回 false，局部任务析构会释放所持资源。
  template <typename DropFn>
  bool Push(T item, DropFn on_drop) {
    // 把被丢任务移到锁外持有，避免析构租约、归还摄像头帧时在队列锁内获取其他锁。
    std::optional<T> dropped;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (closed_) return false;
      if (queue_.size() >= capacity_) { dropped.emplace(std::move(queue_.front())); queue_.pop(); }
      queue_.push(std::move(item));
    }
    // 先释放锁再通知；外部回调也在锁外执行，降低锁嵌套和阻塞的风险。
    cv_.notify_one();
    if (dropped) on_drop(*dropped);  // External callbacks/destructors run outside the queue lock.
    return true;
  }
  // 谓词等待可抵御虚假唤醒；关闭后仍可排空已有任务，关闭且为空才返回 false。
  bool Pop(T& item) {
    std::optional<T> popped;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [&] { return closed_ || !queue_.empty(); });
      if (queue_.empty()) return false;
      popped.emplace(std::move(queue_.front())); queue_.pop();
    }
    // 赋值可能释放 item 原先持有的资源，因此同样放在锁外。
    item = std::move(*popped);
    return true;
  }
  // 关闭是幂等的；notify_all 让所有等待消费者都能观察到退出条件。
  void Close() {
    { std::lock_guard<std::mutex> lock(mutex_); closed_ = true; }
    cv_.notify_all();
  }
 private:
  size_t capacity_;
  bool closed_ = false;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::queue<T> queue_;
};
}  // namespace yolo_clip
