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
    if (!capacity) {
      throw std::invalid_argument("queue capacity must be positive");
    }
  }

  // 按值接收使队列拥有任务；关闭后 Push 返回 false，局部任务析构会释放所持资源。
  // 入队协议：返回 true 表示新任务已接收；false 表示队列已经关闭。
  // 队列满只淘汰旧任务，仍接收新任务，优先保留最新画面。
  template <typename DropFn>
  bool Push(T item, DropFn on_drop) {
    // 把被丢任务移到锁外持有，避免析构租约、归还摄像头帧时在队列锁内获取其他锁。
    std::optional<T> dropped;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      // 关闭检查与修改队列共用一把锁，避免 Close 与 Push 交错后继续接收任务。
      if (closed_) {
        return false;
      }
      // 先把旧任务移入 dropped，再弹出队首。
      // move 转移任务持有的资源，不复制像素或输入张量；资源在锁外才真正释放。
      if (queue_.size() >= capacity_) {
        dropped.emplace(std::move(queue_.front()));
        queue_.pop();
      }
      // 腾出容量后移动新任务入队；从此由队列持有它的 Frame 或 Lease。
      queue_.push(std::move(item));
    }
    // 先释放锁再通知；外部回调也在锁外执行，降低锁嵌套和阻塞的风险。
    cv_.notify_one();
    // 回调只针对因容量被淘汰的旧任务，通常用于计数。
    // 因关闭而拒收的新任务不经过此回调，但按值参数仍会析构并释放资源。
    if (dropped) {
      on_drop(*dropped);  // External callbacks/destructors run outside the queue lock.
    }
    return true;
  }

  // 谓词等待可抵御虚假唤醒；关闭后仍可排空已有任务，关闭且为空才返回 false。
  bool Pop(T& item) {
    std::optional<T> popped;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      // wait 等待时会释放 mutex，让生产者能入队；被唤醒后重新持锁并检查谓词。
      // 即使收到虚假唤醒，也不会在空队列上读取 front。
      cv_.wait(lock, [&] {
        return closed_ || !queue_.empty();
      });
      // 此时已满足唤醒谓词；仍为空说明队列已关闭且没有剩余任务。
      if (queue_.empty()) {
        return false;
      }
      // 先转移队首对象，再移除队列中的空壳；popped 的生命周期延伸到锁作用域之外。
      popped.emplace(std::move(queue_.front()));
      queue_.pop();
    }
    // 赋值可能释放 item 原先持有的资源，因此同样放在锁外。
    item = std::move(*popped);
    return true;
  }

  // 关闭是幂等的；notify_all 让所有等待消费者都能观察到退出条件。
  void Close() {
    // 额外的大括号明确限定锁的生命周期：先写关闭标志并解锁，再唤醒所有消费者。
    {
      std::lock_guard<std::mutex> lock(mutex_);
      closed_ = true;
    }
    cv_.notify_all();
  }

 private:
  // 以下状态均由 mutex_ 保护；capacity_ 构造后固定，closed_ 从 false 单向变为 true。
  size_t capacity_;
  bool closed_ = false;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::queue<T> queue_;
};
}  // namespace yolo_clip
