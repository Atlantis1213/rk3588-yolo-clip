// 应用层 NPU 核准入控制：用 3 个 bit 表示 RK3588 的三个 NPU 核。
// Permit 生命周期覆盖推理请求；互不相交的掩码可并行，占用冲突的请求需等待。

#pragma once
#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace yolo_clip {

// Optional application admission control. It does not change kernel priorities.
class NpuArbiter {
 public:
  // Permit 不可复制；析构释放已占用核并唤醒等待者，异常路径也能归还配额。
  class Permit {
   public:
    Permit() = default;

    ~Permit() {
      Reset();
    }

    Permit(const Permit&) = delete;
    Permit& operator=(const Permit&) = delete;

    // 移动许可后把旧 owner_ 清空，保证一次请求的核掩码只会释放一次。
    Permit(Permit&& other) noexcept : owner_(other.owner_), mask_(other.mask_) {
      other.owner_ = nullptr;
    }

    Permit& operator=(Permit&& other) noexcept {
      if (this != &other) {
        Reset();
        owner_ = other.owner_;
        mask_ = other.mask_;
        other.owner_ = nullptr;
      }
      return *this;
    }

    void Reset() noexcept {
      // 默认空许可无需释放；有效许可先归还核，再清空所有者，允许显式 Reset 后安全析构。
      if (owner_) {
        owner_->Release(mask_);
        owner_ = nullptr;
      }
    }

    uint32_t Mask() const {
      return mask_;
    }

   private:
    friend class NpuArbiter;

    Permit(NpuArbiter* owner, uint32_t mask) : owner_(owner), mask_(mask) {
    }

    NpuArbiter* owner_ = nullptr;
    uint32_t mask_ = 0;
  };

  // 启动 worker 前设置开关；此配置接口不支持运行中并发切换。
  void Enable(bool enabled) {
    enabled_ = enabled;
  }

  // 固定核请求；关闭 gate 时返回空许可，AUTO 掩码保守地按三个核全部预留。
  Permit Acquire(uint32_t mask) {
    if (!enabled_) {
      return {};
    }
    // 二进制 111 表示三个核全部占用；AUTO 无法预知实际选核，因此保守预留全部。
    if (!mask) {
      mask = 7;  // AUTO may use any core; reserve all conservatively.
    }
    std::unique_lock<std::mutex> lock(mutex_);
    // 既要目标核空闲，也要没有等待中的自适应请求；后者优先取得可用核。
    cv_.wait(lock, [&] {
      return !(active_ & mask) && pending_any_ == 0;
    });
    // 按位或标记本请求使用的核；此更新与空闲检查都在 mutex_ 内完成。
    active_ |= mask;
    return Permit(this, mask);
  }

  // 调用方须已启用 per_core（由参数解析保证）；轮转寻找空闲核。
  // 这是进程内准入，不是内核抢占调度，也不提供严格实时或 FIFO 保证。
  Permit AcquireAnyCore() {
    std::unique_lock<std::mutex> lock(mutex_);
    // 登记自适应请求，阻止新的固定掩码请求抢先准入；当前运行中的请求不会被抢占。
    ++pending_any_;
    // active_ 不等于 111 表示至少一个核空闲；等待时释放锁，让已有请求能够归还。
    cv_.wait(lock, [&] {
      return active_ != 7;
    });
    --pending_any_;
    for (int offset = 0; offset < 3; ++offset) {
      const int core = (next_core_ + offset) % 3;
      // 将核编号 0/1/2 映射为 001/010/100，再与 active_ 按位与检查是否忙碌。
      const uint32_t bit = 1u << core;
      if (!(active_ & bit)) {
        active_ |= bit;
        // 下一次从后一个核开始找，避免总是优先占用 core0；这不等价于严格公平排队。
        next_core_ = (core + 1) % 3;
        cv_.notify_all();  // Wake disjoint static waiters after pending-any is consumed.
        return Permit(this, bit);
      }
    }
    // 在锁内确认有空闲核后必能找到一个；执行到此处意味着内部状态不变量遭到破坏。
    std::terminate();
  }

 private:
  void Release(uint32_t mask) noexcept {
    // 按位取反后与 active_ 相与，只清除本许可的核，不影响其他并行请求。
    {
      std::lock_guard<std::mutex> lock(mutex_);
      active_ &= ~mask;
    }
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
