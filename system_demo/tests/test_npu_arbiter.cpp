// NPU 准入的可移植并发测试：不调用 RKNN，仅检查掩码冲突、释放唤醒与许可异常清理。

#include "npu_arbiter.hpp"
#include <future>
#include <iostream>
#include <stdexcept>

int main() {
  using namespace yolo_clip;
  NpuArbiter arbiter; arbiter.Enable(true);
  auto core0 = arbiter.Acquire(1);
  auto core1 = arbiter.Acquire(2); // Disjoint core must not block.
  // 另开线程申请同一核，必须在原许可归还前保持等待；不同核可同时获准。
  auto blocked = std::async(std::launch::async, [&] { auto permit = arbiter.Acquire(1); return true; });
  if (blocked.wait_for(std::chrono::milliseconds(20)) == std::future_status::ready)
    throw std::runtime_error("overlapping core admitted");
  core0.Reset();
  if (blocked.wait_for(std::chrono::seconds(1)) != std::future_status::ready || !blocked.get())
    throw std::runtime_error("release failed to wake waiter");
  core1.Reset();
  try { auto all = arbiter.Acquire(0); throw std::runtime_error("injected error"); }
  catch (const std::runtime_error&) {}
  auto all_again = arbiter.Acquire(7);
  all_again.Reset();
  {
    auto first = arbiter.AcquireAnyCore();
    auto second = arbiter.AcquireAnyCore();
    if (first.Mask() == second.Mask()) throw std::runtime_error("adaptive placement reused busy core");
  }
  std::cout << "PASS: overlapping masks, disjoint cores, wakeup, AUTO and exception release\n";
}
