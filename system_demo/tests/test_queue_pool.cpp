// 不依赖硬件的队列/缓冲池测试：关注关闭、丢弃、异常时槽位是否完整归还。

#include "buffer_pool.hpp"
#include "pipeline_queue.hpp"
#include <atomic>
#include <future>
#include <iostream>
#include <thread>

using Pool = yolo_clip::BufferPool<int>;
using Queue = yolo_clip::DropOldestQueue<Pool::Lease>;

void Check(bool ok, const char* description) {
  if (!ok) {
    throw std::runtime_error(description);
  }
}

// 重点回归：Push 被关闭队列拒绝时，传入租约也必须析构归还，不能只依赖 on_drop。
__attribute__((noinline)) void TestClosedQueueOwnership() {
  Pool pool(std::vector<int>(1));
  Queue queue(1);
  queue.Close();
  auto lease = pool.TryAcquire();
  Check(!queue.Push(std::move(lease),
                    [](const auto&) {
                    }),
        "closed queue accepted task");
  Check(pool.FreeCount() == 1, "closed queue leaked lease");
}

int main() {
  TestClosedQueueOwnership();
  {
    Pool pool(std::vector<int>(3));
    {
      Queue queue(1);
      int dropped = 0;
      auto first = pool.TryAcquire();
      first.Get() = 11;
      queue.Push(std::move(first), [&](const auto&) {
        ++dropped;
      });
      auto second = pool.TryAcquire();
      second.Get() = 22;
      queue.Push(std::move(second), [&](const auto&) {
        ++dropped;
      });
      Check(dropped == 1 && pool.FreeCount() == 2, "oldest drop did not return exactly one slot");
      queue.Close();
      Pool::Lease consumed;
      Check(queue.Pop(consumed) && consumed.Get() == 22, "wrong queue ordering");
      try {
        throw std::runtime_error("injected consumer failure");
      }
      catch (const std::runtime_error&) {
        consumed.Reset();
      }
    }
    Check(pool.FreeCount() == 3, "consumer/queue cleanup leak");
    try {
      auto lease = pool.TryAcquire();
      throw std::runtime_error("injected producer failure");
    }
    catch (const std::runtime_error&) {
    }
    Check(pool.FreeCount() == 3, "producer exception leaked slot");
  }
  {
    yolo_clip::DropOldestQueue<int> queue(2);
    auto waiter = std::async(std::launch::async, [&] {
      int item;
      return queue.Pop(item);
    });
    queue.Close();
    Check(waiter.wait_for(std::chrono::seconds(1)) == std::future_status::ready && !waiter.get(),
          "close did not wake empty consumer");
  }
  {
    Pool pool(std::vector<int>(16));
    std::atomic<int> accepted{0}, dropped{0}, consumed{0};
    {
      Queue queue(8);
      std::vector<std::thread> consumers;
      for (int i = 0; i < 4; ++i) {
        consumers.emplace_back([&] {
          Pool::Lease item;
          while (queue.Pop(item)) {
            ++consumed;
            item.Reset();
          }
        });
      }
      std::vector<std::thread> producers;
      for (int i = 0; i < 2; ++i) {
        producers.emplace_back([&] {
          for (int n = 0; n < 2000; ++n) {
            Pool::Lease item;
            while (!(item = pool.TryAcquire())) {
              std::this_thread::yield();
            }
            if (queue.Push(std::move(item), [&](const auto&) {
                  ++dropped;
                })) {
              ++accepted;
            }
          }
        });
      }
      for (auto& thread : producers) {
        thread.join();
      }
      queue.Close();
      for (auto& thread : consumers) {
        thread.join();
      }
    }
    // 并发对账要求每个被接受任务最终恰好属于消费或丢弃，并在退出时收回全部槽。
    Check(accepted == 4000 && consumed + dropped == accepted,
          "concurrent task accounting mismatch");
    Check(pool.FreeCount() == 16, "concurrent slots not all returned");
  }
  {
    // 验证租约可比池包装对象存活更久，内部 shared State 没有悬空。
    Pool::Lease survivor;
    {
      Pool pool(std::vector<int>{42});
      survivor = pool.TryAcquire();
    }
    Check(survivor.Get() == 42, "lease failed to retain pool lifetime");
  }
  std::cout
      << "PASS: close/drop/exception ownership, wakeup, 4000 concurrent tasks, retained lifetime\n";
}
