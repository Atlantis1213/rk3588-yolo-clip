#include "model_runner.hpp"
#include "input_io.hpp"
#include <atomic>

int main(int argc, char** argv) {
  try {
    if (argc < 6) {
      throw std::runtime_error(
          "bench_clip_concurrency MODEL INPUT_FP16 WORKERS CORE_MODE DURATION [none|per_core] [dma|host]");
    }
    const int workers = std::stoi(argv[3]), duration = std::stoi(argv[5]);
    if (workers <= 0 || workers > 8 || duration <= 0) {
      throw std::runtime_error("invalid worker/duration");
    }
    const std::string mode = argv[4], gate = argc > 6 ? argv[6] : "none";
    const bool dma = argc <= 7 || std::string(argv[7]) == "dma";
    using namespace yolo_clip;
    npu_arbiter.Enable(gate == "per_core");
    const auto model = ReadBinaryFile(argv[1]);
    const auto input = ReadBinaryFile(argv[2]);
    if (input.size() != ClipPreprocessor::InputBytes()) {
      throw std::runtime_error("input tensor size mismatch");
    }
    std::vector<std::unique_ptr<ClipRunner>> runners;
    std::vector<std::shared_ptr<camera_dma::DmaBuffer>> buffers;
    for (int i = 0; i < workers; ++i) {
      auto runner = std::make_unique<ClipRunner>();
      if (dma) {
        runner->EnableDirectInput("/dev/dma_heap/system");
      }
      if (i == 0) {
        runner->InitFromModel(model, ClipCoreMask(i, mode));
      } else {
        runner->InitFromContext(runners[0]->MutableContext(), ClipCoreMask(i, mode));
      }
      if (dma) {
        auto buffer = std::make_shared<camera_dma::DmaBuffer>(
            "/dev/dma_heap/system", runner->InputAllocationBytes(input.size()));
        camera_dma::CpuAccess write(*buffer, DMA_BUF_SYNC_WRITE);
        std::memcpy(buffer->Data(), input.data(), input.size());
        write.Finish();
        buffers.push_back(buffer);
      }
      runners.push_back(std::move(runner));
    }
    std::vector<int> counts(workers, 0);
    std::vector<double> sums(workers, 0), gate_sums(workers, 0);
    std::vector<uint64_t> checksums(workers, 0);
    std::atomic<bool> failed{false};
    std::exception_ptr error;
    std::mutex error_mutex;
    auto run_one = [&](int i) {
      return dma ? runners[i]->RunPreparedDma(buffers[i]) : runners[i]->RunPrepared(input.data());
    };
    // Parallel warmup before timing; context/memory initialization is excluded.
    std::vector<std::thread> warmup;
    for (int i = 0; i < workers; ++i) {
      warmup.emplace_back([&, i] {
        try {
          for (int k = 0; k < 2; ++k) {
            run_one(i);
          }
        }
        catch (...) {
          std::lock_guard<std::mutex> lock(error_mutex);
          error = std::current_exception();
        }
      });
    }
    for (auto& thread : warmup) {
      thread.join();
    }
    if (error) {
      std::rethrow_exception(error);
    }
    const auto start = Clock::now(), deadline = start + std::chrono::seconds(duration);
    std::vector<std::thread> threads;
    for (int i = 0; i < workers; ++i) {
      threads.emplace_back([&, i] {
        try {
          while (!failed.load() && Clock::now() < deadline) {
            auto result = run_one(i);
            ++counts[i];
            sums[i] += result.run_ms;
            gate_sums[i] += result.gate_wait_ms;
            checksums[i] = result.checksum;
          }
        }
        catch (...) {
          failed.store(true);
          std::lock_guard<std::mutex> lock(error_mutex);
          error = std::current_exception();
        }
      });
    }
    for (auto& thread : threads) {
      thread.join();
    }
    if (error) {
      std::rethrow_exception(error);
    }
    const double wall = std::chrono::duration<double>(Clock::now() - start).count();
    const int completed = std::accumulate(counts.begin(), counts.end(), 0);
    std::cout << std::fixed << std::setprecision(4) << "{\"workers\":" << workers
              << ",\"core_mode\":\"" << mode << "\",\"gate\":\"" << gate << "\",\"input_mode\":\""
              << (dma ? "dma" : "host") << "\",\"completed\":" << completed
              << ",\"wall_sec\":" << wall << ",\"fps\":" << completed / wall
              << ",\"workers_detail\":[";
    for (int i = 0; i < workers; ++i) {
      if (i) {
        std::cout << ',';
      }
      std::cout << "{\"index\":" << i << ",\"count\":" << counts[i]
                << ",\"run_ms_mean\":" << (counts[i] ? sums[i] / counts[i] : 0)
                << ",\"gate_ms_mean\":" << (counts[i] ? gate_sums[i] / counts[i] : 0)
                << ",\"checksum\":" << checksums[i] << '}';
    }
    std::cout << "]}\n";
  }
  catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
