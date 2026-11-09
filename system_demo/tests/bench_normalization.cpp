// 纯 CPU 归一化微基准：固定 RGB 输入，在 YOLO/CLIP 尺寸上对比 scalar 与 LUT。
// 只测归一化平均耗时，不代表摄像头或 NPU 的端到端吞吐。

#include "normalization.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
  using namespace yolo_clip::normalization;
  const auto mode = ParseMode(argc > 1 ? argv[1] : "scalar");
  const int iterations = argc > 2 ? std::stoi(argv[2]) : 300;
  if (iterations <= 0) {
    throw std::invalid_argument("iterations must be positive");
  }
  std::cout << std::fixed << std::setprecision(6) << "[\n";
  bool first = true;
  for (bool clip : {false, true}) {
    const int size = clip ? 224 : 640;
    const size_t stride = size * 3;
    std::vector<uint8_t> rgb(stride * size);
    for (size_t i = 0; i < rgb.size(); ++i) {
      rgb[i] = static_cast<uint8_t>((i * 17 + 31) & 255);
    }
    std::vector<uint16_t> output(size * size * 3);
    // 预热包括 LUT 首次初始化，避免把一次性成本算进每帧稳态耗时。
    for (int n = 0; n < 20; ++n) {
      Run(rgb.data(), size, size, stride, output.data(), clip, mode);
    }
    const auto start = std::chrono::steady_clock::now();
    for (int n = 0; n < iterations; ++n) {
      Run(rgb.data(), size, size, stride, output.data(), clip, mode);
    }
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                    .count() /
                iterations;
    // 对输出求校验和用于结果对照，并保持计算结果可观察。
    uint64_t checksum = 0;
    for (auto value : output) {
      checksum = checksum * 131 + value;
    }
    if (!first) {
      std::cout << ",\n";
    }
    first = false;
    std::cout << "{\"model\":\"" << (clip ? "clip" : "yolo") << "\",\"mode\":\""
              << (mode == Mode::Lut ? "lut" : "scalar") << "\",\"iterations\":" << iterations
              << ",\"ms_mean\":" << ms << ",\"checksum\":" << checksum << "}";
  }
  std::cout << "\n]\n";
}
