#include "normalization.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
  using namespace yolo_clip::normalization;
  const auto mode = ParseMode(argc > 1 ? argv[1] : "scalar");
  const int iterations = argc > 2 ? std::stoi(argv[2]) : 300;
  if (iterations <= 0) throw std::invalid_argument("iterations must be positive");
  std::cout << std::fixed << std::setprecision(6) << "[\n";
  bool first = true;
  for (bool clip : {false, true}) {
    const int size = clip ? 224 : 640;
    const size_t stride = size * 3;
    std::vector<uint8_t> rgb(stride * size);
    for (size_t i = 0; i < rgb.size(); ++i) rgb[i] = static_cast<uint8_t>((i * 17 + 31) & 255);
    std::vector<uint16_t> output(size * size * 3);
    for (int n = 0; n < 20; ++n) Run(rgb.data(), size, size, stride, output.data(), clip, mode);
    const auto start = std::chrono::steady_clock::now();
    for (int n = 0; n < iterations; ++n) Run(rgb.data(), size, size, stride, output.data(), clip, mode);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / iterations;
    uint64_t checksum = 0;
    for (auto value : output) checksum = checksum * 131 + value;
    if (!first) std::cout << ",\n";
    first = false;
    std::cout << "{\"model\":\"" << (clip ? "clip" : "yolo") << "\",\"mode\":\""
              << (mode == Mode::Lut ? "lut" : "scalar") << "\",\"iterations\":" << iterations
              << ",\"ms_mean\":" << ms << ",\"checksum\":" << checksum << "}";
  }
  std::cout << "\n]\n";
}
