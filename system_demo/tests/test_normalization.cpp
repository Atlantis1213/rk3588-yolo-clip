// 独立验证 FP16 转换和 LUT：逐通道遍历像素值，同时测试带 padding 的行与输出尾部哨兵。

#include "normalization.hpp"
#include <algorithm>
#include <iostream>
#include <vector>

void Check(bool ok, const char* description) {
  if (!ok) {
    throw std::runtime_error(description);
  }
}
#if defined(__aarch64__) || defined(__arm__)
using HalfReference = __fp16;
#else
using HalfReference = _Float16;
#endif
int main() {
  using namespace yolo_clip::normalization;
  // Exhaustive per-channel byte coverage, both YOLO and CLIP; padding/tail coverage.
  for (int width : {1, 7, 256, 640}) {
    for (bool clip : {false, true}) {
      const int height = 3;
      size_t stride = static_cast<size_t>(width) * 3 + 17;
      std::vector<uint8_t> rgb(stride * height, 0xff);
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          for (int c = 0; c < 3; ++c) {
            rgb[y * stride + x * 3 + c] = static_cast<uint8_t>(x + y * 19 + c * 71);
          }
        }
      }
      std::vector<uint16_t> scalar(width * height * 3 + 2, 0xdead), lut(scalar);
      Scalar(rgb.data(), width, height, stride, scalar.data(), clip);
      Lut(rgb.data(), width, height, stride, lut.data(), clip);
      Check(scalar == lut, "LUT differs from original scalar bits or overwrote tail");
    }
  }
  const Tables tables;
  for (int c = 0; c < 3; ++c) {
    for (int v = 0; v < 256; ++v) {
      Check(tables.yolo[c][v] == Float32ToFloat16(Value(v, c, false)), "YOLO LUT mismatch");
      Check(tables.clip[c][v] == Float32ToFloat16(Value(v, c, true)), "CLIP LUT mismatch");
      for (bool clip : {false, true}) {
        const float value = Value(v, c, clip);
        // 使用编译器的半精度类型作独立参考，避免 scalar/LUT 共享同一个 bug 却互相通过。
        const HalfReference half = static_cast<HalfReference>(value);
        uint16_t reference;
        std::memcpy(&reference, &half, sizeof(reference));
        Check(Float32ToFloat16(value) == reference,
              "conversion disagrees with independent IEEE binary16 reference");
      }
    }
  }
  // 固定历史指数进位缺陷的复现点，同时确认当前转换得到正确位模式。
  Check(LegacyFloat32ToFloat16(Value(250, 1, true)) == 0x3c00, "legacy bug reproduction changed");
  Check(Float32ToFloat16(Value(250, 1, true)) == 0x4000, "exponent-carry regression not fixed");
  for (float value : {0.f,
                      -0.f,
                      1.9998449f,
                      -1.9998449f,
                      65504.f,
                      65520.f,
                      0x1p-24f,
                      0x1p-25f,
                      0x1.8p-25f,
                      -0x1p-24f}) {
    const HalfReference half = static_cast<HalfReference>(value);
    uint16_t reference;
    std::memcpy(&reference, &half, sizeof(reference));
    Check(Float32ToFloat16(value) == reference, "IEEE boundary/tie/subnormal mismatch");
    const float decoded = Float16ToFloat32(reference);
    Check(Float32ToFloat16(decoded) == reference, "half decode roundtrip mismatch");
  }
  bool rejected = false;
  try {
    Scalar(nullptr, 1, 1, 3, nullptr, false);
  }
  catch (const std::invalid_argument&) {
    rejected = true;
  }
  Check(rejected, "invalid input not rejected");
  std::cout << "PASS: exhaustive byte/channel equality, padded stride, tails, invalid inputs\n";
}
