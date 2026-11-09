// CPU RGB → 紧凑 NHWC FP16 张量；输入 stride 的单位为字节，输出按 RGB 顺序连续写入。
// YOLO 使用 pixel/255；CLIP 还需按通道减均值、除标准差。LUT 与 scalar 共享公式。

#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace yolo_clip::normalization {
enum class Mode {
  Scalar,
  Lut,
  ScalarLegacy,
  LutLegacy
};

inline Mode ParseMode(const std::string& value) {
  if (value == "scalar") {
    return Mode::Scalar;
  }
  if (value == "lut") {
    return Mode::Lut;
  }
  if (value == "scalar_legacy") {
    return Mode::ScalarLegacy;
  }
  if (value == "lut_legacy") {
    return Mode::LutLegacy;
  }
  throw std::invalid_argument("normalization must be scalar, lut, scalar_legacy or lut_legacy");
}

// 历史转换保留用于复现实验，存在尾数舍入向指数进位的缺陷；部署应使用当前转换。
inline uint16_t LegacyFloat32ToFloat16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u, mantissa = bits & 0x007fffffu;
  int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
  if (exponent <= 0) {
    if (exponent < -10) {
      return static_cast<uint16_t>(sign);
    }
    const uint32_t shifted = (mantissa | 0x00800000u) >> (1 - exponent);
    return static_cast<uint16_t>(sign | ((shifted + 0x00001000u) >> 13));
  }
  if (exponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00u);
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) |
                               ((mantissa + 0x00001000u) >> 13));
}

// IEEE binary16, round-to-nearest ties-to-even, with explicit exponent carry.
// 返回的是 IEEE binary16 位模式，不是 uint16 数值转换；memcpy 避免违反别名规则。
inline uint16_t Float32ToFloat16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  const uint32_t raw_exp = (bits >> 23) & 255u, mantissa = bits & 0x7fffffu;
  // FP32 的特殊指数对应无穷或 NaN；保留符号，并为 NaN 设置半精度尾数。
  if (raw_exp == 255) {
    return static_cast<uint16_t>(sign | 0x7c00u | (mantissa ? 0x200u : 0));
  }
  int exponent = static_cast<int>(raw_exp) - 127;
  if (exponent > 15) {
    return static_cast<uint16_t>(sign | 0x7c00u);
  }
  if (exponent < -25) {
    return static_cast<uint16_t>(sign);
  }
  if (exponent >= -14) {
    // 保留 10 位尾数：0xfff 加最低保留位，实现恰好中点时向偶数舍入。
    uint32_t rounded = mantissa + 0xfffu + ((mantissa >> 13) & 1u);
    int half_exp = exponent + 15;
    // 尾数舍入可能溢出到隐含首位；必须将尾数归零并让指数加一，不能简单按位或拼接。
    if (rounded & 0x800000u) {
      rounded = 0;
      ++half_exp;
    }
    if (half_exp >= 31) {
      return static_cast<uint16_t>(sign | 0x7c00u);
    }
    return static_cast<uint16_t>(sign | (half_exp << 10) | (rounded >> 13));
  }
  // 非正规半精度数没有隐含首位，需要恢复 FP32 隐含位后按更大位移量舍入。
  const uint32_t full = mantissa | 0x800000u;
  const int shift = -exponent - 1;
  uint32_t rounded = full >> shift;
  const uint32_t remainder = full & ((1u << shift) - 1u), halfway = 1u << (shift - 1);
  if (remainder > halfway || (remainder == halfway && (rounded & 1u))) {
    ++rounded;
  }
  return static_cast<uint16_t>(sign | rounded);
}

// 解码半精度位模式：分别处理零/非正规数、正规数和 Inf/NaN，输出对应 FP32。
inline float Float16ToFloat32(uint16_t value) {
  const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
  uint32_t exponent = (value >> 10) & 31u, mantissa = value & 1023u, bits;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      int e = -14;
      // 非正规数没有隐含首位，左移直到首位出现，并同步减小指数以保持数值不变。
      while (!(mantissa & 1024u)) {
        mantissa <<= 1;
        --e;
      }
      bits = sign | (static_cast<uint32_t>(e + 127) << 23) | ((mantissa & 1023u) << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000u | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
  }
  // 把构造好的 IEEE 位模式复制到 float 对象；不能用数值类型转换代替位模式解释。
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

// channel=0/1/2 必须对应 R/G/B；若传入 OpenCV BGR 会使用错误的 CLIP 通道统计。
inline float Value(uint8_t pixel, int channel, bool clip) {
  static constexpr float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
  static constexpr float stdv[3] = {0.26862954f, 0.26130258f, 0.27577711f};
  float value = pixel / 255.0f;
  return clip ? (value - mean[channel]) / stdv[channel] : value;
}

// 8 位像素每通道只有 256 种输入，可预计算最终 FP16 位模式，逐像素只查表。
// 当前 YOLO+CLIP 表共 3 KiB，另有 3 KiB 历史表；这不是近似量化。
struct Tables {
  std::array<std::array<uint16_t, 256>, 3> yolo{}, clip{};
  std::array<std::array<uint16_t, 256>, 3> legacy_yolo{}, legacy_clip{};

  Tables() {
    // 外层分别生成 RGB 三通道，内层枚举 0～255，保存当前与历史两套最终 FP16 编码。
    for (int c = 0; c < 3; ++c) {
      for (int v = 0; v < 256; ++v) {
        yolo[c][v] = Float32ToFloat16(Value(static_cast<uint8_t>(v), c, false));
        clip[c][v] = Float32ToFloat16(Value(static_cast<uint8_t>(v), c, true));
        legacy_yolo[c][v] = LegacyFloat32ToFloat16(Value(static_cast<uint8_t>(v), c, false));
        legacy_clip[c][v] = LegacyFloat32ToFloat16(Value(static_cast<uint8_t>(v), c, true));
      }
    }
  }
};

// 允许源图像行尾有 padding，但每行至少容纳 width×3 字节；目标空间由调用方保证。
inline void Validate(
    const uint8_t* source, int width, int height, size_t stride, uint16_t* output) {
  if (!source || !output || width <= 0 || height <= 0 || stride < static_cast<size_t>(width) * 3) {
    throw std::invalid_argument("invalid RGB normalization shape/stride");
  }
}

// Named noinline functions remain visible to profilers; no fast-math is used.
inline __attribute__((noinline)) void Scalar(const uint8_t* source,
                                             int width,
                                             int height,
                                             size_t stride,
                                             uint16_t* output,
                                             bool clip,
                                             bool legacy = false) {
  Validate(source, width, height, stride, output);
  for (int y = 0; y < height; ++y) {
    // 每行按真实 stride 定位，跳过源图行尾 padding；输出指针持续递增，生成紧凑 NHWC。
    const uint8_t* row = source + y * stride;
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < 3; ++c) {
        *output++ = legacy ? LegacyFloat32ToFloat16(Value(row[3 * x + c], c, clip))
                           : Float32ToFloat16(Value(row[3 * x + c], c, clip));
      }
    }
  }
}

inline __attribute__((noinline)) void Lut(const uint8_t* source,
                                          int width,
                                          int height,
                                          size_t stride,
                                          uint16_t* output,
                                          bool clip,
                                          bool legacy = false) {
  Validate(source, width, height, stride, output);
  // 函数局部 static 只初始化一次，C++11 起初始化是线程安全的；之后所有 worker 只读。
  static const Tables tables;
  const auto& table = legacy ? (clip ? tables.legacy_clip : tables.legacy_yolo)
                             : (clip ? tables.clip : tables.yolo);
  for (int y = 0; y < height; ++y) {
    const uint8_t* row = source + y * stride;
    for (int x = 0; x < width; ++x) {
      // 一个像素依次写 R/G/B 三个半精度值；查表同时完成缩放、标准化和 FP16 编码。
      *output++ = table[0][row[3 * x]];
      *output++ = table[1][row[3 * x + 1]];
      *output++ = table[2][row[3 * x + 2]];
    }
  }
}

// 统一分派保证上层预处理不需要了解 scalar/LUT 与历史舍入实现的细节。
inline void Run(const uint8_t* source,
                int width,
                int height,
                size_t stride,
                uint16_t* output,
                bool clip,
                Mode mode) {
  if (mode == Mode::Lut || mode == Mode::LutLegacy) {
    Lut(source, width, height, stride, output, clip, mode == Mode::LutLegacy);
  } else {
    Scalar(source, width, height, stride, output, clip, mode == Mode::ScalarLegacy);
  }
}
}  // namespace yolo_clip::normalization
