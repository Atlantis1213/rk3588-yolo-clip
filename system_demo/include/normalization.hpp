#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace yolo_clip::normalization {
enum class Mode { Scalar, Lut, ScalarLegacy, LutLegacy };
inline Mode ParseMode(const std::string& value) {
  if (value == "scalar") return Mode::Scalar;
  if (value == "lut") return Mode::Lut;
  if (value == "scalar_legacy") return Mode::ScalarLegacy;
  if (value == "lut_legacy") return Mode::LutLegacy;
  throw std::invalid_argument("normalization must be scalar, lut, scalar_legacy or lut_legacy");
}
inline uint16_t LegacyFloat32ToFloat16(float value) {
  uint32_t bits; std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u, mantissa = bits & 0x007fffffu;
  int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
  if (exponent <= 0) {
    if (exponent < -10) return static_cast<uint16_t>(sign);
    const uint32_t shifted = (mantissa | 0x00800000u) >> (1 - exponent);
    return static_cast<uint16_t>(sign | ((shifted + 0x00001000u) >> 13));
  }
  if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) |
                               ((mantissa + 0x00001000u) >> 13));
}
// IEEE binary16, round-to-nearest ties-to-even, with explicit exponent carry.
inline uint16_t Float32ToFloat16(float value) {
  uint32_t bits; std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  const uint32_t raw_exp = (bits >> 23) & 255u, mantissa = bits & 0x7fffffu;
  if (raw_exp == 255) return static_cast<uint16_t>(sign | 0x7c00u | (mantissa ? 0x200u : 0));
  int exponent = static_cast<int>(raw_exp) - 127;
  if (exponent > 15) return static_cast<uint16_t>(sign | 0x7c00u);
  if (exponent < -25) return static_cast<uint16_t>(sign);
  if (exponent >= -14) {
    uint32_t rounded = mantissa + 0xfffu + ((mantissa >> 13) & 1u);
    int half_exp = exponent + 15;
    if (rounded & 0x800000u) { rounded = 0; ++half_exp; }
    if (half_exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (half_exp << 10) | (rounded >> 13));
  }
  const uint32_t full = mantissa | 0x800000u;
  const int shift = -exponent - 1;
  uint32_t rounded = full >> shift;
  const uint32_t remainder = full & ((1u << shift) - 1u), halfway = 1u << (shift - 1);
  if (remainder > halfway || (remainder == halfway && (rounded & 1u))) ++rounded;
  return static_cast<uint16_t>(sign | rounded);
}
inline float Float16ToFloat32(uint16_t value) {
  const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
  uint32_t exponent = (value >> 10) & 31u, mantissa = value & 1023u, bits;
  if (exponent == 0) {
    if (mantissa == 0) bits = sign;
    else {
      int e = -14;
      while (!(mantissa & 1024u)) { mantissa <<= 1; --e; }
      bits = sign | (static_cast<uint32_t>(e + 127) << 23) | ((mantissa & 1023u) << 13);
    }
  } else if (exponent == 31) bits = sign | 0x7f800000u | (mantissa << 13);
  else bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
  float result; std::memcpy(&result, &bits, sizeof(result)); return result;
}
inline float Value(uint8_t pixel, int channel, bool clip) {
  static constexpr float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
  static constexpr float stdv[3] = {0.26862954f, 0.26130258f, 0.27577711f};
  float value = pixel / 255.0f;
  return clip ? (value - mean[channel]) / stdv[channel] : value;
}
struct Tables {
  std::array<std::array<uint16_t, 256>, 3> yolo{}, clip{};
  std::array<std::array<uint16_t, 256>, 3> legacy_yolo{}, legacy_clip{};
  Tables() {
    for (int c = 0; c < 3; ++c) for (int v = 0; v < 256; ++v) {
      yolo[c][v] = Float32ToFloat16(Value(static_cast<uint8_t>(v), c, false));
      clip[c][v] = Float32ToFloat16(Value(static_cast<uint8_t>(v), c, true));
      legacy_yolo[c][v] = LegacyFloat32ToFloat16(Value(static_cast<uint8_t>(v), c, false));
      legacy_clip[c][v] = LegacyFloat32ToFloat16(Value(static_cast<uint8_t>(v), c, true));
    }
  }
};
inline void Validate(const uint8_t* source, int width, int height, size_t stride, uint16_t* output) {
  if (!source || !output || width <= 0 || height <= 0 || stride < static_cast<size_t>(width) * 3)
    throw std::invalid_argument("invalid RGB normalization shape/stride");
}
// Named noinline functions remain visible to profilers; no fast-math is used.
inline __attribute__((noinline)) void Scalar(const uint8_t* source, int width, int height,
                                           size_t stride, uint16_t* output, bool clip, bool legacy = false) {
  Validate(source, width, height, stride, output);
  for (int y = 0; y < height; ++y) {
    const uint8_t* row = source + y * stride;
    for (int x = 0; x < width; ++x) for (int c = 0; c < 3; ++c)
      *output++ = legacy ? LegacyFloat32ToFloat16(Value(row[3 * x + c], c, clip)) :
                          Float32ToFloat16(Value(row[3 * x + c], c, clip));
  }
}
inline __attribute__((noinline)) void Lut(const uint8_t* source, int width, int height,
                                        size_t stride, uint16_t* output, bool clip, bool legacy = false) {
  Validate(source, width, height, stride, output);
  static const Tables tables;
  const auto& table = legacy ? (clip ? tables.legacy_clip : tables.legacy_yolo) : (clip ? tables.clip : tables.yolo);
  for (int y = 0; y < height; ++y) {
    const uint8_t* row = source + y * stride;
    for (int x = 0; x < width; ++x) {
      *output++ = table[0][row[3 * x]];
      *output++ = table[1][row[3 * x + 1]];
      *output++ = table[2][row[3 * x + 2]];
    }
  }
}
inline void Run(const uint8_t* source, int width, int height, size_t stride, uint16_t* output,
                bool clip, Mode mode) {
  if (mode == Mode::Lut || mode == Mode::LutLegacy)
    Lut(source, width, height, stride, output, clip, mode == Mode::LutLegacy);
  else Scalar(source, width, height, stride, output, clip, mode == Mode::ScalarLegacy);
}
}  // namespace yolo_clip::normalization
