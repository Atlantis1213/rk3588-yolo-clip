// 通用数学与报告辅助；当前部署归一化使用 normalization.hpp 中的转换实现。

#pragma once
#include "common.hpp"

namespace yolo_clip {

// 按值接收后排序，不改变统计中的原始样本；使用 ceil 选取上侧顺序统计量。
inline double Percentile(std::vector<double> samples, double percentile) {
  if (samples.empty()) {
    return 0.0;
  }
  std::sort(samples.begin(), samples.end());
  const size_t index = static_cast<size_t>(std::ceil(percentile * (samples.size() - 1)));
  return samples[index];
}

// 保留的历史 FP16 转换，舍入进位行为不同于 normalization::Float32ToFloat16。
// 阅读当前输入归一化时应查看后者，勿将此辅助函数当作 IEEE 参考实现。
inline uint16_t Float32ToFloat16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  const uint32_t mantissa = bits & 0x007fffffu;
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

// 按 FP32 位模式滚动计算校验和，适合发现输出变化；不是准确率或密码学散列。
inline uint64_t ChecksumFloat(const float* data, size_t count) {
  uint64_t sum = 0;
  for (size_t i = 0; i < count; ++i) {
    uint32_t bits;
    std::memcpy(&bits, &data[i], sizeof(bits));
    sum = sum * 131 + bits;
  }
  return sum;
}

inline float Dot(const float* a, const float* b, size_t count) {
  float sum = 0.0f;
  for (size_t i = 0; i < count; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

// 给平方范数设置下限，避免接近零的向量归一化时直接除零。
inline float L2Norm(const float* data, size_t count) {
  return std::sqrt(std::max(1e-12f, Dot(data, data, count)));
}

inline std::string JsonEscape(const std::string& value) {
  std::ostringstream oss;
  for (char ch : value) {
    switch (ch) {
      case '\\':
        oss << "\\\\";
        break;
      case '"':
        oss << "\\\"";
        break;
      case '\n':
        oss << "\\n";
        break;
      case '\r':
        oss << "\\r";
        break;
      case '\t':
        oss << "\\t";
        break;
      default:
        oss << ch;
        break;
    }
  }
  return oss.str();
}

inline std::string Basename(const std::string& path) {
  const auto pos = path.find_last_of("/\\");
  return pos == std::string::npos ? path : path.substr(pos + 1);
}


}  // namespace yolo_clip
