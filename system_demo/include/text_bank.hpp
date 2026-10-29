#pragma once
#include "types.hpp"
#include "input_io.hpp"

namespace yolo_clip {

class TextBank {
 public:
  void Load(const std::string& label_tsv_path, const std::string& feature_bin_path) {
    std::ifstream label_file(label_tsv_path);
    if (!label_file) {
      throw std::runtime_error("failed to open label tsv: " + label_tsv_path);
    }
    labels_.clear();
    categories_.clear();
    std::string line;
    while (std::getline(label_file, line)) {
      if (line.empty()) {
        continue;
      }
      const auto tab = line.find('\t');
      if (tab == std::string::npos) {
        throw std::runtime_error("invalid label tsv line: " + line);
      }
      labels_.push_back(line.substr(0, tab));
      categories_.push_back(line.substr(tab + 1));
    }
    if (labels_.empty()) {
      throw std::runtime_error("empty label bank");
    }

    std::vector<uint8_t> feature_bytes = ReadBinaryFile(feature_bin_path);
    if (feature_bytes.size() % sizeof(float) != 0) {
      throw std::runtime_error("invalid feature file size");
    }
    const size_t total_floats = feature_bytes.size() / sizeof(float);
    if (total_floats % labels_.size() != 0) {
      throw std::runtime_error("feature rows do not match label count");
    }
    dim_ = total_floats / labels_.size();
    features_.resize(total_floats);
    std::memcpy(features_.data(), feature_bytes.data(), feature_bytes.size());

    for (size_t row = 0; row < labels_.size(); ++row) {
      float* ptr = features_.data() + row * dim_;
      const float norm = L2Norm(ptr, dim_);
      for (size_t i = 0; i < dim_; ++i) {
        ptr[i] /= norm;
      }
    }
  }

  ClassificationResult Classify(const float* embedding, size_t count, int topk,
                                float reject_threshold) const {
    if (dim_ == 0 || count != dim_) {
      throw std::runtime_error("embedding dim mismatch");
    }

    std::vector<float> normalized(count);
    std::memcpy(normalized.data(), embedding, count * sizeof(float));
    const float norm = L2Norm(normalized.data(), count);
    for (size_t i = 0; i < count; ++i) {
      normalized[i] /= norm;
    }

    std::vector<std::pair<float, int>> scored;
    scored.reserve(labels_.size());
    for (size_t row = 0; row < labels_.size(); ++row) {
      const float* text = features_.data() + row * dim_;
      scored.emplace_back(Dot(text, normalized.data(), dim_), static_cast<int>(row));
    }
    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    ClassificationResult result;
    const int k = std::min(topk, static_cast<int>(scored.size()));
    result.topk_labels.reserve(k);
    result.scores.reserve(k);
    for (int i = 0; i < k; ++i) {
      result.topk_labels.push_back(labels_[scored[i].second]);
      result.scores.push_back(scored[i].first);
    }
    if (k > 0) {
      const int idx = scored[0].second;
      result.top1_label = labels_[idx];
      result.mapped_category = categories_[idx];
      result.is_unknown = scored[0].first < reject_threshold;
      if (result.is_unknown) {
        result.top1_label = "unknown";
        result.mapped_category = "unknown";
      }
    }
    return result;
  }

 private:
  size_t dim_ = 0;
  std::vector<std::string> labels_;
  std::vector<std::string> categories_;
  std::vector<float> features_;
};


}  // namespace yolo_clip
