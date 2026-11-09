// 离线文本特征库：TSV 每行对应一个 label/category，二进制文件按同一行顺序保存 FP32 特征。
// 加载后只读，可被所有 CLIP worker 共享；每次分类的临时数组为局部变量。

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
      // 左列是细粒度文本标签，右列是部署用粗类别；多条细标签可以映射到同一个粗类别。
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
    // 二进制无自带形状元数据，通过标签行数推导特征维度，因此标签顺序必须与生成时一致。
    dim_ = total_floats / labels_.size();
    features_.resize(total_floats);
    std::memcpy(features_.data(), feature_bytes.data(), feature_bytes.size());

    // 文本特征在加载时归一化一次；单位文本特征与单位图像特征点积就是余弦相似度。
    for (size_t row = 0; row < labels_.size(); ++row) {
      // 二维 [标签数,dim_] 被平铺到 vector；row*dim_ 正好跳到这一标签的特征首元素。
      float* ptr = features_.data() + row * dim_;
      const float norm = L2Norm(ptr, dim_);
      for (size_t i = 0; i < dim_; ++i) {
        ptr[i] /= norm;
      }
    }
  }

  ClassificationResult Classify(const float* embedding,
                                size_t count,
                                int topk,
                                float reject_threshold) const {
    if (dim_ == 0 || count != dim_) {
      throw std::runtime_error("embedding dim mismatch");
    }

    // 复制图像 embedding 后归一化，保留 runner 输出原值供校验和其他诊断使用。
    std::vector<float> normalized(count);
    std::memcpy(normalized.data(), embedding, count * sizeof(float));
    const float norm = L2Norm(normalized.data(), count);
    // 向量模长不代表类别置信度；先除以模长，只比较图像与文本在共同语义空间中的方向。
    for (size_t i = 0; i < count; ++i) {
      normalized[i] /= norm;
    }

    std::vector<std::pair<float, int>> scored;
    scored.reserve(labels_.size());
    // 文本已在加载时归一化；逐行点积后按余弦分数降序排列。
    for (size_t row = 0; row < labels_.size(); ++row) {
      const float* text = features_.data() + row * dim_;
      scored.emplace_back(Dot(text, normalized.data(), dim_), static_cast<int>(row));
    }
    std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
      return a.first > b.first;
    });

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
      // 先选择最高分细标签，再读取它的粗类别；此处没有按粗类别聚合多个标签的分数。
      result.top1_label = labels_[idx];
      result.mapped_category = categories_[idx];
      // 拒识阈值作用于最高余弦分数；topk 原始排名保留，便于检查被拒识样本。
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
