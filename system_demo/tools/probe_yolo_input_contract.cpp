#include "pipeline.hpp"
#include <filesystem>

class ProbeYolo : public yolo_clip::BaseRunner {
 public:
  void Submit(void* bytes, size_t size, rknn_tensor_type type, bool pass_through) {
    rknn_input input{};
    input.index = 0; input.buf = bytes; input.size = static_cast<uint32_t>(size);
    input.type = type; input.fmt = RKNN_TENSOR_NHWC; input.pass_through = pass_through;
    if (rknn_inputs_set(ctx_, 1, &input) != RKNN_SUCC || rknn_run(ctx_, nullptr) != RKNN_SUCC)
      throw std::runtime_error("YOLO input probe submission/run failed");
    rknn_output output{};
    output.index = 0; output.want_float = 1; output.is_prealloc = 1;
    output.buf = output_float_.data(); output.size = static_cast<uint32_t>(output_float_.size() * sizeof(float));
    if (rknn_outputs_get(ctx_, 1, &output, nullptr) != RKNN_SUCC)
      throw std::runtime_error("YOLO input probe output failed");
    if (rknn_outputs_release(ctx_, 1, &output) != RKNN_SUCC)
      throw std::runtime_error("YOLO input probe release failed");
  }
  void PrintBest(std::ostream& stream) {
    const int count = static_cast<int>(output_float_.size() / 5);
    int best = 0;
    for (int i = 1; i < count; ++i) if (output_float_[4 * count + i] > output_float_[4 * count + best]) best = i;
    stream << "\"max_confidence\":" << output_float_[4 * count + best] << ",\"box_xywh\":[";
    for (int c = 0; c < 4; ++c) { if (c) stream << ','; stream << output_float_[c * count + best]; }
    stream << ']';
  }
};

int main(int argc, char** argv) {
  try {
    if (argc != 4) throw std::runtime_error("probe_yolo_input_contract YOLO DATA_DIR OUTPUT_FILE");
    using namespace yolo_clip;
    const std::filesystem::path data(argv[2]), output(argv[3]);
    if (std::filesystem::exists(output)) throw std::runtime_error("probe output exists; preserve evidence");
    cv::setNumThreads(1);
    ProbeYolo yolo;
    yolo.InitFromModel(ReadBinaryFile(argv[1]), RKNN_NPU_CORE_0);
    std::ofstream report(output);
    report << std::setprecision(10) << "{\"rows\":[";
    bool first = true;
    for (const auto& category : {"cycle", "harmful", "kitchen", "others", "empty"}) {
      for (int index = 0; index < 2; ++index) {
        const std::string name = "real_" + std::string(category) + (index == 0 ? "_000.jpg" : "_001.jpg");
        const auto image = cv::imread((data / "images" / name).string());
        if (image.empty()) throw std::runtime_error("failed to read probe image " + name);
        const int side = std::max(image.rows, image.cols);
        cv::Mat canvas(side, side, CV_8UC3, cv::Scalar(114, 114, 114)), rgb;
        image.copyTo(canvas(cv::Rect(0, 0, image.cols, image.rows)));
        cv::resize(canvas, rgb, cv::Size(640, 640));
        cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
        std::vector<uint16_t> normalized(640 * 640 * 3), raw(normalized.size());
        for (size_t i = 0; i < normalized.size(); ++i) {
          normalized[i] = normalization::Float32ToFloat16(rgb.data[i] / 255.f);
          raw[i] = normalization::Float32ToFloat16(static_cast<float>(rgb.data[i]));
        }
        for (const auto& mode : {"normalized_fp16_host", "raw_fp16_host", "raw_u8_host", "normalized_fp16_native"}) {
          const std::string setting(mode);
          const bool u8 = setting == "raw_u8_host", native = setting == "normalized_fp16_native";
          void* pointer = u8 ? static_cast<void*>(rgb.data) : setting == "raw_fp16_host"
              ? static_cast<void*>(raw.data()) : static_cast<void*>(normalized.data());
          const size_t bytes = u8 ? normalized.size() : normalized.size() * sizeof(uint16_t);
          yolo.Submit(pointer, bytes, u8 ? RKNN_TENSOR_UINT8 : RKNN_TENSOR_FLOAT16, native);
          if (!first) report << ',';
          first = false;
          report << "{\"image\":\"" << name << "\",\"mode\":\"" << setting << "\",";
          yolo.PrintBest(report); report << '}';
        }
      }
    }
    report << "]}\n";
    if (!report) throw std::runtime_error("failed to save probe report");
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
