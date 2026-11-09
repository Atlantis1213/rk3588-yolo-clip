// Feed an otherwise FP16 model with quantize/dequantize input to isolate input rounding.
#include "pipeline.hpp"
#include <filesystem>

int main(int argc, char** argv) {
  using namespace yolo_clip;
  try {
    if (argc < 6) {
      throw std::runtime_error(
          "probe_quant_input FP16_MODEL SCALE ZP OUTPUT_DIR INPUT_FP16 [INPUT_FP16 ...]");
    }
    const float scale = std::stof(argv[2]);
    const int zero_point = std::stoi(argv[3]);
    if (!std::isfinite(scale) || scale <= 0) {
      throw std::runtime_error("invalid quantization scale");
    }
    ClipRunner model;
    model.InitFromModel(ReadBinaryFile(argv[1]), RKNN_NPU_CORE_0_1_2);
    const std::filesystem::path output(argv[4]);
    std::filesystem::create_directories(output);
    std::ofstream manifest(output / "manifest.json");
    manifest << std::setprecision(10) << "{\"input_scale\":" << scale
             << ",\"input_zp\":" << zero_point << ",\"embedding_dim\":" << model.OutputCount()
             << ",\"rows\":[";
    bool first = true;
    for (int index = 5; index < argc; ++index) {
      auto bytes = ReadBinaryFile(argv[index]);
      if (bytes.size() != 224 * 224 * 3 * 2) {
        throw std::runtime_error("wrong input tensor size");
      }
      const auto* half = reinterpret_cast<const uint16_t*>(bytes.data());
      std::vector<float> original(bytes.size() / 2), reconstructed(original.size());
      double squared_error = 0;
      for (size_t i = 0; i < original.size(); ++i) {
        original[i] = normalization::Float16ToFloat32(half[i]);
        const float q = std::clamp(std::nearbyint(original[i] / scale) + zero_point, -128.f, 127.f);
        reconstructed[i] = (q - zero_point) * scale;
        const double difference = reconstructed[i] - original[i];
        squared_error += difference * difference;
      }
      for (const auto& mode : {std::string("original"), std::string("input_qdq")}) {
        model.RunPreparedFloat32(mode == "original" ? original.data() : reconstructed.data());
        const auto filename = Basename(argv[index]) + "." + mode + ".f32";
        std::ofstream file(output / filename, std::ios::binary);
        file.write(reinterpret_cast<const char*>(model.OutputData()),
                   model.OutputCount() * sizeof(float));
        if (!file) {
          throw std::runtime_error("failed to write control embedding");
        }
        if (!first) {
          manifest << ',';
        }
        first = false;
        manifest << "{\"image\":\"" << JsonEscape(Basename(argv[index])) << "\",\"mode\":\"" << mode
                 << "\",\"file\":\"" << filename
                 << "\",\"input_qdq_rmse\":" << std::sqrt(squared_error / original.size()) << '}';
      }
    }
    manifest << "]}\n";
    if (!manifest) {
      throw std::runtime_error("failed to write control manifest");
    }
    std::cout << "Input quantization control saved to " << output << '\n';
  }
  catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
