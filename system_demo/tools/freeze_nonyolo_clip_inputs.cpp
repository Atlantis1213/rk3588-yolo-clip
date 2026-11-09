#include "pipeline.hpp"
#include <filesystem>

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      throw std::runtime_error("freeze_nonyolo_clip_inputs DATA_DIR");
    }
    using namespace yolo_clip;
    cv::setNumThreads(1);
    const std::filesystem::path data(argv[1]), output = data / "frozen";
    if (std::filesystem::exists(output)) {
      throw std::runtime_error("preserve previous frozen inputs");
    }
    std::filesystem::create_directories(output / "inputs_fp16");
    std::ifstream samples(data / "dataset/samples.tsv");
    if (!samples) {
      throw std::runtime_error("missing selected samples");
    }
    ClipPreprocessor prep(false, normalization::Mode::Lut);
    std::vector<uint8_t> input(ClipPreprocessor::InputBytes());
    std::ofstream manifest(output / "manifest.json");
    manifest << "{\"count\":200,\"yolo_used\":false,\"samples\":[";
    int count = 0;
    std::string line;
    while (std::getline(samples, line)) {
      const auto tab = line.find('\t');
      if (tab == std::string::npos) {
        throw std::runtime_error("invalid sample row");
      }
      const auto name = line.substr(0, tab);
      if (name.find('/') != std::string::npos || name.find('\\') != std::string::npos) {
        throw std::runtime_error("simple sample basename required");
      }
      const auto image = cv::imread((data / "crops" / (name + ".png")).string());
      if (image.empty()) {
        throw std::runtime_error("missing crop");
      }
      double milliseconds = 0;
      prep.PrepareInput(image, input.data(), milliseconds);
      std::ofstream tensor(output / "inputs_fp16" / (name + ".bin"), std::ios::binary);
      tensor.write(reinterpret_cast<const char*>(input.data()), input.size());
      if (!tensor) {
        throw std::runtime_error("failed to freeze tensor");
      }
      if (count++) {
        manifest << ',';
      }
      manifest << "{\"image\":\"" << JsonEscape(name) << "\"}";
    }
    if (count != 200) {
      throw std::runtime_error("all 200 samples required");
    }
    manifest << "]}\n";
    if (!manifest) {
      throw std::runtime_error("manifest write failed");
    }
    std::cout << "Frozen 200 non-YOLO crop inputs\n";
  }
  catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
