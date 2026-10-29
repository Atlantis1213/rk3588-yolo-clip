#include "pipeline.hpp"
#include <filesystem>

class ObservedYolo : public yolo_clip::YoloRunner {
 public:
  ObservedYolo() : YoloRunner(false, yolo_clip::normalization::Mode::Lut) {}
  float Confidence() const {
    const auto count = output_float_.size() / 5;
    return *std::max_element(output_float_.begin() + 4 * count, output_float_.end());
  }
};

// Evaluate the user's requested offline branch: GT empty skips YOLO but still reaches CLIP.
// Freeze ROIs once so every CLIP variant sees exactly the same tensor bytes.
int main(int argc, char** argv) {
  try {
    if (argc != 5) throw std::runtime_error("prepare_real_clip_inputs YOLO DATA_DIR OUTPUT_DIR CONF");
    using namespace yolo_clip;
    cv::setNumThreads(1);
    const std::filesystem::path data(argv[2]), output(argv[3]);
    if (std::filesystem::exists(output)) throw std::runtime_error("output exists; never overwrite frozen crops");
    std::filesystem::create_directories(output / "inputs_fp16");
    std::filesystem::create_directories(output / "rois");
    std::ifstream samples(data / "samples.tsv");
    if (!samples) throw std::runtime_error("missing selected samples.tsv");
    ObservedYolo yolo;
    yolo.InitFromModel(ReadBinaryFile(argv[1]), RKNN_NPU_CORE_0);
    ClipPreprocessor prep(false, normalization::Mode::Lut);
    std::vector<uint8_t> input(ClipPreprocessor::InputBytes());
    std::ofstream manifest(output / "manifest.json");
    manifest << std::setprecision(9) << "{\"count\":200,\"samples\":[";
    std::string line;
    int count = 0, detections = 0, empty_count = 0;
    while (std::getline(samples, line)) {
      const auto tab = line.find('\t');
      if (tab == std::string::npos) throw std::runtime_error("invalid samples row");
      const std::string name = line.substr(0, tab), category = line.substr(tab + 1);
      if (name.find('/') != std::string::npos || name.find('\\') != std::string::npos)
        throw std::runtime_error("sample must be a simple basename");
      const auto image = cv::imread((data / "images" / name).string());
      if (image.empty()) throw std::runtime_error("failed to read " + name);
      const bool skip = category == "empty";
      bool detected = false;
      double yolo_preprocess_ms = 0, yolo_run_ms = 0, clip_preprocess_ms = 0;
      float confidence = 0;
      cv::Rect roi(0, 0, image.cols, image.rows);
      if (skip) ++empty_count;
      else {
        const auto result = yolo.Run(image, std::stof(argv[4]), yolo_preprocess_ms);
        detected = result.detected;
        yolo_run_ms = result.run_ms;
        confidence = yolo.Confidence();
        roi = result.roi;
        if (detected) ++detections;
      }
      // Undetected nonempty samples keep full-frame tensors for numeric diagnosis only.
      // The task evaluator will count them as pipeline detection failures, not use this fallback prediction.
      prep.PrepareInput(image(roi), input.data(), clip_preprocess_ms);
      std::ofstream tensor(output / "inputs_fp16" / (name + ".bin"), std::ios::binary);
      tensor.write(reinterpret_cast<const char*>(input.data()), input.size());
      if (!tensor || !cv::imwrite((output / "rois" / (name + ".png")).string(), image(roi)))
        throw std::runtime_error("failed to save frozen tensor/ROI");
      if (count++) manifest << ',';
      manifest << "{\"image\":\"" << JsonEscape(name) << "\",\"class\":\"" << JsonEscape(category)
          << "\",\"skip_yolo\":" << (skip ? "true" : "false") << ",\"detected\":" << (detected ? "true" : "false")
          << ",\"diagnostic_full_frame_fallback\":" << (!skip && !detected ? "true" : "false")
          << ",\"max_confidence\":" << confidence
          << ",\"roi\":[" << roi.x << ',' << roi.y << ',' << roi.width << ',' << roi.height << ']'
          << ",\"yolo_preprocess_ms\":" << yolo_preprocess_ms << ",\"yolo_run_ms\":" << yolo_run_ms
          << ",\"clip_preprocess_ms\":" << clip_preprocess_ms << '}';
    }
    if (count != 200 || empty_count != 40) throw std::runtime_error("expected 200 balanced images including 40 empty");
    manifest << "],\"detected_nonempty\":" << detections << ",\"empty_count\":" << empty_count << "}\n";
    if (!manifest) throw std::runtime_error("failed to write crop manifest");
    std::cout << "Frozen inputs: " << count << "; detected nonempty: " << detections << "/160\n";
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
