#include "pipeline.hpp"
#include <filesystem>

int main(int argc, char** argv) {
  try {
    if (argc != 5) throw std::runtime_error("dump_frozen_clip_inputs CLIP DATA_DIR FROZEN_DIR OUTPUT_DIR");
    using namespace yolo_clip;
    const std::filesystem::path data(argv[2]), frozen(argv[3]), output(argv[4]);
    if (std::filesystem::exists(output)) throw std::runtime_error("output exists; preserve completed results");
    std::filesystem::create_directories(output);
    std::ifstream samples(data / "samples.tsv");
    if (!samples) throw std::runtime_error("missing selected samples.tsv");
    ClipRunner clip;
    clip.InitFromModel(ReadBinaryFile(argv[1]), RKNN_NPU_CORE_0_1_2);
    std::ofstream embeddings(output / "embeddings.f32", std::ios::binary);
    std::ofstream manifest(output / "manifest.json");
    manifest << std::setprecision(9) << "{\"embedding_dim\":" << clip.OutputCount() << ",\"samples\":[";
    std::string line;
    int count = 0;
    while (std::getline(samples, line)) {
      const auto tab = line.find('\t');
      if (tab == std::string::npos) throw std::runtime_error("invalid sample row");
      const auto name = line.substr(0, tab);
      const auto input = ReadBinaryFile((frozen / "inputs_fp16" / (name + ".bin")).string());
      if (input.size() != ClipPreprocessor::InputBytes()) throw std::runtime_error("frozen input size mismatch");
      clip.RunPrepared(input.data());
      const auto timing = clip.RunPrepared(input.data());
      embeddings.write(reinterpret_cast<const char*>(clip.OutputData()), clip.OutputCount() * sizeof(float));
      if (count++) manifest << ',';
      manifest << "{\"image\":\"" << JsonEscape(name) << "\",\"run_ms\":" << timing.run_ms << '}';
    }
    if (count != 200 || !embeddings) throw std::runtime_error("incomplete frozen CLIP dump");
    manifest << "]}\n";
    if (!manifest) throw std::runtime_error("failed to write embedding manifest");
    std::cout << "Exported " << count << " frozen-input embeddings\n";
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
