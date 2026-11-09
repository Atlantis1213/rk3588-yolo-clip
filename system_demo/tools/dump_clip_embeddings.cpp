// 导出固定输入 FP16 位模式、FP32 embedding 和样本 manifest，供跨后端数值对照。
// 可选择整图或 YOLO ROI；这是串行诊断工具，其耗时不代表多上下文聚合吞吐。

#include "pipeline.hpp"
#include <filesystem>

int main(int argc, char** argv) {
  try {
    if (argc < 4) {
      throw std::runtime_error(
          "dump_clip_embeddings MODEL IMAGE_DIR OUTPUT_DIR [host|dma] [YOLO_MODEL]");
    }
    using namespace yolo_clip;
    cv::setNumThreads(1);
    const bool dma = argc > 4 && std::string(argv[4]) == "dma";
    const bool native_int8 = argc > 4 && std::string(argv[4]) == "native-int8";
    const bool fp32 = argc > 4 && std::string(argv[4]) == "float32";
    const auto paths = LoadImagePaths(argv[2]);
    const auto model = ReadBinaryFile(argv[1]);
    ClipRunner clip;
    if (dma || native_int8) {
      clip.EnableDirectInput("/dev/dma_heap/system", native_int8);
    }
    clip.InitFromModel(model, RKNN_NPU_CORE_0_1_2);
    std::unique_ptr<YoloRunner> yolo;
    if (argc > 5) {
      yolo = std::make_unique<YoloRunner>(false, normalization::Mode::Lut);
      yolo->InitFromModel(ReadBinaryFile(argv[5]), RKNN_NPU_CORE_0);
    }
    ClipPreprocessor preprocessor(false, normalization::Mode::Lut);
    std::vector<uint8_t> input(clip.InputBytes());
    std::vector<float> input_f32(224 * 224 * 3);
    std::shared_ptr<camera_dma::DmaBuffer> buffer;
    if (dma || native_int8) {
      buffer = std::make_shared<camera_dma::DmaBuffer>("/dev/dma_heap/system",
                                                       clip.InputAllocationBytes(input.size()));
    }
    const std::filesystem::path output(argv[3]);
    std::filesystem::create_directories(output / "inputs_fp16");
    std::ofstream embeddings(output / "embeddings.f32", std::ios::binary);
    std::ofstream manifest(output / "manifest.json");
    manifest << std::setprecision(9) << "{\"embedding_dim\":" << clip.OutputCount()
             << ",\"samples\":[";
    bool first = true;
    for (const auto& path : paths) {
      auto image = cv::imread(path);
      if (image.empty()) {
        throw std::runtime_error("failed to read image: " + path);
      }
      cv::Rect roi(0, 0, image.cols, image.rows);
      if (yolo) {
        double ms;
        roi = yolo->Run(image, 0.05f, ms).roi;
      }
      double preprocessing_ms;
      preprocessor.PrepareInput(image(roi), input.data(), preprocessing_ms);
      // FP32 对照来自已经舍入的同一 FP16 张量，而非重新生成更高精度输入，便于隔离 SDK 行为。
      if (fp32) {
        const auto* half = reinterpret_cast<const uint16_t*>(input.data());
        for (size_t k = 0; k < input_f32.size(); ++k) {
          input_f32[k] = normalization::Float16ToFloat32(half[k]);
        }
      }
      const std::string basename = Basename(path);
      std::ofstream tensor(output / "inputs_fp16" / (basename + ".bin"), std::ios::binary);
      tensor.write(reinterpret_cast<const char*>(input.data()), input.size());
      if (dma || native_int8) {
        camera_dma::CpuAccess write(*buffer, DMA_BUF_SYNC_WRITE);
        // 实验分支按模型原生 scale/zero_point 量化并饱和到 INT8；此路径不等于默认 FP16 部署。
        if (native_int8) {
          const auto& attr = clip.NativeInputAttr();
          const auto* half = reinterpret_cast<const uint16_t*>(input.data());
          auto* destination = static_cast<int8_t*>(buffer->Data());
          for (size_t k = 0; k < input.size() / 2; ++k) {
            const float value = normalization::Float16ToFloat32(half[k]);
            const float quantized = std::nearbyint(value / attr.scale) + attr.zp;
            destination[k] = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, quantized)));
          }
        } else {
          std::memcpy(buffer->Data(), input.data(), input.size());
        }
        write.Finish();
      }
      // 每个样本先预热一次，再导出后一次执行的 embedding 和耗时。
      for (int warm = 0; warm < 1; ++warm) {
        if (dma || native_int8) {
          clip.RunPreparedDma(buffer);
        } else if (fp32) {
          clip.RunPreparedFloat32(input_f32.data());
        } else {
          clip.RunPrepared(input.data());
        }
      }
      const auto result = (dma || native_int8) ? clip.RunPreparedDma(buffer)
                          : fp32               ? clip.RunPreparedFloat32(input_f32.data())
                                               : clip.RunPrepared(input.data());
      embeddings.write(reinterpret_cast<const char*>(clip.OutputData()),
                       clip.OutputCount() * sizeof(float));
      if (!first) {
        manifest << ',';
      }
      first = false;
      manifest << "{\"image\":\"" << JsonEscape(basename) << "\",\"run_ms\":" << result.run_ms
               << ",\"roi\":[" << roi.x << ',' << roi.y << ',' << roi.width << ',' << roi.height
               << "]}";
    }
    manifest << "]}\n";
    if (!embeddings || !manifest) {
      throw std::runtime_error("failed to write embedding artifacts");
    }
    std::cout << "Embedding/input tensors exported to " << output << '\n';
  }
  catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
