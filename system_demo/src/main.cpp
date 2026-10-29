#include "pipeline.hpp"
using namespace yolo_clip;

int main(int argc, char** argv) {
  try {
    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
      std::cout << "YOLO-CLIP system demo\n"
                << "Required: --yolo MODEL --clip MODEL --label-tsv TSV --text-features F32\n"
                << "Input: exactly one of --images DIR | --video FILE | --camera DEVICE\n"
                << "Camera: --camera-path cpu|dma --camera-format YUYV|NV12 --camera-width 640 --camera-height 480\n"
                << "Preprocess: --normalization scalar|lut (default lut), --opencv-threads 1\n"
                << "Scheduling: --yolo-threads 2 --clip-threads 4 --clip-core-mode all|round_robin|auto\n"
                << "Input binding: --zero-copy-input --io-mem; camera dma uses native DMA input automatically\n"
                << "Benchmark: --duration SECONDS --fps FPS --fallback-full-image --serial\n"
                << "Hardware self-test: --dma-self-test [--dma-heap PATH]\n";
      return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "--dma-self-test") {
      if (argc != 2 && !(argc == 4 && std::string(argv[2]) == "--dma-heap"))
        throw std::runtime_error("--dma-self-test [--dma-heap PATH]");
      camera_dma::SelfTest(argc == 4 ? argv[3] : "/dev/dma_heap/system");
      std::cout << "DMA heap + RGA FD import + color/resize/letterbox/ROI + cache sync: PASS\n";
      return 0;
    }
    const Options opt = ParseArgs(argc, argv);
    npu_arbiter.Enable(opt.npu_gate == "per_core");
    cv::setNumThreads(opt.opencv_threads);
    const auto image_paths = opt.image_dir.empty() ? std::vector<std::string>{} : LoadImagePaths(opt.image_dir);
    const auto images = opt.image_dir.empty() ? std::vector<cv::Mat>{} : LoadImages(opt.image_dir);
    TextBank text_bank;
    text_bank.Load(opt.label_tsv_path, opt.text_feature_path);
    const auto yolo_model = ReadBinaryFile(opt.yolo_model_path);
    const auto clip_model = ReadBinaryFile(opt.clip_model_path);

    if (opt.serial) {
      RunSerial(opt, images, image_paths, text_bank, yolo_model, clip_model);
    } else {
      RunPool(opt, images, image_paths, text_bank, yolo_model, clip_model);
    }
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << std::endl;
    return 1;
  }
}
