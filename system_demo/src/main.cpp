// 程序入口：加载配置、离线文本特征和两个 RKNN 模型，再选择串行或两阶段线程池。
// 建议按 main.cpp → pipeline.cpp → types.hpp → model_runner.hpp 的顺序阅读。

#include "pipeline.hpp"
using namespace yolo_clip;

int main(int argc, char** argv) {
  try {
    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
      std::cout
          << "YOLO-CLIP system demo\n"
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
    // 硬件自检独立于模型推理，可先验证 DMA heap、RGA 导入和颜色转换是否正常。
    if (argc >= 2 && std::string(argv[1]) == "--dma-self-test") {
      if (argc != 2 && !(argc == 4 && std::string(argv[2]) == "--dma-heap")) {
        throw std::runtime_error("--dma-self-test [--dma-heap PATH]");
      }
      camera_dma::SelfTest(argc == 4 ? argv[3] : "/dev/dma_heap/system");
      std::cout << "DMA heap + RGA FD import + color/resize/letterbox/ROI + cache sync: PASS\n";
      return 0;
    }
    // 参数检查在创建线程前完成；后续各线程只读取同一个 Options。
    const Options opt = ParseArgs(argc, argv);
    npu_arbiter.Enable(opt.npu_gate == "per_core");
    // 限制 OpenCV 内部线程数，避免每个流水线 worker 再创建大量 CPU 线程。
    cv::setNumThreads(opt.opencv_threads);
    const auto image_paths =
        opt.image_dir.empty() ? std::vector<std::string>{} : LoadImagePaths(opt.image_dir);
    const auto images = opt.image_dir.empty() ? std::vector<cv::Mat>{} : LoadImages(opt.image_dir);
    // 文本编码在离线阶段完成；运行时只加载特征并与图像特征计算余弦相似度。
    TextBank text_bank;
    text_bank.Load(opt.label_tsv_path, opt.text_feature_path);
    const auto yolo_model = ReadBinaryFile(opt.yolo_model_path);
    const auto clip_model = ReadBinaryFile(opt.clip_model_path);

    // 两种模式共享模型与分类逻辑，便于比较串行处理和流水线重叠的效果。
    if (opt.serial) {
      RunSerial(opt, images, image_paths, text_bank, yolo_model, clip_model);
    } else {
      RunPool(opt, images, image_paths, text_bank, yolo_model, clip_model);
    }
    return 0;
  }
  catch (const std::exception& ex) {
    // worker 的异常会在 join 后重抛到此处，统一以非零退出码报告失败。
    std::cerr << ex.what() << std::endl;
    return 1;
  }
}
