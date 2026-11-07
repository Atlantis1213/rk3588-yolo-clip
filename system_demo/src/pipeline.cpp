// 流水线编排：生产者 → frame_queue → YOLO workers → clip_queue → CLIP workers。
// 两级有界队列控制积压；Frame 的共享所有权和 CLIP Lease 管理跨线程资源。
// 阅读重点：正常结束逐级关闭，异常结束同时唤醒所有等待者，最后 join 并检查资源归还。

#include "pipeline.hpp"

namespace yolo_clip {
// 把命令行解析与组合约束集中在这里，使 worker 不必重复检查配置。
Options ParseArgs(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    // 读取带值选项时同时推进 i；缺值立即抛错，防止越界访问 argv。
    auto next = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + name);
      }
      ++i;
      return argv[i];
    };
    if (arg == "--yolo") opt.yolo_model_path = next(arg);
    else if (arg == "--clip") opt.clip_model_path = next(arg);
    else if (arg == "--images") opt.image_dir = next(arg);
    else if (arg == "--video") opt.video_path = next(arg);
    else if (arg == "--camera") opt.camera_device = next(arg);
    else if (arg == "--camera-format") opt.camera_format = next(arg);
    else if (arg == "--camera-path") opt.camera_path = next(arg);
    else if (arg == "--normalization") opt.normalization_mode = next(arg);
    else if (arg == "--opencv-threads") opt.opencv_threads = std::stoi(next(arg));
    else if (arg == "--camera-width") opt.camera_width = std::stoi(next(arg));
    else if (arg == "--camera-height") opt.camera_height = std::stoi(next(arg));
    else if (arg == "--camera-buffers") opt.camera_buffers = std::stoi(next(arg));
    else if (arg == "--dma-heap") opt.dma_heap = next(arg);
    else if (arg == "--zero-copy-input") opt.zero_copy_input = true;
    else if (arg == "--label-tsv") opt.label_tsv_path = next(arg);
    else if (arg == "--text-features") opt.text_feature_path = next(arg);
    else if (arg == "--duration") opt.duration_sec = std::stoi(next(arg));
    else if (arg == "--fps") opt.input_fps = std::stod(next(arg));
    else if (arg == "--frame-queue") opt.frame_queue = std::stoi(next(arg));
    else if (arg == "--clip-queue") opt.clip_queue = std::stoi(next(arg));
    else if (arg == "--clip-buffers") opt.clip_buffers = std::stoi(next(arg));
    else if (arg == "--yolo-threads") opt.yolo_threads = std::stoi(next(arg));
    else if (arg == "--clip-threads") opt.clip_threads = std::stoi(next(arg));
    else if (arg == "--clip-core-mode") opt.clip_core_mode = next(arg);
    else if (arg == "--yolo-core-mode") opt.yolo_core_mode = next(arg);
    else if (arg == "--npu-gate") opt.npu_gate = next(arg);
    else if (arg == "--topk") opt.topk = std::stoi(next(arg));
    else if (arg == "--reject") opt.reject_threshold = std::stof(next(arg));
    else if (arg == "--preprocess") opt.preprocess_backend = next(arg);
    else if (arg == "--io-mem") opt.use_io_mem = true;
    else if (arg == "--affinity") {
      opt.use_affinity = true;
      opt.affinity_profile = "legacy";
    }
    else if (arg == "--affinity-profile") {
      opt.affinity_profile = next(arg);
      opt.use_affinity = opt.affinity_profile != "none";
    }
    else if (arg == "--fallback-full-image") opt.fallback_full_image = true;
    else if (arg == "--conf") opt.yolo_conf_threshold = std::stof(next(arg));
    else if (arg == "--serial") opt.serial = true;
    else throw std::runtime_error("unknown arg: " + arg);
  }
  // 输入源三选一。摄像头由设备节奏驱动，图片/视频由生产者按 input_fps 投递。
  const bool has_image_dir = !opt.image_dir.empty();
  const bool has_video = !opt.video_path.empty();
  const bool has_camera = !opt.camera_device.empty();
  if (static_cast<int>(has_image_dir) + has_video + has_camera != 1) {
    throw std::runtime_error("provide exactly one of --images DIR, --video FILE, or --camera /dev/videoN");
  }
  if (opt.yolo_model_path.empty() || opt.clip_model_path.empty() ||
      opt.label_tsv_path.empty() || opt.text_feature_path.empty()) {
    throw std::runtime_error(
        "usage: yolo_clip_system_demo --yolo YOLO.rknn --clip CLIP.rknn (--images DIR | --video FILE | --camera DEVICE) "
        "--label-tsv LABELS.tsv --text-features FEATURES.f32 "
        "[--duration 10] [--fps 30] [--serial] [--yolo-threads 2] [--clip-threads 4] "
        "[--frame-queue 4] [--clip-queue 8] [--clip-buffers 8] [--topk 5] "
        "[--clip-core-mode all|round_robin|auto|spread|split12|core0|core1|core2] "
        "[--yolo-core-mode legacy|auto|all|adaptive|core0|core1|core2] [--npu-gate none|per_core] "
        "[--normalization scalar|lut|scalar_legacy|lut_legacy] [--opencv-threads 1] "
        "[--reject 0.22] [--preprocess opencv|rga] [--io-mem] "
        "[--affinity-profile none|legacy|segregated] "
        "[--fallback-full-image] [--conf 0.05] [--zero-copy-input] [--dma-heap /dev/dma_heap/system] "
        "[--camera-width 640] [--camera-height 480] [--camera-format YUYV|NV12] [--camera-buffers 12] "
        "[--camera-path cpu|dma]");
  }
  if (opt.duration_sec <= 0 || !std::isfinite(opt.input_fps) || opt.input_fps <= 0.0 ||
      opt.frame_queue <= 0 || opt.clip_queue <= 0 || opt.clip_buffers <= 0 ||
      opt.yolo_threads <= 0 || opt.clip_threads <= 0 || opt.topk <= 0) {
    throw std::runtime_error("duration, fps, queue sizes, buffer count, thread counts, and topk must be positive");
  }
  if (opt.preprocess_backend != "opencv" && opt.preprocess_backend != "rga") {
    throw std::runtime_error("--preprocess must be opencv or rga");
  }
  if (opt.clip_core_mode != "all" && opt.clip_core_mode != "round_robin" && opt.clip_core_mode != "auto" &&
      opt.clip_core_mode != "spread" && opt.clip_core_mode != "split12" &&
      opt.clip_core_mode != "core0" && opt.clip_core_mode != "core1" && opt.clip_core_mode != "core2") {
    throw std::runtime_error("unsupported --clip-core-mode");
  }
  if (opt.yolo_core_mode != "legacy" && opt.yolo_core_mode != "auto" && opt.yolo_core_mode != "all" && opt.yolo_core_mode != "adaptive" &&
      opt.yolo_core_mode != "core0" && opt.yolo_core_mode != "core1" && opt.yolo_core_mode != "core2")
    throw std::runtime_error("unsupported --yolo-core-mode");
  if (opt.npu_gate != "none" && opt.npu_gate != "per_core")
    throw std::runtime_error("--npu-gate must be none or per_core");
  if (opt.yolo_core_mode == "adaptive" && opt.npu_gate != "per_core")
    throw std::runtime_error("adaptive YOLO placement requires --npu-gate per_core");
  if (opt.affinity_profile != "none" && opt.affinity_profile != "legacy" &&
      opt.affinity_profile != "segregated") {
    throw std::runtime_error("--affinity-profile must be none, legacy, or segregated");
  }
  if (opt.camera_width <= 0 || opt.camera_width > 8192 || opt.camera_height <= 0 ||
      opt.camera_height > 8192 || opt.camera_buffers < 3 || opt.camera_buffers > 64 ||
      (opt.camera_format != "YUYV" && opt.camera_format != "NV12") || opt.input_fps > 1000.0)
    throw std::runtime_error("invalid camera size, buffer count, format, or FPS");
  if (has_camera) {
    if (opt.input_fps < 0.001) throw std::runtime_error("camera FPS must be >= 0.001");
    // 采集缓冲需覆盖排队帧和正在运行的 YOLO 帧，并留两个余量供驱动继续采集。
    const int minimum = opt.frame_queue + (opt.serial ? 1 : opt.yolo_threads) + 2;
    if (opt.camera_buffers < minimum)
      throw std::runtime_error("--camera-buffers must be >= frame-queue + YOLO workers + 2");
    // 摄像头 DMA 路径要求 RGA 与原生输入绑定；CPU 路径用于对照测量。
    opt.preprocess_backend = opt.camera_path == "dma" ? "rga" : "opencv";
    if (opt.camera_path == "dma") opt.zero_copy_input = true;
  }
  if (opt.camera_path != "cpu" && opt.camera_path != "dma")
    throw std::runtime_error("--camera-path must be cpu or dma");
  normalization::ParseMode(opt.normalization_mode);
  if (opt.opencv_threads <= 0) throw std::runtime_error("--opencv-threads must be positive");
  return opt;
}

// Open 只完成协商和缓冲准备；模型、worker 资源准备完毕后才 Start 采集。
std::shared_ptr<camera_dma::V4l2Capture> OpenCamera(const Options& opt) {
  if (opt.camera_device.empty()) return nullptr;
  auto camera = camera_dma::V4l2Capture::Open(opt.camera_device, opt.camera_width,
      opt.camera_height, opt.camera_format, opt.camera_buffers, opt.input_fps,
      opt.frame_queue + (opt.serial ? 1 : opt.yolo_threads) + 2, opt.camera_path == "cpu");
  std::cerr << "V4L2: " << camera->Description() << '\n';
  return camera;
}

// Acquire 暂时超时可继续等待；任务持有 Frame，保证队列中的摄像头缓冲不会被驱动覆写。
void ProduceCamera(const Options& opt, const std::shared_ptr<camera_dma::V4l2Capture>& camera,
                   DropOldestQueue<FrameTask>& queue, Stats& stats, Clock::time_point start,
                   const ThreadFailure& failure) {
  const auto deadline = start + std::chrono::seconds(opt.duration_sec);
  int sequence = 0;
  while (!failure.stopped.load() && Clock::now() < deadline) {
    auto frame = camera->Acquire();
    if (!frame) continue;
    FrameTask task;
    task.seq = sequence++;
    task.image_index = -1;
    task.source_name = "camera_frame_" + std::to_string(frame->sequence);
    // 端到端延迟从任务投递附近开始计算，不包含传感器曝光与 USB 传输时间。
    task.capture_time = Clock::now();
    task.camera_frame = std::move(frame);
    stats.generated.fetch_add(1);
    queue.Push(std::move(task), [&](const FrameTask&) { stats.frame_dropped.fetch_add(1); });
  }
  if (!failure.stopped.load() && sequence == 0)
    throw std::runtime_error("camera produced no valid frames within duration");
}

// 将原图 ROI 变为 224×224 RGB 并归一化到目标 FP16 张量；目标可为普通内存或 DMA。
void PrepareClipTask(const FrameTask& task, const cv::Rect& roi, ClipPreprocessor& cpu,
                     camera_dma::Preprocessor& hardware, uint8_t* destination,
                     const std::shared_ptr<camera_dma::DmaBuffer>& dma, double& elapsed_ms,
                     normalization::Mode mode) {
  const auto start = Clock::now();
  std::unique_ptr<camera_dma::CpuAccess> write;
  if (dma) write = std::make_unique<camera_dma::CpuAccess>(*dma, DMA_BUF_SYNC_WRITE);
  // DMA 摄像头帧复用该 YOLO worker 已转换好的完整图像；CPU/文件输入走 cv::Mat 裁剪。
  if (task.camera_frame && !task.frame) {
    auto& rgb = hardware.ClipRgb(roi.x, roi.y, roi.width, roi.height);
    PackRgbFp16(rgb, destination, true, mode);
  } else {
    cpu.PrepareInput((*task.frame)(roi), destination, elapsed_ms);
  }
  // 显式结束 CPU 写访问，让后续设备读取看到更新；析构仍提供异常路径的收尾。
  if (write) write->Finish();
  elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// CPU 摄像头路径先复制并转 BGR；DMA 路径由 YoloRunner 直接调用硬件预处理。
YoloResult RunDetection(FrameTask& task, const Options& opt, YoloRunner& yolo,
                        camera_dma::Preprocessor& hardware, CameraCpuPreprocessor& cpu,
                        double& preprocess_ms) {
  double conversion_ms = 0.0;
  if (task.camera_frame && opt.camera_path == "cpu")
    task.frame = std::make_shared<cv::Mat>(cpu.Convert(*task.camera_frame, conversion_ms));
  const auto result = task.frame ? yolo.Run(*task.frame, opt.yolo_conf_threshold, preprocess_ms) :
      yolo.RunCamera(*task.camera_frame, hardware, opt.yolo_conf_threshold, preprocess_ms);
  preprocess_ms += conversion_ms;
  return result;
}

// 预先分配固定数量的 CLIP 输入，减少每帧分配；空 heap 表示使用宿主 vector。
std::vector<ClipSlot> MakeClipSlots(int count, size_t bytes, const std::string& heap) {
  std::vector<ClipSlot> slots(count);
  for (auto& slot : slots) {
    if (heap.empty()) slot.input_bytes.resize(bytes);
    else slot.dma = std::make_shared<camera_dma::DmaBuffer>(heap, bytes);
  }
  return slots;
}

// 串行基线仍保留独立输入生产线程；单个 worker 内依次执行 YOLO、ROI、CLIP 和文本匹配。
void RunSerial(const Options& opt, const std::vector<cv::Mat>& images,
               const std::vector<std::string>& image_paths, const TextBank& text_bank,
               const std::vector<uint8_t>& yolo_model, const std::vector<uint8_t>& clip_model) {
  Stats stats;
  DropOldestQueue<FrameTask> frame_queue(static_cast<size_t>(opt.frame_queue));
  const bool use_rga = opt.preprocess_backend == "rga";

  auto camera = OpenCamera(opt);
  ThreadFailure failure;
  auto wake = [&]() { frame_queue.Close(); };
  camera_dma::Preprocessor hardware(opt.dma_heap);
  CameraCpuPreprocessor cpu_capture;
  const auto normalization_mode = normalization::ParseMode(opt.normalization_mode);

  YoloRunner yolo(use_rga, normalization_mode);
  yolo.EnableIoMem(opt.use_io_mem);
  if (opt.zero_copy_input) yolo.EnableDirectInput(opt.dma_heap);
  yolo.InitFromModel(yolo_model, RKNN_NPU_CORE_AUTO);
  ClipRunner clip;
  clip.EnableIoMem(opt.use_io_mem);
  if (opt.zero_copy_input) clip.EnableDirectInput(opt.dma_heap);
  clip.InitFromModel(clip_model, RKNN_NPU_CORE_AUTO);
  ClipPreprocessor clip_preprocessor(use_rga, normalization_mode);
  std::vector<uint8_t> clip_input(clip.InputBytes());
  std::shared_ptr<camera_dma::DmaBuffer> clip_dma;
  if (opt.zero_copy_input) clip_dma = std::make_shared<camera_dma::DmaBuffer>(
      opt.dma_heap, clip.InputAllocationBytes(clip.InputBytes()));

  if (camera) camera->Start();
  const auto wall_start = Clock::now();
  std::thread producer([&]() {
    failure.Guard([&]() {
    PinThreadToCore(ProducerCore(opt));
    if (camera) {
      ProduceCamera(opt, camera, frame_queue, stats, wall_start, failure);
      return;
    }
    // 文件输入按绝对目标时刻节流，避免每轮相对 sleep 的误差不断累积。
    const int total_frames = static_cast<int>(opt.duration_sec * opt.input_fps);
    const auto interval = std::chrono::duration<double>(1.0 / opt.input_fps);
    cv::VideoCapture cap;
    if (!opt.video_path.empty()) {
      cap = OpenVideo(opt.video_path);
    }
    for (int i = 0; i < total_frames && !failure.stopped.load(); ++i) {
      const auto target_time = wall_start + std::chrono::duration_cast<Clock::duration>(interval * i);
      std::this_thread::sleep_until(target_time);
      FrameTask task;
      task.seq = i;
      if (!opt.video_path.empty()) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
          cap.release();
          cap = OpenVideo(opt.video_path);
          if (!cap.read(frame) || frame.empty()) {
            throw std::runtime_error("failed to loop video input");
          }
        }
        task.image_index = -1;
        task.source_name = "video_frame";
        // 视频帧复制为任务独有的数据，避免下一次解码复用底层内存。
        task.frame = std::make_shared<cv::Mat>(frame.clone());
      } else {
        task.image_index = i % static_cast<int>(images.size());
        task.source_name = Basename(image_paths[task.image_index]);
        task.frame = std::make_shared<cv::Mat>(images[task.image_index]);
      }
      // 端到端延迟从任务投递附近开始计算，不包含传感器曝光与 USB 传输时间。
      task.capture_time = Clock::now();
      stats.generated.fetch_add(1);
      frame_queue.Push(std::move(task), [&](const FrameTask&) { stats.frame_dropped.fetch_add(1); });
    }
    }, wake);
    frame_queue.Close();
  });

  std::thread worker([&]() {
    failure.Guard([&]() {
    PinThreadToCore(YoloWorkerCore(opt, 0));
    FrameTask task;
    while (!failure.stopped.load() && frame_queue.Pop(task)) {
      stats.AddYoloQueueWait(
          std::chrono::duration<double, std::milli>(Clock::now() - task.capture_time).count());
      double yolo_pre_ms = 0.0;
      const YoloResult yolo_result = RunDetection(task, opt, yolo, hardware, cpu_capture, yolo_pre_ms);
      stats.AddYoloPreprocess(yolo_pre_ms);
      stats.AddYoloRun(yolo_result.run_ms, yolo_result.gate_wait_ms);

      // 无有效检测时可跳过分类；开启 fallback 后使用 YoloResult 提供的整图 ROI。
      if (!yolo_result.detected && !opt.fallback_full_image) {
        stats.skipped_no_object.fetch_add(1);
        continue;
      }

      double clip_pre_ms = 0.0;
      PrepareClipTask(task, yolo_result.roi, clip_preprocessor, hardware,
          clip_dma ? static_cast<uint8_t*>(clip_dma->Data()) : clip_input.data(), clip_dma, clip_pre_ms,
          normalization_mode);
      // CLIP 输入准备完成后即可归还原始采集帧；后续分类只依赖已经写好的输入张量。
      task.camera_frame.reset();
      stats.AddClipPreprocess(clip_pre_ms);
      const ClipResult clip_result = clip_dma ? clip.RunPreparedDma(clip_dma) :
                                               clip.RunPrepared(clip_input.data());
      stats.AddClipRun(clip_result.run_ms, clip_result.gate_wait_ms);
      const ClassificationResult cls =
          text_bank.Classify(clip.OutputData(), clip.OutputCount(), opt.topk, opt.reject_threshold);

      const auto now = Clock::now();
      const double latency_ms =
          std::chrono::duration<double, std::milli>(now - task.capture_time).count();
      stats.AddCompleted(yolo_result.detected, latency_ms, clip_result.checksum, task.image_index,
                         task.source_name, cls);
    }
    }, wake);
  });

  // 等待线程全部结束后再读取最终统计、重抛工作异常，避免使用仍在被修改的数据。
  producer.join();
  worker.join();
  failure.Rethrow();
  const auto wall_end = Clock::now();
  PrintReport(opt, stats, image_paths,
              std::chrono::duration<double, std::milli>(wall_end - wall_start).count());
}

// 两阶段线程池让不同帧的检测与分类重叠；同一 runner/context 只交给一个 worker 使用。
void RunPool(const Options& opt, const std::vector<cv::Mat>& images,
             const std::vector<std::string>& image_paths, const TextBank& text_bank,
             const std::vector<uint8_t>& yolo_model,
             const std::vector<uint8_t>& clip_model) {
  Stats stats;
  DropOldestQueue<FrameTask> frame_queue(static_cast<size_t>(opt.frame_queue));
  DropOldestQueue<ClipTask> clip_queue(static_cast<size_t>(opt.clip_queue));
  const bool use_rga = opt.preprocess_backend == "rga";
  const auto normalization_mode = normalization::ParseMode(opt.normalization_mode);
  auto camera = OpenCamera(opt);
  ThreadFailure failure;
  // 任一 worker 失败时同时关闭两级队列，唤醒阻塞在 Pop 的线程以便退出。
  auto wake = [&]() { frame_queue.Close(); clip_queue.Close(); };

  // 首个 context 从模型创建，其余通过 RKNN dup_context 获得各自的执行上下文。
  std::vector<std::unique_ptr<YoloRunner>> yolo_workers;
  yolo_workers.reserve(opt.yolo_threads);
  auto yolo_root = std::make_unique<YoloRunner>(use_rga, normalization_mode);
  yolo_root->EnableIoMem(opt.use_io_mem);
  yolo_root->EnableAdaptiveCore(opt.yolo_core_mode == "adaptive");
  if (opt.zero_copy_input) yolo_root->EnableDirectInput(opt.dma_heap);
  yolo_root->InitFromModel(yolo_model, YoloCoreMask(0, opt.yolo_core_mode));
  yolo_workers.push_back(std::move(yolo_root));
  for (int i = 1; i < opt.yolo_threads; ++i) {
    auto runner = std::make_unique<YoloRunner>(use_rga, normalization_mode);
    runner->EnableAdaptiveCore(opt.yolo_core_mode == "adaptive");
    runner->EnableIoMem(opt.use_io_mem);
    if (opt.zero_copy_input) runner->EnableDirectInput(opt.dma_heap);
    runner->InitFromContext(yolo_workers[0]->MutableContext(), YoloCoreMask(i, opt.yolo_core_mode));
    yolo_workers.push_back(std::move(runner));
  }

  std::vector<std::unique_ptr<ClipRunner>> clip_workers;
  clip_workers.reserve(opt.clip_threads);
  auto clip_root = std::make_unique<ClipRunner>();
  clip_root->EnableIoMem(opt.use_io_mem);
  if (opt.zero_copy_input) clip_root->EnableDirectInput(opt.dma_heap);
  clip_root->InitFromModel(clip_model, ClipCoreMask(0, opt.clip_core_mode));
  // DMA 分配大小还需满足模型原生 stride 要求，不能只凭逻辑元素数猜测。
  const size_t clip_input_bytes = clip_root->InputAllocationBytes(clip_root->InputBytes());
  clip_workers.push_back(std::move(clip_root));
  for (int i = 1; i < opt.clip_threads; ++i) {
    auto runner = std::make_unique<ClipRunner>();
    runner->EnableIoMem(opt.use_io_mem);
    if (opt.zero_copy_input) runner->EnableDirectInput(opt.dma_heap);
    runner->InitFromContext(clip_workers[0]->MutableContext(), ClipCoreMask(i, opt.clip_core_mode));
    clip_workers.push_back(std::move(runner));
  }

  // Lease 随 ClipTask 移动；任务被丢弃或遇到异常也会归还槽位，无需手动维护 slot_id。
  ClipSlotPool clip_slots(MakeClipSlots(opt.clip_buffers, clip_input_bytes,
                                      opt.zero_copy_input ? opt.dma_heap : ""));
  // 每个 YOLO worker 持有自己的可复用图像缓存，防止并发覆盖预处理结果。
  std::vector<CameraCpuPreprocessor> cpu_capture(opt.yolo_threads);
  std::vector<std::unique_ptr<ClipPreprocessor>> clip_preprocessors;
  clip_preprocessors.reserve(opt.yolo_threads);
  std::vector<std::unique_ptr<camera_dma::Preprocessor>> hardware_preprocessors;
  for (int i = 0; i < opt.yolo_threads; ++i) {
    clip_preprocessors.push_back(std::make_unique<ClipPreprocessor>(use_rga, normalization_mode));
    hardware_preprocessors.push_back(std::make_unique<camera_dma::Preprocessor>(opt.dma_heap));
  }
  std::atomic<int> yolo_alive{opt.yolo_threads};
  std::atomic<int> clip_alive{opt.clip_threads};

  if (camera) camera->Start();
  const auto wall_start = Clock::now();
  std::thread producer([&]() {
    failure.Guard([&]() {
    PinThreadToCore(ProducerCore(opt));
    if (camera) {
      ProduceCamera(opt, camera, frame_queue, stats, wall_start, failure);
      return;
    }
    // 文件输入按绝对目标时刻节流，避免每轮相对 sleep 的误差不断累积。
    const int total_frames = static_cast<int>(opt.duration_sec * opt.input_fps);
    const auto interval = std::chrono::duration<double>(1.0 / opt.input_fps);
    cv::VideoCapture cap;
    if (!opt.video_path.empty()) {
      cap = OpenVideo(opt.video_path);
    }
    for (int i = 0; i < total_frames && !failure.stopped.load(); ++i) {
      const auto target_time = wall_start + std::chrono::duration_cast<Clock::duration>(interval * i);
      std::this_thread::sleep_until(target_time);
      FrameTask task;
      task.seq = i;
      if (!opt.video_path.empty()) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
          cap.release();
          cap = OpenVideo(opt.video_path);
          if (!cap.read(frame) || frame.empty()) {
            throw std::runtime_error("failed to loop video input");
          }
        }
        task.image_index = -1;
        task.source_name = "video_frame";
        // 视频帧复制为任务独有的数据，避免下一次解码复用底层内存。
        task.frame = std::make_shared<cv::Mat>(frame.clone());
      } else {
        task.image_index = i % static_cast<int>(images.size());
        task.source_name = Basename(image_paths[task.image_index]);
        task.frame = std::make_shared<cv::Mat>(images[task.image_index]);
      }
      // 端到端延迟从任务投递附近开始计算，不包含传感器曝光与 USB 传输时间。
      task.capture_time = Clock::now();
      stats.generated.fetch_add(1);
      frame_queue.Push(std::move(task), [&](const FrameTask&) { stats.frame_dropped.fetch_add(1); });
    }
    }, wake);
    frame_queue.Close();
  });

  std::vector<std::thread> yolo_threads;
  for (int i = 0; i < opt.yolo_threads; ++i) {
    yolo_threads.emplace_back([&, i]() {
      failure.Guard([&]() {
      PinThreadToCore(YoloWorkerCore(opt, i));
      FrameTask task;
      while (!failure.stopped.load() && frame_queue.Pop(task)) {
        stats.AddYoloQueueWait(
            std::chrono::duration<double, std::milli>(Clock::now() - task.capture_time).count());
        double yolo_pre_ms = 0.0;
        const YoloResult yolo_result = RunDetection(task, opt, *yolo_workers[i],
            *hardware_preprocessors[i], cpu_capture[i], yolo_pre_ms);
        stats.AddYoloPreprocess(yolo_pre_ms);
        stats.AddYoloRun(yolo_result.run_ms, yolo_result.gate_wait_ms);

        // 无有效检测时可跳过分类；开启 fallback 后使用 YoloResult 提供的整图 ROI。
        if (!yolo_result.detected && !opt.fallback_full_image) {
          stats.skipped_no_object.fetch_add(1);
          continue;
        }

        // 无可用输入槽时直接丢弃当前任务，避免 YOLO 等待 CLIP 造成上游持续积压。
        auto lease = clip_slots.TryAcquire();
        if (!lease) {
          stats.clip_slot_dropped.fetch_add(1);
          continue;
        }

        double clip_pre_ms = 0.0;
        auto& slot = lease.Get();
        PrepareClipTask(task, yolo_result.roi, *clip_preprocessors[i], *hardware_preprocessors[i],
                        slot.Data(), slot.dma, clip_pre_ms, normalization_mode);
        // CLIP 输入准备完成后即可归还原始采集帧；后续分类只依赖已经写好的输入张量。
        task.camera_frame.reset();
        stats.AddClipPreprocess(clip_pre_ms);
        slot.valid = true;

        ClipTask clip_task;
        clip_task.seq = task.seq;
        clip_task.image_index = task.image_index;
        clip_task.source_name = task.source_name;
        // 移动转交唯一归还责任；局部 lease 变为空，不会把仍在队列中的输入提前释放。
        clip_task.slot = std::move(lease);
        clip_task.capture_time = task.capture_time;
        // 保留 capture_time 计算总延迟，另记 enqueue_time 单独统计 CLIP 队列等待。
        clip_task.enqueue_time = Clock::now();
        clip_task.detected = yolo_result.detected;
        clip_queue.Push(std::move(clip_task), [&](const ClipTask&) {
          stats.clip_queue_dropped.fetch_add(1);
        });
      }
      }, wake);
      // 只有最后一个 YOLO worker 能关闭 CLIP 队列，否则其他 YOLO 结果会被提前拒收。
      if (yolo_alive.fetch_sub(1) == 1) {
        clip_queue.Close();
      }
    });
  }

  std::vector<std::thread> clip_threads;
  for (int i = 0; i < opt.clip_threads; ++i) {
    clip_threads.emplace_back([&, i]() {
      failure.Guard([&]() {
      PinThreadToCore(ClipWorkerCore(opt, i));
      ClipTask task;
      while (!failure.stopped.load() && clip_queue.Pop(task)) {
        stats.AddClipQueueWait(
            std::chrono::duration<double, std::milli>(Clock::now() - task.enqueue_time).count());
        ClipSlot& slot = task.slot.Get();
        const ClipResult clip_result = slot.dma ? clip_workers[i]->RunPreparedDma(slot.dma) :
                                                 clip_workers[i]->RunPrepared(slot.Data());
        stats.AddClipRun(clip_result.run_ms, clip_result.gate_wait_ms);
        const ClassificationResult cls = text_bank.Classify(
            clip_workers[i]->OutputData(), clip_workers[i]->OutputCount(), opt.topk,
            opt.reject_threshold);
        const auto now = Clock::now();
        const double latency_ms =
            std::chrono::duration<double, std::milli>(now - task.capture_time).count();
        stats.AddCompleted(task.detected, latency_ms, clip_result.checksum, task.image_index,
                           task.source_name, cls);
        // 一次分类结束立即归还槽位，不必等到下一次 Pop 或线程析构。
        task.slot.Reset();
      }
      }, wake);
      clip_alive.fetch_sub(1);
    });
  }

  // 等待线程全部结束后再读取最终统计、重抛工作异常，避免使用仍在被修改的数据。
  producer.join();
  for (auto& t : yolo_threads) t.join();
  for (auto& t : clip_threads) t.join();
  failure.Rethrow();
  // 正常退出必须回收全部输入槽；这是跨队列资源生命周期的最后一道一致性检查。
  if (clip_slots.FreeCount() != static_cast<size_t>(opt.clip_buffers))
    throw std::logic_error("not all CLIP input slots returned at shutdown");
  const auto wall_end = Clock::now();
  PrintReport(opt, stats, image_paths,
              std::chrono::duration<double, std::milli>(wall_end - wall_start).count());
}


}
