// RKNN 执行封装：初始化/复制 context、输入绑定、NPU 准入、输出读取和模型后处理。
// 每个 worker 独占 runner；输出缓存在下一次 Run 时被覆盖，分类须在此之前完成。

#pragma once
#include "preprocess.hpp"
#include "npu_arbiter.hpp"

namespace yolo_clip {

// 这里选择 NPU 核，不是 CPU 线程亲和性；adaptive 的实际核在每次 RunRaw 前决定。
inline rknn_core_mask YoloCoreMask(int worker_id, const std::string& mode = "legacy") {
  if (mode == "core0") {
    return RKNN_NPU_CORE_0;
  }
  if (mode == "core1") {
    return RKNN_NPU_CORE_1;
  }
  if (mode == "core2") {
    return RKNN_NPU_CORE_2;
  }
  if (mode == "all") {
    return RKNN_NPU_CORE_0_1_2;
  }
  if (mode == "auto") {
    return RKNN_NPU_CORE_AUTO;
  }
  return worker_id % 2 == 0 ? RKNN_NPU_CORE_0 : RKNN_NPU_CORE_1;
}

inline rknn_core_mask ClipCoreMask(int worker_id, const std::string& mode) {
  if (mode == "spread") {
    const rknn_core_mask cores[] = {RKNN_NPU_CORE_0, RKNN_NPU_CORE_1, RKNN_NPU_CORE_2};
    return cores[worker_id % 3];
  }
  if (mode == "split12") {
    return worker_id % 2 ? RKNN_NPU_CORE_2 : RKNN_NPU_CORE_1;
  }
  if (mode == "core0") {
    return RKNN_NPU_CORE_0;
  }
  if (mode == "core1") {
    return RKNN_NPU_CORE_1;
  }
  if (mode == "core2") {
    return RKNN_NPU_CORE_2;
  }
  if (mode == "all") {
    return RKNN_NPU_CORE_0_1_2;
  }
  if (mode == "auto") {
    return RKNN_NPU_CORE_AUTO;
  }
  switch (worker_id % 4) {
    case 0:
      return RKNN_NPU_CORE_2;
    case 1:
      return RKNN_NPU_CORE_0;
    case 2:
      return RKNN_NPU_CORE_1;
    default:
      return RKNN_NPU_CORE_2;
  }
}

// 公共资源按依赖顺序释放：先销毁导入内存和输入内存，最后销毁引用它们的 context。
class BaseRunner {
 public:
  BaseRunner() = default;

  ~BaseRunner() {
    for (auto& entry : imported_inputs_) {
      rknn_destroy_mem(ctx_, entry.second.memory);
    }
    imported_inputs_.clear();
    if (input_mem_ != nullptr) {
      rknn_destroy_mem(ctx_, input_mem_);
      input_mem_ = nullptr;
    }
    if (initialized_) {
      rknn_destroy(ctx_);
    }
  }

  BaseRunner(const BaseRunner&) = delete;
  BaseRunner& operator=(const BaseRunner&) = delete;

  void InitFromModel(const std::vector<uint8_t>& model_bytes, rknn_core_mask core_mask) {
    int ret = rknn_init(&ctx_,
                        const_cast<uint8_t*>(model_bytes.data()),
                        static_cast<uint32_t>(model_bytes.size()),
                        0,
                        nullptr);
    if (ret != RKNN_SUCC) {
      throw std::runtime_error("rknn_init failed: " + std::to_string(ret));
    }
    initialized_ = true;
    PostInit(core_mask);
  }

  // 复制执行上下文供其他 worker 使用，避免多个线程并发操作同一个 ctx_。
  void InitFromContext(rknn_context* root_ctx, rknn_core_mask core_mask) {
    int ret = rknn_dup_context(root_ctx, &ctx_);
    if (ret != RKNN_SUCC) {
      throw std::runtime_error("rknn_dup_context failed: " + std::to_string(ret));
    }
    initialized_ = true;
    PostInit(core_mask);
  }

  rknn_context* MutableContext() {
    return &ctx_;
  }

  void EnableIoMem(bool enabled) {
    use_io_mem_ = enabled;
  }

  void EnableAdaptiveCore(bool enabled) {
    adaptive_core_ = enabled;
  }

  void EnableDirectInput(const std::string& heap, bool allow_quantized = false) {
    direct_heap_ = heap;
    allow_quantized_input_ = allow_quantized;
  }

  const rknn_tensor_attr& NativeInputAttr() const {
    return native_input_attr_;
  }

  double LastGateWaitMs() const {
    return last_gate_wait_ms_;
  }

  // 逻辑张量字节数与 SDK 要求的物理分配大小可能不同；原生输入要满足 size_with_stride。
  size_t InputAllocationBytes(size_t packed_bytes) const {
    if (direct_heap_.empty()) {
      return packed_bytes;
    }
    if (native_input_attr_.n_elems * sizeof(uint16_t) != packed_bytes) {
      throw std::runtime_error("direct input tensor size does not match model");
    }
    return std::max(packed_bytes, static_cast<size_t>(native_input_attr_.size_with_stride));
  }

  // 同一 context 按 FD 缓存导入结果；ImportedInput 同时持有 shared_ptr，防止 FD 提前关闭并复用。
  void RunDma(const std::shared_ptr<camera_dma::DmaBuffer>& buffer, double& run_ms) {
    auto found = imported_inputs_.find(buffer->Fd());
    if (found == imported_inputs_.end()) {
      rknn_tensor_mem* mem =
          rknn_create_mem_from_fd(ctx_, buffer->Fd(), buffer->Data(), buffer->Size(), 0);
      if (!mem) {
        throw std::runtime_error("rknn_create_mem_from_fd failed");
      }
      try {
        found = imported_inputs_.emplace(buffer->Fd(), ImportedInput{buffer, mem}).first;
        // FD 导入已成功，但 map 插入仍可能因分配失败而抛错；先销毁临时 SDK 句柄再原样重抛。
      }
      catch (...) {
        rknn_destroy_mem(ctx_, mem);
        throw;
      }
    }
    rknn_tensor_attr attr = native_input_attr_;
    // 原生绑定跳过 SDK 输入转换；因此调用方必须已经正确处理布局、数据类型和归一化。
    attr.pass_through = 1;
    const int ret = rknn_set_io_mem(ctx_, found->second.memory, &attr);
    if (ret != RKNN_SUCC) {
      throw std::runtime_error("bind native DMA input failed: " + std::to_string(ret));
    }
    RunRaw(nullptr, 0, RKNN_TENSOR_FLOAT16, RKNN_TENSOR_NHWC, run_ms, true);
  }

 protected:
  void PostInit(rknn_core_mask core_mask) {
    // 两种初始化方式最终走同一查询流程；属性由实际 RKNN 模型给出，不能仅凭文件名推断。
    core_bits_ = static_cast<uint32_t>(core_mask);
    int ret = rknn_set_core_mask(ctx_, core_mask);
    if (ret != RKNN_SUCC) {
      throw std::runtime_error("rknn_set_core_mask failed: " + std::to_string(ret));
    }

    rknn_input_output_num io_num;
    std::memset(&io_num, 0, sizeof(io_num));
    ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
      throw std::runtime_error("query io num failed: " + std::to_string(ret));
    }
    // 本封装的模型契约仅接受单输入/单输出，不能直接用于任意多输出 YOLO 导出。
    if (io_num.n_input != 1 || io_num.n_output != 1) {
      throw std::runtime_error("unexpected tensor count");
    }

    std::memset(&input_attr_, 0, sizeof(input_attr_));
    input_attr_.index = 0;
    ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_attr_, sizeof(input_attr_));
    if (ret != RKNN_SUCC) {
      throw std::runtime_error("query input attr failed: " + std::to_string(ret));
    }

    std::memset(&output_attr_, 0, sizeof(output_attr_));
    output_attr_.index = 0;
    ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attr_, sizeof(output_attr_));
    if (ret != RKNN_SUCC) {
      throw std::runtime_error("query output attr failed: " + std::to_string(ret));
    }

    output_float_.resize(output_attr_.n_elems);
    initialized_ = true;
    if (!direct_heap_.empty()) {
      native_input_attr_ = {};
      native_input_attr_.index = 0;
      ret = rknn_query(
          ctx_, RKNN_QUERY_NATIVE_NHWC_INPUT_ATTR, &native_input_attr_, sizeof(native_input_attr_));
      // 默认只接受经过验证的紧凑 FP16 NHWC；INT8 开关供工具诊断使用，不是默认部署输入。
      const bool type_supported =
          native_input_attr_.type == RKNN_TENSOR_FLOAT16 ||
          (allow_quantized_input_ && native_input_attr_.type == RKNN_TENSOR_INT8);
      if (ret != RKNN_SUCC || !type_supported || native_input_attr_.fmt != RKNN_TENSOR_NHWC ||
          native_input_attr_.n_dims != 4 || native_input_attr_.dims[0] != 1 ||
          native_input_attr_.dims[3] != 3 ||
          (native_input_attr_.w_stride != 0 &&
           native_input_attr_.w_stride != native_input_attr_.dims[2])) {
        throw std::runtime_error(
            "zero-copy requires packed native FP16 NHWC input; model layout unsupported");
      }
      if (native_input_attr_.type == RKNN_TENSOR_INT8 &&
          (native_input_attr_.qnt_type != RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC ||
           !std::isfinite(native_input_attr_.scale) || native_input_attr_.scale <= 0 ||
           native_input_attr_.size_with_stride != native_input_attr_.n_elems)) {
        throw std::runtime_error("native INT8 experiment requires packed affine input");
      }
    }
  }

  // 统一三种输入路径：已绑定原生 DMA、复制到 SDK IO 内存、普通宿主输入提交。
  void RunRaw(const uint8_t* input_buf,
              uint32_t input_size,
              rknn_tensor_type type,
              rknn_tensor_format fmt,
              double& run_ms,
              bool already_bound = false) {
    int ret = RKNN_SUCC;
    // already_bound 仅由 RunDma 在绑定成功后传入，避免重复调用输入转换或复制。
    if (already_bound) {
      // CPU normalization already wrote directly into the imported native tensor.
    } else if (use_io_mem_) {
      // io-mem 仍有一次 CPU memcpy；直接写入导入的 DMA 张量才省去该复制。
      EnsureInputMem(input_size);
      std::memcpy(input_mem_->virt_addr, input_buf, input_size);
      copy_metrics.InputCopy(input_size);
    } else {
      rknn_input input;
      std::memset(&input, 0, sizeof(input));
      input.index = 0;
      input.buf = const_cast<uint8_t*>(input_buf);
      input.size = input_size;
      input.pass_through = 0;
      input.type = type;
      input.fmt = fmt;

      ret = rknn_inputs_set(ctx_, 1, &input);
      copy_metrics.HostSubmission(input_size);
      if (ret != RKNN_SUCC) {
        throw std::runtime_error("rknn_inputs_set failed: " + std::to_string(ret));
      }
    }

    rknn_output output;
    std::memset(&output, 0, sizeof(output));
    // SDK 将输出转换为 float 写入预分配 vector，后处理不依赖模型内部量化类型。
    output.want_float = 1;
    // 输出由 output_float_ 的连续内存承载；SDK 可写入该地址，但不负责释放 vector 的内存。
    output.is_prealloc = 1;
    output.index = 0;
    output.buf = output_float_.data();
    output.size = static_cast<uint32_t>(output_float_.size() * sizeof(float));

    // run_ms 从准入等待前开始，到输出获取结束；gate_wait_ms 是其中的等待/动态核设置部分。
    auto start = Clock::now();
    // 许可一直活到本次请求结束；即使 SDK 抛错，也会通过 RAII 释放占用核。
    auto permit = adaptive_core_ ? npu_arbiter.AcquireAnyCore() : npu_arbiter.Acquire(core_bits_);
    if (adaptive_core_) {
      const int change = rknn_set_core_mask(ctx_, static_cast<rknn_core_mask>(permit.Mask()));
      if (change != RKNN_SUCC) {
        throw std::runtime_error("dynamic NPU mask change failed: " + std::to_string(change));
      }
    }
    last_gate_wait_ms_ = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    ret = rknn_run(ctx_, nullptr);
    if (ret != RKNN_SUCC) {
      throw std::runtime_error("rknn_run failed: " + std::to_string(ret));
    }
    ret = rknn_outputs_get(ctx_, 1, &output, nullptr);
    if (ret != RKNN_SUCC) {
      throw std::runtime_error("rknn_outputs_get failed: " + std::to_string(ret));
    }
    auto end = Clock::now();
    run_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // 读取完成后结束 SDK 的输出访问；预分配的向量仍保留给后处理和下一次执行复用。
    ret = rknn_outputs_release(ctx_, 1, &output);
    if (ret != RKNN_SUCC) {
      throw std::runtime_error("rknn_outputs_release failed: " + std::to_string(ret));
    }
  }

  void EnsureInputMem(uint32_t input_size) {
    if (input_mem_ != nullptr && input_mem_->size >= input_size) {
      return;
    }
    if (input_mem_ != nullptr) {
      rknn_destroy_mem(ctx_, input_mem_);
      input_mem_ = nullptr;
    }
    input_mem_ = rknn_create_mem(ctx_, input_size);
    if (input_mem_ == nullptr) {
      throw std::runtime_error("rknn_create_mem failed");
    }
    rknn_tensor_attr attr = input_attr_;
    attr.type = RKNN_TENSOR_FLOAT16;
    attr.fmt = RKNN_TENSOR_NHWC;
    const int ret = rknn_set_io_mem(ctx_, input_mem_, &attr);
    if (ret != RKNN_SUCC) {
      throw std::runtime_error("rknn_set_io_mem failed: " + std::to_string(ret));
    }
  }

  rknn_context ctx_ = 0;
  rknn_tensor_attr input_attr_;
  rknn_tensor_attr output_attr_;
  std::vector<float> output_float_;
  rknn_tensor_mem* input_mem_ = nullptr;
  bool use_io_mem_ = false;
  bool initialized_ = false;
  uint32_t core_bits_ = 0;
  bool adaptive_core_ = false;
  bool allow_quantized_input_ = false;
  double last_gate_wait_ms_ = 0.0;
  std::string direct_heap_;
  rknn_tensor_attr native_input_attr_{};

  struct ImportedInput {
    std::shared_ptr<camera_dma::DmaBuffer> buffer;
    rknn_tensor_mem* memory;
  };

  std::map<int, ImportedInput> imported_inputs_;
  std::shared_ptr<camera_dma::DmaBuffer> direct_input_;
};

// YOLO 预处理固定 640×640，后处理只取置信度最高的单个候选框。
class YoloRunner : public BaseRunner {
 public:
  explicit YoloRunner(bool use_rga, normalization::Mode mode = normalization::Mode::Scalar)
      : use_rga_(use_rga), mode_(mode) {
  }

  YoloResult Run(const cv::Mat& image_bgr, float conf_threshold, double& preprocess_ms) {
    PrepareInput(image_bgr, preprocess_ms);
    return Infer(image_bgr.cols, image_bgr.rows, conf_threshold);
  }

  YoloResult RunCamera(const camera_dma::Frame& frame,
                       camera_dma::Preprocessor& prep,
                       float conf_threshold,
                       double& preprocess_ms) {
    const auto start = Clock::now();
    const int width = frame.owner->Width(), height = frame.owner->Height();
    letterbox_scale_ = 640.0f / std::max(width, height);
    prep.Convert(frame);
    auto& rgb = prep.YoloRgb();
    uint8_t* destination = InputStorage();
    std::unique_ptr<camera_dma::CpuAccess> write;
    if (direct_input_) {
      write = std::make_unique<camera_dma::CpuAccess>(*direct_input_, DMA_BUF_SYNC_WRITE);
    }
    PackRgbFp16(rgb, destination, false, mode_);
    if (write) {
      write->Finish();
    }
    preprocess_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    return Infer(width, height, conf_threshold);
  }

 private:
  uint8_t* InputStorage() {
    const size_t bytes = 640 * 640 * 3 * sizeof(uint16_t);
    if (!direct_heap_.empty()) {
      if (!direct_input_) {
        direct_input_ =
            std::make_shared<camera_dma::DmaBuffer>(direct_heap_, InputAllocationBytes(bytes));
      }
      return static_cast<uint8_t*>(direct_input_->Data());
    }
    input_bytes_.resize(bytes);
    return input_bytes_.data();
  }

  YoloResult Infer(int width, int height, float conf_threshold) {
    double run_ms = 0.0;
    if (direct_input_) {
      RunDma(direct_input_, run_ms);
    } else {
      RunRaw(input_bytes_.data(),
             static_cast<uint32_t>(input_bytes_.size()),
             RKNN_TENSOR_FLOAT16,
             RKNN_TENSOR_NHWC,
             run_ms);
    }

    YoloResult result;
    result.run_ms = run_ms;
    result.gate_wait_ms = LastGateWaitMs();
    const float* out = output_float_.data();
    // 约定输出为按通道展开的 [5,N]：cx、cy、w、h、confidence；这里没有多框 NMS。
    const int count = static_cast<int>(output_attr_.n_elems / 5);
    float best_conf = -1.0f;
    int best_idx = -1;
    for (int i = 0; i < count; ++i) {
      const float conf = out[4 * count + i];
      if (conf > best_conf) {
        best_conf = conf;
        best_idx = i;
      }
    }
    if (best_idx < 0 || best_conf < conf_threshold) {
      // 即使未检出也返回合法整图区域；调用方再根据 fallback 选项决定是否继续 CLIP。
      result.detected = false;
      result.roi = cv::Rect(0, 0, width, height);
      return result;
    }

    const float cx = out[0 * count + best_idx];
    const float cy = out[1 * count + best_idx];
    const float w = out[2 * count + best_idx];
    const float h = out[3 * count + best_idx];
    // 预处理左上角对齐、仅右/下补边，因此还原坐标只除 scale，不减居中 padding 偏移。
    const float x1 = std::max(0.0f, (cx - w * 0.5f) / letterbox_scale_);
    const float y1 = std::max(0.0f, (cy - h * 0.5f) / letterbox_scale_);
    const float x2 = std::min(static_cast<float>(width - 1), (cx + w * 0.5f) / letterbox_scale_);
    const float y2 = std::min(static_cast<float>(height - 1), (cy + h * 0.5f) / letterbox_scale_);
    cv::Rect rect(cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
                  cv::Point(static_cast<int>(x2), static_cast<int>(y2)));
    if (rect.width <= 1 || rect.height <= 1) {
      result.detected = false;
      result.roi = cv::Rect(0, 0, width, height);
      return result;
    }

    // 浮点框坐标已转换成整数矩形，再与原图边界求交，确保后续 cv::Mat ROI 不越界。
    result.detected = true;
    result.roi = rect & cv::Rect(0, 0, width, height);
    return result;
  }

  void PrepareInput(const cv::Mat& image_bgr, double& preprocess_ms) {
    const auto prep_start = Clock::now();
    const int src_h = image_bgr.rows;
    const int src_w = image_bgr.cols;
    const int side = std::max(src_h, src_w);
    letterbox_scale_ = 640.0f / static_cast<float>(side);

    if (canvas_.rows != side || canvas_.cols != side) {
      canvas_.create(side, side, CV_8UC3);
    }
    // CPU letterbox 每帧填背景并复制原图；CopyMetrics 只统计显式的原图复制部分。
    canvas_.setTo(cv::Scalar(114, 114, 114));
    image_bgr.copyTo(canvas_(cv::Rect(0, 0, src_w, src_h)));
    copy_metrics.Letterbox(static_cast<size_t>(src_w) * src_h * 3);

    if (resized_.rows != 640 || resized_.cols != 640) {
      resized_.create(640, 640, CV_8UC3);
    }
    bool rga_ok = false;
    if (use_rga_) {
      rga_ok = ResizeWithRga(canvas_, resized_);
    }
    if (!rga_ok) {
      cv::resize(canvas_, resized_, cv::Size(640, 640), 0, 0, cv::INTER_LINEAR);
    }
    if (rgb_.rows != 640 || rgb_.cols != 640) {
      rgb_.create(640, 640, CV_8UC3);
    }
    bool color_ok = false;
    if (use_rga_) {
      color_ok = ConvertColorWithRga(resized_, rgb_, RK_FORMAT_BGR_888, RK_FORMAT_RGB_888);
    }
    if (!color_ok) {
      cv::cvtColor(resized_, rgb_, cv::COLOR_BGR2RGB);
    }

    auto* dst = reinterpret_cast<uint16_t*>(InputStorage());
    std::unique_ptr<camera_dma::CpuAccess> write;
    if (direct_input_) {
      write = std::make_unique<camera_dma::CpuAccess>(*direct_input_, DMA_BUF_SYNC_WRITE);
    }
    normalization::Run(rgb_.data, 640, 640, rgb_.step, dst, false, mode_);
    copy_metrics.Normalized(640 * 640 * 3 * sizeof(uint16_t));
    if (write) {
      write->Finish();
    }
    preprocess_ms = std::chrono::duration<double, std::milli>(Clock::now() - prep_start).count();
  }

  bool use_rga_ = false;
  normalization::Mode mode_;
  float letterbox_scale_ = 1.0f;
  cv::Mat canvas_;
  cv::Mat resized_;
  cv::Mat rgb_;
  std::vector<uint8_t> input_bytes_;
};

// CLIP runner 只运行图像塔并保留 embedding；离线文本特征匹配由 TextBank 完成。
class ClipRunner : public BaseRunner {
 public:
  size_t InputBytes() const {
    return ClipPreprocessor::InputBytes();
  }

  size_t OutputCount() const {
    return output_float_.size();
  }

  const float* OutputData() const {
    return output_float_.data();
  }

  ClipResult RunPrepared(const uint8_t* prepared_bytes) {
    double run_ms = 0.0;
    RunRaw(prepared_bytes,
           static_cast<uint32_t>(InputBytes()),
           RKNN_TENSOR_FLOAT16,
           RKNN_TENSOR_NHWC,
           run_ms);
    ClipResult result;
    result.run_ms = run_ms;
    result.gate_wait_ms = LastGateWaitMs();
    result.checksum = ChecksumFloat(output_float_.data(), output_float_.size());
    return result;
  }

  // 诊断入口：提交 float32 输入以比较 SDK 输入转换行为，常规流水线使用 FP16。
  ClipResult RunPreparedFloat32(const float* input) {
    ClipResult result;
    RunRaw(reinterpret_cast<const uint8_t*>(input),
           224 * 224 * 3 * sizeof(float),
           RKNN_TENSOR_FLOAT32,
           RKNN_TENSOR_NHWC,
           result.run_ms);
    result.gate_wait_ms = LastGateWaitMs();
    result.checksum = ChecksumFloat(output_float_.data(), output_float_.size());
    return result;
  }

  ClipResult RunPreparedDma(const std::shared_ptr<camera_dma::DmaBuffer>& buffer) {
    ClipResult result;
    RunDma(buffer, result.run_ms);
    result.gate_wait_ms = LastGateWaitMs();
    result.checksum = ChecksumFloat(output_float_.data(), output_float_.size());
    return result;
  }
};

}  // namespace yolo_clip
