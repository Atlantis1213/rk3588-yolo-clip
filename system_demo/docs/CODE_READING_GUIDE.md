# 中文代码阅读路线

项目分为在线 C++ 推理与离线 Python 量化实验两部分。建议先读当前主工程
`system_demo/`，再根据需要回看根目录的历史单文件版本。

源码采用分行的 C++ 风格：条件和循环使用独立的大括号块，一行只放一条执行语句，
函数和 lambda 的函数体展开，`catch` 另起一行。仓库根目录的 `.clang-format` 固定了这些
规则；使用 clang-format 18 可对修改过的文件执行 `clang-format -i 文件路径` 保持一致。
`for` 头中的两个分号及初始化列表仍按其语法保留，不拆成独立语句。

## 1. 先把一帧的路径串起来

1. `src/main.cpp`：解析配置，加载模型字节和离线文本特征，选择运行模式。
2. `include/types.hpp`：理解 `Options`、`FrameTask`、`ClipTask` 和结果结构。
3. `src/pipeline.cpp`：先读 `RunSerial` 理解顺序，再读 `RunPool` 理解并行。
4. `include/model_runner.hpp`：观察 YOLO 如何产生 ROI，CLIP 如何产生 embedding。
5. `include/text_bank.hpp`：图像和文本特征归一化后点积，得到类别排名。

线程池中的一帧路径为：

```text
摄像头 / 文件输入
    → FrameTask → frame_queue
    → YOLO worker：预处理、检测、选 ROI、准备 CLIP 输入
    → ClipTask（持有输入槽租约）→ clip_queue
    → CLIP worker：图像编码、文本匹配、记录结果、归还输入槽
```

`RunSerial` 中只有模型处理顺序串行，输入仍由独立生产者线程投递。
`RunPool` 中不同帧可处在不同阶段；worker 数量增加后的收益取决于 NPU、CPU、
带宽和排队情况，并不保证线性增长。

## 2. 重点跟踪两种资源的归还

**摄像头帧**：`v4l2_capture.hpp` 的 `Acquire` 将缓冲从驱动出队，返回的
`Frame` 由 `shared_ptr` 持有。最后一个引用析构时才调用 `Requeue`，让驱动重新使用
该缓冲。`pipeline.cpp` 在 CLIP 输入准备好后尽早释放原始帧。

**CLIP 输入槽**：`buffer_pool.hpp` 的 `TryAcquire` 返回可移动、不可复制的
`Lease`。YOLO worker 把租约移动到 `ClipTask`，CLIP worker 消费后 `Reset`。
任务在队列中被丢弃、被关闭队列拒收或因异常析构时，也通过相同的 RAII 机制归还槽位。

阅读 `pipeline_queue.hpp` 时留意：满队列丢弃的是最旧的待处理任务；`Close`
会拒收新任务并唤醒消费者，但已有任务仍可被 `Pop` 排空。外部回调、任务赋值及
被丢任务析构放在队列锁外，避免资源释放时形成不必要的锁嵌套。

## 3. 区分图像、张量与设备缓存

| 阶段 | 数据约定 | 对应代码 |
| --- | --- | --- |
| 摄像头原始帧 | 协商的 YUYV/NV12，行宽可能有 padding | `v4l2_capture.hpp` |
| RGA 中间图 | BGR/RGB，物理 stride 可大于逻辑宽度 | `dma_buffer.hpp`、`rga_preprocess.hpp` |
| YOLO 输入 | 左上角贴图，右/下补 114，resize640，RGB / 255 | `model_runner.hpp` |
| CLIP 输入 | ROI 直接 resize224，RGB / 255 后减均值、除标准差 | `preprocess.hpp` |
| 常规输入张量 | 紧凑 NHWC、FP16 | `normalization.hpp` |
| CLIP 输出 | 浮点图像 embedding，再与离线文本特征匹配 | `model_runner.hpp`、`text_bank.hpp` |

DMA-BUF 让设备共享同一缓冲，但 CPU 访问仍需要 `CpuAccess` 的 START/END 缓存同步。
当前归一化由 CPU 完成；直接写入导入的输入 DMA 张量可以省掉额外输入复制。
`--io-mem` 分支仍有显式 memcpy，不能仅凭选项名称判断数据是否被复制。

`normalization.hpp` 的 LUT 为 256 种像素值预计算最终半精度位模式，与当前 scalar
共享公式。`utils.hpp` 以及根目录旧程序中保留的历史 FP16 转换存在舍入进位差异；
复现历史结果与阅读当前部署路径时，要选对应的参考。

## 4. 再理解调度和性能报告

`scheduler.hpp` 控制 CPU 线程亲和性，`npu_arbiter.hpp` 控制本进程发往 NPU 的请求准入。
二者的“核”不是同一资源。`Permit` 自动释放核占用；自适应 YOLO 从空闲 NPU 核中选择
一个执行，固定请求等待其指定掩码全部可用。

`metrics.hpp` 中 `completed_fps` 只统计完成分类的任务；`processed_fps` 还包括因无检测
目标而跳过 CLIP 的任务。端到端延迟从采集出队/任务生成附近开始，包括排队，
不包括传感器曝光与 USB 传输。模型 `run_ms` 包含准入等待，报告另列等待耗时；
聚合 FPS 的倒数不能代替单请求延迟。复制计数仅覆盖明确插桩的位置。

正常结束时，生产者关闭帧队列，最后一个 YOLO worker 关闭 CLIP 队列；异常时
`ThreadFailure` 记录首个异常、设置停止标志并关闭两级队列，主线程 join 后重抛。

## 5. 离线量化代码按问题阅读

- **参考与转换**：`quantization/export_reference.py` → `convert_rknn.py`。
  先对齐模型来源、同一输入位模式和预处理，再讨论模型误差。
- **校准与混合精度**：`calibration_selection.py` → `calibration_policy.py` →
  `partial_mlp_policy.py`。校准参数与精度策略分开管理，并核实 SDK 实际产物。
- **范围与重参数化**：`histogram_range_search.py`、`affine_reparameterization.py`。
  前者用直方图近似量化误差，后者通过相邻算子参数抵消保持浮点组合等价。
- **学习舍入**：`adaptive_rounding_core.py` → `reconstruct_rounding.py`。
  固定 scale/zero_point，学习舍入选择，使用校准激活重建线性层输出。
- **QAT/蒸馏**：`qat_visual_core.py` → `train_qat.py`。
  fake quant 是训练代理，结果仍需转换并在板端验证。
- **评估**：`numerical_fidelity.py`、`evaluate_protocol_v2.py`。
  数值方向、幅度、类别准确率和配对得失各自回答不同问题。

实验目录保留多个历史阶段的工具和机器路径。具体候选、模型来源和验收结论应结合
`MODEL_PROVENANCE.md` 及对应日期实验文档阅读，不能从“转换成功”推断已通过部署验收。

## 6. 用已有测试核对理解

`tests/test_queue_pool.cpp` 检查关闭/丢弃/异常归还、并发任务对账与租约存活期；
`test_normalization.cpp` 对照独立半精度参考并检查 stride、尾部和边界；
`test_npu_arbiter.cpp` 验证冲突阻塞、分离核并行和释放唤醒。
`legacy_slot_leak_repro.cpp` 则故意复现旧漏洞，预期失败，不是应通过的回归测试。

板端工具 `tools/dump_clip_embeddings.cpp` 会同时导出输入和输出，适合定位跨后端差异；
`tools/compare_camera.py` 轮换 A/B/C 配置测试系统表现。硬件自检和真实摄像头测试
仍需在配齐 RKNN、RGA、模型和摄像头的 RK3588 环境运行。
