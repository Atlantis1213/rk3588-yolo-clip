# RK3588 调度与 CLIP 量化实验报告（2026-10-10）

## 结论与部署状态

已实现可选的逐核准入控制、三 CLIP context 分核执行和 YOLO 自适应核分配，并完成
多配置短测。FP16 舍入修正后的三轮对照确认降低尾延迟而非提高吞吐，单独保存了
当前与历史输入版本的测量。默认 worker/core/gate 配置没有自动更改。

已下载官方 ViT-L/14、导出 ONNX，并建立独立 FP16/INT8 速度及精度评估链。
普通 INT8 有约1.62倍单次加速，但精度损失过大；自动混合量化和通道均衡也没有达到
可用精度。原始工程、模型、文本 bank 以及系统 Runtime 均保留。没有实现“5 FPS且不掉精度”。

## 1. 工程、模型和数据

- 原始工程：`/home/orangepi/yolo_clip_bench`，未修改。
- 系统优化副本：`/home/orangepi/yolo_clip_bench_systemopt`。
- 本地源码/实验：`F:\YOLO-CLIP\system_demo`。
- 未找到历史 CLIP 的 checkpoint、ONNX、转换配置或原场景数据集。
- 已下载并 SHA 校验官方 OpenAI ViT-L/14 与 TrashNet resized 数据。
- 官方模型与历史 RKNN 五样本 embedding cosine 为0.9655–0.9904，不能证明同权重转换。
  因此所有量化比较均使用同一新 ONNX 导出的新 FP16 基线，不冒充历史模型量化结果。
- Toolkit/实验 Runtime2.3.2；原安装 Runtime2.3.0、driver0.9.6 不替换。
  子进程通过 `LD_LIBRARY_PATH=./quant_runtime232` 隔离实验 Runtime，未改 governor。
- 校准、验证、最终测试各120张，六类各20张，文件级互斥、seed20261009。
  最终测试不用于选择策略。采用项目直接 resize224、固定六类文本模板。
  这是公开六类 zero-shot 分类，不是原四类垃圾+unknown 或完整 YOLO-ROI 场景准确率。

来源、模型结构、SHA、输入契约详见 `MODEL_PROVENANCE.md`；复现命令见
`../quantization/README.md`。板端本轮摄像头未连接，新增调度比较使用固定图片。

## 2. NPU 调度：三核吞吐不是单次延迟除以三

`include/npu_arbiter.hpp` 用三位 active mask、条件变量和 move-only RAII Permit 管理
本进程的核占用。固定 context 在申请的核空闲时准入；adaptive YOLO 申请任意一个空闲核，
再执行 `rknn_set_core_mask`。待准入 adaptive 请求获得优先权，减少被长 CLIP 请求持续
挤占的风险。它不控制其他进程、不修改驱动、不支持 NPU 抢占，也不是硬实时调度。

CLIP spread=0/1/2，旧 round_robin=2/0/1/2。同一 context 的一次请求允许 SDK 使用
多核，与多个 context 同时处理不同帧是两种不同的并行方式。
gate wait 计入总 run_ms，并另外报告；不能把等待移出计时后称为模型变快。

历史固定 tensor 的 CLIP-only 饱和实验：

| context / 核配置 | gate | 完成FPS |
| --- | --- | ---: |
| 1 / core0 | none | 1.516 |
| 1 / all | none | 2.366 |
| 3 / spread | none | 3.835 |
| 4 / round_robin | none | 3.896 |
| 3 / spread | per_core | 3.917 |
| 4 / round_robin | per_core | 3.812 |

以662ms单核假定计算，理想上限为 `3/0.662≈4.53 FPS`，输出间隔约221ms，
不是单帧模型延迟221ms，也不是已达到5 FPS。实际 contexts 共享资源，YOLO还占用核，
故不能保证线性三倍；尚无 DDR 带宽 profiling，不能确定地归因为“DDR瓶颈”。

修复前的完整固定图片流水线，3轮轮换对照：

| 配置 | 完成FPS | 每轮P95的均值ms |
| --- | ---: | ---: |
| 原2 YOLO+4 CLIP RR | 3.695 | 2635.44 |
| 1 YOLO adaptive+3 CLIP spread/gate | 3.609 | 1798.34 |
| 2 YOLO adaptive+3 CLIP spread/gate | 3.455 | 3118.17 |

1 YOLO adaptive 在该历史对照中 P95 降31.8%，吞吐降2.3%；并非 FPS 增益。
2 YOLO adaptive 更差。固定队列仍会积压，降低核冲突不等于消除排队。
原始证据：`results/npu_scheduling_20261009`、`results/npu_adaptive_20261009`。
这些完整图片调度数据使用修复前的输入转换，不覆盖历史记录。

修正IEEE FP16输入后的最新确认：每配置3轮、每轮输入12秒、包含队列drain完成时间，
轮换ABC/BCA/CAB；相同模型、图片、输入FPS8、frame queue4、CLIP queue/pool8及
OpenCV threads1，无并行板端推理负载：

| 配置 | 完成FPS | 每轮P95的均值ms |
| --- | ---: | ---: |
| 原2 YOLO+4 CLIP RR | 3.628 | 2604.90 |
| 1 YOLO adaptive+3 CLIP spread/gate | 3.582 | 1733.58 |
| 2 YOLO adaptive+3 CLIP spread/gate | 3.475 | 3184.79 |

新版本1 YOLO adaptive的P95降低33.45%、吞吐降低1.27%，仍是尾延迟收益。
证据：`results/npu_adaptive_ieee_20261010/summary.json`及九份含命令/原始stdout的报告。
这是每轮P95的算术平均，不是合并样本P95。短比较不用于宣称长时间稳定性或精度改善。

FP16 batch=3 的独立实验：57帧/19 batch，3.746 FPS，batch_run约800.7ms。
仅检验三个相同输入输出一致（最大差0），没有验证不同输入的顺序及完整 pipeline。
没有吞吐优势，未整合为默认策略。不能把800/3当作单帧端到端延迟。

## 3. 量化：速度与精度必须一起验收

校准为已标准化 NCHW `.npy`，FP32值经过 IEEE FP16 舍入再还原；mean=0/std=1，
避免重复归一化。各变体共享同一输入、文本 bank、Runtime和评估方式。

120张验证集结果及独立固定 tensor 测速：

| 候选 | accuracy | macro-F1 | 对FP16 embedding cosine均值 | 两轮独立FPS | 模型bytes |
| --- | ---: | ---: | ---: | --- | ---: |
| 新FP16基线 | 74.17% | 0.6991 | reference | 2.156 / 2.230 | 623049472 |
| 普通W8A8 INT8 | 18.33% | 0.1631 | 0.2972 | 3.594 / 3.588 | 319908186 |
| auto-hybrid（阈值0.995） | 17.50% | 0.1424 | 0.5052 | 2.186 / 2.165 | 618945280 |
| 通道均衡+INT8 | 15.83% | 0.1295 | 0.3010 | 3.641 / 3.587 | 319908186 |

全部 INT8 候选被拒绝。自动混合量化的大小/速度接近FP16，但精度仍严重退化；
提高局部 cosine 阈值并不保证最终分类或跨模态对齐。
普通 INT8 的早期验证日志含一次并行回归负载污染的347.8ms计时，不作为独立测速依据。
独立测速原始数据：`results/quant_speed_20261010/summary.json`。

策略选择后冻结，另用120张第三组最终测试复核：

| 模型 | accuracy | macro-F1 | mean run_ms |
| --- | ---: | ---: | ---: |
| 新FP16 | 70.83% | 0.6680 | 415.87 |
| 已拒绝的普通INT8 | 22.50% | 0.2003 | 257.10 |

INT8单次加速1.618倍，准确率下降48.33个百分点，对FP16 embedding cosine均值0.2926，
Top1一致率14.17%。不能将“变快”当作可用结果，更不能据此外推完整模型三context的FPS。
证据：`quantization/artifacts/accuracy_final_test.json`。

### 已排除与尚未定位的问题

- 输入归一化已对齐并修复FP16舍入；没有给已标准化输入再应用CLIP mean/std。
- Toolkit拒绝NHWC校准负对照，明确期望NCHW；正常校准格式符合SDK契约。
- 普通INT8模型在120张验证图上，FP16与同值FP32宿主提交输出完全一致。
  严重误差不是只由FP16宿主接口造成。
- 原生INT8 DMA诊断六样本中只有三个与SDK宿主输出cosine=1，其余约0.42–0.51；
  尚未证明该路径正确，不能作零拷贝量化验收。默认native绑定继续拒绝INT8。
- 还没有做完整 simulator/逐层 `accuracy_analysis`，没有定位第一个显著发散算子。
  因此不能断言“CLIP不能量化”或确定某个LayerNorm/Softmax就是根因。

### 常用及前沿方法的实际适用性

| 方法 | 本轮工作 | 当前结论 |
| --- | --- | --- |
| RKNN channel-wise PTQ | 已转换并实测普通W8A8 | 变快但不可用 |
| RKNN自动混合量化 | 已转换，cos阈值0.995 | 无精度恢复，也无速度收益 |
| SmoothQuant/RepQ风格等价重参数化 | 实现校准激活/权重通道均衡，alpha=0.5 | FP32等价，INT8仍失败 |
| RKNN MMSE | 已用30张分层校准启动 | 转换状态独立记录；不把未完成算失败 |
| PTQ4ViT / FQ-ViT / RepQ-ViT / RepQuant | 核对方法思路 | 特殊量化粒度/算子需SDK和硬件落地，未完整复现 |
| QAT / P4Q余弦蒸馏与prompt/adapter | 核对训练方向 | 未训练；应先定位数值问题再投入训练 |
| W8A16 / W4A16 | Toolkit config探针 | 当前RK3588环境拒绝，不是可直接部署路线 |
| early exit / 换小模型 | 非本轮默认方向 | 改变深度/架构或流程，不用于包装系统调度成果 |

通道均衡仅处理LN1→QKV、LN2→c_fc及LNpost→projection的可折叠路径，LNpre还喂残差，
因此不直接改动。FP32等价检查cosine≈1，最大绝对误差≤7.63e-6，无新增推理算子。
这是方法适配实验，不能写成完整复现SmoothQuant或RepQ论文并已恢复精度。

MMSE本轮已主动中止：运行约40分钟，日志进度65/300、约22%，尚未导出模型，
SDK估计仍需小时级转换时间。为先交付已完成结果而结束本轮实验，不属于编译算法失败
或精度不合格。状态记录：`quantization/artifacts/mmse_status.json`；完整日志和30张
校准输入保留，可重新运行，但当前脚本没有断点恢复。

## 4. FP16正确性修复与验收

旧转换在尾数进位时没有正确提高指数。green=250经CLIP公式得到约1.9998449，
旧bits=0x3c00（1.0），正确IEEE binary16=0x4000（2.0）。原先LUT只与旧scalar比较，
共同错误没有被发现，这是验证缺口，不是模型本身的问题。

当前已加入IEEE round-to-nearest ties-to-even、指数进位、subnormal、overflow、signed zero
处理及half反解码。compiler `__fp16`/`_Float16`作为独立外部参考，覆盖全部byte/channel
与舍入边界；新scalar/LUT互等，legacy模式对照保留旧程序。
8种推理回归通过；normal与ASan/UBSan的3个CTest程序全部通过。
新版和旧版表合计6 KiB，历史3 KiB说明已更正。
测试日志：`results/verification_20261010/ctest_normal.log`与`ctest_asan_ubsan.log`；
推理回归：`results/regression.json`（IEEE与legacy分别使用对应参考）。

## 5. 下一步优先级

后续推进：已完成固定样本FP32/FP16对照、输入/输出契约探针、输入量化负对照及
373条SDK逐层误差分析。归一化/残差/注意力路径成为手动混合量化候选；模拟器与板端
仍有差异，不能断言Runtime无问题。新增证据及候选状态见 `QUANT_DIAGNOSIS_20261010.md`。
下方为本报告初次交付时的优先级，不表示这些检查全部尚未开展。

1. 先建立逐层误差闭环：同一规范化输入，对比PyTorch/ONNX、RKNN FP16 simulator、
   INT8 simulator及板端；找到第一个显著发散算子，核验layout、scale/zp和融合转换。
   输入DMA契约与量化算法分开验证，不在错误的边界上继续调参。
2. 在定位结果上做手动混合量化：按误差与实测耗时选择保留FP16的敏感层，
   不仅看模型文件大小或局部cosine。每个候选同时报告分类、embedding/ranking与板端速度。
3. 若确认是量化敏感性而非接口/编译错误，再做QAT或特征余弦蒸馏，使用训练/验证/最终
   测试分离及更匹配的真实场景数据。保留FP16教师/文本bank，实测导出RKNN后是否仍保持收益。
4. 系统调度根据目标选择：吞吐优先保留已验证基线；实时交互优先评估adaptive与较短队列。
   不能通过降低输入FPS、扩大排队、跳过模型步骤或移除等待计时制造“加速”。

## 2026-10-10 下午追加：QAT 与部署排查并行

完整模型新验收：绕开混合FP16融合SDPA异常后，精度控制版70.83%（基线74.17%），
速度版59.17%、同配置CLIP-only吞吐6.25FPS（FP16基线3.82FPS）。
精度/速度尚未同时满足，不替换默认模型；协议与原始数据见 `SDPA_FULL_MODEL_20261010.md`。

再推进结果：新增三组300步QAT消融完成，投影微调将代理特征余弦提升至0.85994，
分类与未训练代理持平64.17%，尚未通过实际RKNN验收。最小前缀复现将板端差异
收敛到第一个注意力相关区域，原因未唯一定位。详见 `QAT_FOLLOWUP_20261010.md`。

4060 已完成真实300步LN-only QAT/蒸馏首轮，非仅脚本准备。
训练代理平均特征余弦0.7862→0.8075，但分类64.17%→61.67%，不采用为默认模型。
板端固定6张差分排除了本轮单核/三核及2.3.0/2.3.2版本之间的输出差异；
全浮点混合导出控制仍错误，原因未定位，下一步针对导出图变换。
方法、数据隔离、checkpoint及所有新证据见 `QAT_PILOT_20261010.md`。

## 参考资料（原列表）

- [RKNN Toolkit2 v2.3.2](https://github.com/airockchip/rknn-toolkit2/releases/tag/v2.3.2)
- [官方混合量化示例](https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit2/examples/functions/hybrid_quant)
- [官方CLIP示例](https://github.com/airockchip/rknn_model_zoo/tree/main/examples/clip)：示例FP转换不等于已验证INT8。
- [OpenAI CLIP](https://github.com/openai/CLIP)、[TrashNet](https://github.com/garythung/trashnet)
- [SmoothQuant](https://arxiv.org/abs/2211.10438)、[PTQ4ViT](https://arxiv.org/abs/2111.12293)
- [FQ-ViT](https://arxiv.org/abs/2111.13824)、[RepQ-ViT](https://arxiv.org/abs/2212.08254)
- [RepQuant](https://arxiv.org/abs/2402.05628)、[P4Q](https://arxiv.org/abs/2409.17634)
- 本地SDK输入/精度排查手册：`quantization/assets/RKNN_UserGuide_2.3.2_EN.pdf`。
