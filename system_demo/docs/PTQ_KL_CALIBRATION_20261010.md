# CLIP PTQ：normal 与 KL 校准的完整板端对照

后续分段保护已完成：中段验证69.17% / 5.50FPS，但冻结后独立划分只有64.17%，
低于同划分KL的68.33%（FP16为70.83%），分段候选拒绝采用。
详见 `PTQ_KL_PARTIAL_MLP_20261011.md`，不把验证集局部收益当成稳健优化。

## 结论

在相同混合精度策略下，将校准算法由 `normal` 改为 `kl_divergence`，完成重新校准、
导出、120 张板端验证、实际算子图核验及三种模型的同轮短时测速。

KL 准确率从 59.17% 升至 66.67%，多 9 张正确，追回原 15 个百分点损失的一半；
三 context CLIP-only 吞吐为 6.20 FPS，normal 为 6.27 FPS，普通 FP16 为 3.81 FPS。
这是校准参数选择确实影响精度的证据，不是通过扩大 FP16 范围、缩小网络或改变输入
取得的收益。尚未证明全部误差仅来自激活，也没有完成真实场景或最终测试集验收。
KL 仍比 FP16 低 7.5 个百分点，默认模型不替换。

## 单变量与有效性核验

- 同一官方参考 ViT-L/14 FP32 ONNX、224 输入、24 个 block、原 120 张校准图及文本 bank。
- 源模型、校准列表和全部校准 tensor 的 SHA、Toolkit2 2.3.2 及其它配置保持一致。
- W8A8、权重 `channel`、optimization 3、mean=0/std=1 保持不变，唯一配置变化是算法。
- KL 新执行 `hybrid_quantization_step1`，约 715 秒；没有复用 normal IR 的范围或 scale。
- 完成新的 step1 model/data/cfg SHA 校验，只迁移原 126 个 FP16 保护选择。
- 构建完成，六张模拟器有限值检查通过；模拟器检查不替代板端精度验收。
- 24 个融合 SDPA 的实际算子均为 INT8 NPU，绕开已复现的混合 FP16 SDPA 执行异常。
- 板端 473 条算子的 `(OpType, DataType, Target, FullName)` 多重集与 normal 完全一致。
  该核验不比较性能计数，也不宣称能检查驱动内所有隐藏的融合实现细节。
- 120 张验证图的输入 FP16 位模式及样本顺序与普通 FP16 基线逐文件一致。
- 新候选 CPU 项仅为 InputOperator/OutputOperator，没有观察到大量 CPU fallback。
- 最终测试集未用于本轮校准、策略选择或评价；没有采用 QAT checkpoint。

## 120 张验证精度

官方参考 CLIP＋TrashNet 六类零样本分类，每类 20 张。这不是历史部署模型的
四分类＋unknown 真实场景；小样本结果不代表已通过泛化或统计显著性验收。

| 模型 | 正确数/120 | 准确率 | macro-F1 | 对 FP16 特征余弦 | 模型 MB（十进制） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 普通 FP16 | 89 | 74.17% | 0.6991 | 1.0000 | 623.05 |
| 相同混合策略，normal | 71 | 59.17% | 0.5384 | 0.7809 | 321.15 |
| 相同混合策略，KL | 80 | 66.67% | 0.6010 | 0.8242 | 321.11 |

KL 对 FP16 top1 一致率为 77.5%，最低样本特征余弦为 0.6314。
最终 embedding RMSE 从 0.4449 降至 0.3876。没有以特征余弦替代分类指标。

## 校准范围变化：收益与裁剪代价同时存在

523 个参数张量中 417 个参数配置发生变化。这是量化参数的变化，不是 417 个算子
改变了计算精度。不能将算法效果独占归因于一处 GELU、某个权重或单个 scale。

从原来保存的一张 FP32 golden 验证 trace（cardboard_020.png）重放 24 个 GELU 的
近似 QDQ：全部 24 层步长和归零比例减小，全部出现少量饱和裁剪；仅 8 层的
局部 QDQ RMSE 下降。它不是 120 张校准的完整分布，也不是逐位 RKNN 内核模拟。

第 12 层示例：

| 指标 | normal | KL |
| --- | ---: | ---: |
| 校准范围 | [-0.1636, 58.3121] | [-0.1636, 14.6065] |
| INT8 scale | 0.2293165 | 0.0579220 |
| 单样本 QDQ 零值比例 | 64.84% | 13.26% |
| 单样本 QDQ 裁剪比例 | 0 | 约 0.002375% |
| 单样本 QDQ RMSE | 0.07211 | 0.11124 |
| 单样本 QDQ 余弦 | 0.95558 | 0.88708 |

KL 让主体小值更细致，但少数大值被裁剪，局部误差反而增大；完整模型分类仍改善。
这说明归零比例、局部 RMSE 与任务正确率不是可相互替代的指标。不能据此直接裁剪
所有大值，也不能反过来用单层 RMSE 增大否定整个 KL 候选。

## 独立同轮短时测速

Runtime 2.3.2 使用工程内隔离库，系统 Runtime 2.3.0 和 driver 0.9.6 不替换。
相同固定预编码 FP16 tensor、host 提交，两轮正序/反序，各配置 8 秒，初始化和
热身除外。不启用 perf 收集，没有其它并行 NPU 验证或任务。

| 配置 | FP16 | normal 混合版 | KL 混合版 |
| --- | ---: | ---: | ---: |
| 1 context/all，FPS 均值 | 2.202 | 2.935 | 2.947 |
| 3 context/spread，FPS 均值 | 3.813 | 6.265 | 6.201 |
| 3 context 两轮 FPS | 3.792 / 3.834 | 6.302 / 6.229 | 6.215 / 6.187 |

KL 三 context 吞吐比同轮 FP16 高约 62.6%，比 normal 低约 1.0%。两轮短基准不足以
宣称长期稳定性、统计等速或显著速度差异；当前没有明显吞吐损失的证据。
KL 各 spread worker 的平均 run_ms 约 444–457 ms；1000/6.20≈161 ms 是聚合
完成间隔，不是单张 CLIP 延迟。这些也不是含 YOLO、RGA、DMA、摄像头的系统 FPS。
分类报告内跨轮 mean_run_ms/speed_ratio 字段不作为独立加速结论。

## 对参考文章及此前判断的落实

知乎文章“模型量化_RKNN”的校准、混合精度与后端适配建议已与官方文档对照。
本轮证明“完成 PTQ”不等于“normal 的范围估计最适合这个模型”，但也不证明 KL
普遍优于 normal。文章的 `fbgemm` QAT 和旧版 `asym_u8` 示例不直接套用到 RK3588。
当前接口以 Toolkit2 2.3.2 为准，不改网络结构或使用文章提到的轻量化架构。

此前混合 FP16 SDPA 的均匀输出是另一个执行正确性问题；本轮继续避免该路径，
不能把 KL 的精度收益当成 SDK/driver 错误已修复，也不用于合理化原错误分支。

## 下一步

1. 以 KL 作为下一轮实验底座，仍保留 normal 与 FP16 对照；默认模型不变。
2. 在 120 张校准集上统计各层下投影输入分布与离群值，再做少量下投影层 FP16
   保护对照，寻找能追回剩余精度、同时保持至少约 5 FPS 的组合。必须同时验收实际
   dtype、分类和同轮速度，不依据一张 trace 直接宣称已找到最敏感层。
3. 检查 SDK 量化配置的约束后，可探索逐层范围选择，而不是统一采用更激进的裁剪。
   normal/KL 混合范围或 MMSE 尚未构建验收，不能写成已实现的精度提升。
4. QAT/蒸馏需要匹配实际 SDPA 与 MLP 量化边界，最终以导出后板端结果为准。

## 代码、证据与交付状态

- 新校准：`quantization/build_calibration_ir.py`。
- 单变量指纹：`quantization/record_ir_provenance.py`（默认 normal 行为保持兼容）。
- 精度选择迁移：`quantization/calibration_policy.py`，5 项回归检查全部通过；
  在实际 normal IR 上迁移后与原策略完全相同。
- 范围诊断：`quantization/compare_calibration_ranges.py`。
- 新 IR/日志/指纹：`quantization/artifacts/calibration_algorithms_20261010/kl_divergence/`。
- 完整候选/策略/SHA/模拟器：`quantization/artifacts/sdpa_safe_full_20261010/sensitive_kl_except_sdpa/`。
- 分类指标：`quantization/artifacts/accuracy_calibration_kl_20261010.json`。
- 范围/QDQ：`quantization/artifacts/calibration_kl_ranges_20261010.json`。
- 原始板端 embeddings/manifest：`quantization/artifacts/quant_sdpa_safe_sensitive_kl_except_sdpa_ieee/`。
- 算子核验/输入一致记录/验证状态：`results/sdpa_safe_full_20261010/`。
- 原始同轮测速：上述目录 `speed_calibration_kl/speed_summary.json`。

转换、验证和测速已经结束；相关 Python 编译检查和三个原有板端 CTest 全部通过。
完成本机模型 SHA、板端验证状态和无活动实验核验后，仅删除板端唯一临时槽中的
KL 模型副本，本机完整文件可重新上传恢复。旧模型、原工程和系统配置未修改。
本轮沿用 diagnose 的固定输入、单变量和板端反馈环，没有新增项目管理配置。
