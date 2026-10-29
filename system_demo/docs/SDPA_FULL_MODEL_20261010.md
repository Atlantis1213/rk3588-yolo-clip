# 融合 Attention 保留 INT8：完整 CLIP 验收

后续分组对照完成：恢复QKV仍59.17%，恢复输出投影60.83%，恢复MLP到69.17%，
但MLP版CLIP三context吞吐仅3.97FPS。新的误差线索和下一步见 `PTQ_GROUP_ABLATION_20261010.md`。

MLP进一步拆分：仅恢复上投影59.17%、下投影65.00%；下投影版同轮吞吐4.27FPS，
FP16为3.83FPS，未达到精度与速度联合目标。详见 `PTQ_MLP_SPLIT_20261010.md`。

校准进一步对照：不扩大FP16保护范围，重新用KL校准后准确率升至66.67%，
CLIP-only三context吞吐6.20FPS，详见 `PTQ_KL_CALIBRATION_20261010.md`。

后续KL分段下投影保护：中段验证69.17% / 5.50FPS，独立划分复测64.17%，
不如KL基线68.33%（FP16为70.83%），未采用。详见 `PTQ_KL_PARTIAL_MLP_20261011.md`。

## 本轮结论

已把最小复现中的绕开策略应用到完整24-block ViT-L/14，完成三种混合精度模型的
120张板端验证、实际算子dtype核验，以及前两种模型的独立短时吞吐测试。
错误FP16融合Attention分支已在这些候选中绕开，完整模型精度明显恢复；
但没有同时达到“接近FP16精度且明显更快”，不替换默认模型。

所有实验仍为独立官方参考模型，同一原权重、ONNX、原120张校准数据及文本bank。
没有使用QAT checkpoint、缩小网络、改变224输入、跳过任何block，或删掉YOLO步骤
来宣称加速。本轮测量范围本身是CLIP-only，不含YOLO、摄像头、RGA和预处理流水线。

## 验证集精度

TrashNet六类零样本分类，120张验证，每类20张。三个候选的输入FP16位模式、图片
顺序均与原普通FP16基线逐文件核验相同。没有重新用最终测试集筛选这些策略。

| 模型 | FP16保护边界数 | 文件大小（十进制MB） | 准确率 | 对FP16特征余弦均值 |
| --- | ---: | ---: | ---: | ---: |
| 普通非量化FP16基线 | 不适用 | 623.05 | 74.17% | 1.0000 |
| 只让24个Attention为INT8，其它保持FP16 | 348 | 624.54 | 70.83% | 0.9562 |
| 归一化/残差/GELU/IO保护，Attention和大矩阵INT8 | 126 | 321.15 | 59.17% | 0.7809 |
| 上一策略额外保护stem/head三个输出 | 129 | 321.92 | 58.33% | 0.7849 |

第一种候选比普通FP16低3.33个百分点（少4张正确），macro-F1为0.67137；
第二种低15个百分点（少18张正确），macro-F1为0.53845。
第三种没有体现分类改善（比第二种少1张正确），不额外做测速或采用。
这里“保护边界数”不是实际算子数，邻接精度可由SDK传播；实际变化以算子报告为准。

与对应旧混合策略比较：仅移除24个SDPA的FP16保护，全浮点IO控制从17.50%恢复到
70.83%；敏感层/GELU/IO策略从19.17%恢复到59.17%。这不是普通纯INT8模型已恢复，
而是这两个混合策略绕开异常后改善。

三个完整候选在Runtime算子表中均核验了24个`exSDPAttention INT8 NPU`。
第一种主体矩阵仍为FP16，文件并没有变小，不能把70.83%当成纯INT8准确率。

## 独立测速：前两种候选

板端串行执行，隔离Runtime2.3.2、driver0.9.6；不改governor、驱动或系统库。
固定相同的预编码FP16 tensor，host提交，两轮正序/反序，每配置8秒，热身与
初始化不计入时间。没有开启收集perf的标志进行测速，也没有同时跑验证/其它NPU探针。
这些是Demo级短基准，不是长时间稳定性或统计显著性测试。

| 配置 | 普通FP16 | 只量化Attention | 大矩阵INT8混合版 |
| --- | ---: | ---: | ---: |
| 1 context，all核，两轮FPS均值 | 2.156 | 2.132 | 3.127 |
| 3 context，spread三核，两轮FPS均值 | 3.821 | 3.637 | 6.246 |
| 3 context，两轮FPS范围 | 3.795–3.846 | 3.571–3.703 | 6.203–6.290 |

同样3 context/三核配置下，速度版比FP16的完成吞吐提高约63.5%。
只量化Attention版本没有速度收益，3-context均值反而低约4.8%，不能包装成加速模型。
速度版的相同固定输入校验和在all核与三个spread worker之间一致；不把校验和当作
120张并发准确率的替代验证。

3-context FPS是聚合吞吐，不是单帧延迟：速度版每个worker平均run_ms约442–460ms，
不是1000/6.25≈160ms的单帧延迟。以上都不是完整YOLO-CLIP系统FPS。
验证报告中的mean_run_ms仅为该轮分类运行数据，不使用其与旧基线的比值替代独立测速。

## 交付状态与下一步

- 没有升级driver、替换系统Runtime、修改原工程或默认模型。
- 三个模型各保存在独立文件，policy、源模型/校准/SDK指纹和模型SHA256保留。
- 原有3个CTest仍通过，所有本轮构建、验证、测速进程已结束。
- 第一种模型可作为绕开异常的精度控制；第二种只证明吞吐潜力，15个百分点损失
  不应默认接受。第三种stem/head保护未体现收益。
- 下一步重点是QKV/MLP大矩阵的细粒度混合精度与激活误差，而不是继续把融合
  Attention指定为FP16，或盲目增加LayerNorm-only QAT步数。
- 板端剩余空间约1.9GB。没有删除实验或用户文件；进一步大模型实验前应先整理
  已有副本，必要时从本机已校验的模型归档恢复。

## 代码和证据

构建：`quantization/build_sdpa_safe_full.py`，复用原SDK step1 IR，按SDK约定不在
step2重新调用config。三个variant：`fp16_except_sdpa`、`sensitive_except_sdpa`、
`sensitive_stem_head_except_sdpa`。

板端工具：`tools/run_sdpa_safe_full_board.py`、`tools/verify_sdpa_safe_inputs.py`。

模型、日志、policy、SHA、模拟器：`quantization/artifacts/sdpa_safe_full_20261010/`。
完整分类指标：`quantization/artifacts/accuracy_sdpa_safe_full_three.json`。
板端原始embeddings：`quantization/artifacts/quant_sdpa_safe_*_ieee/`。
实际算子表、输入位模式核验、原始测速：`results/sdpa_safe_full_20261010/`。
测速汇总：该目录的`speed_summary.json`，保存了两轮每worker run_ms及完成计数。

对应最小复现与失效根因证据见 `SDPA_ROOT_DIAGNOSIS_20261010.md`。
