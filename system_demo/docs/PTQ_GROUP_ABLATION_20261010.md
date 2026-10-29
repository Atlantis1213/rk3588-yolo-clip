# PTQ 精度损失：QKV、输出投影与 MLP 分组对照

后续拆分已完成：只恢复上投影仍59.17%，只恢复下投影65.00%，后者同轮
CLIP-only三context吞吐4.27FPS（FP16为3.83FPS）。尚未满足精度与速度联合目标，
详见 `PTQ_MLP_SPLIT_20261010.md`。

## 结果

继续在正确执行的INT8融合Attention路径上，对原59.17%的混合速度版一次只改变
一个矩阵组的精度。保留同一官方ViT-L/14权重、原120张校准、120张验证及文本bank。
没有改输入尺寸、删block、加入新分类头或使用QAT checkpoint，最终测试集未用于选择。

| 模型/额外FP16组 | 准确率 | 对普通FP16特征余弦均值 | 文件MB（十进制） |
| --- | ---: | ---: | ---: |
| 普通FP16参考 | 74.17% | 1.0000 | 623.05 |
| 原混合速度版 | 59.17% | 0.7809 | 321.15 |
| 恢复24个QKV矩阵 | 59.17% | 0.7777 | 396.64 |
| 恢复24个Attention输出投影 | 60.83% | 0.7751 | 346.37 |
| 恢复48个MLP矩阵 | 69.17% | 0.8695 | 522.45 |

MLP组追回10个百分点（多12张正确）；输出投影只多2张，QKV准确率持平。
这些组对照中，MLP有最大的恢复收益，支持把下一轮重点放在MLP，而不支持
把QKV作为当前首要保护对象。小样本、非线性误差交互及SDK融合变化仍限制因果解释，
不能断言所有误差都来自MLP或QKV永远不敏感。

恢复一组矩阵会同时改变其权重和输入激活计算精度，尚未分离W8和A8分别贡献多少，
也没有区分MLP上投影c_fc与下投影c_proj。所谓“主要”是本轮三组恢复收益排序，
不是完成了所有24层的逐算子独立误差归因。

## 有效性核验

- 三种模型均通过本机/板端SHA一致检查。
- 板端实际算子表核验所有24个融合SDPA仍为INT8 NPU；没有触发FP16 SDPA均匀化。
- 分别核验24、24、48个目标矩阵实际为FLOAT16 NPU，包括SDK融合后的ConvAdd。
- 新候选的120张输入位模式、顺序与普通FP16基线逐文件一致。
- Attention输出投影初次核验被脚本拒绝，因为SDK将其融合为ConvAdd，而旧统计
  只认Conv；修正统计后重新核验再开始验证。失败日志保留，不算模型精度失败。
- 当前已有的LayerNorm、残差、GELU和IO保护维持不变；均复用原step1 IR，未重复校准。

## 激活范围线索（不等于最终归因）

原校准配置中，第12层GELU输出范围约[-0.1636,58.3121]，INT8 scale为0.2293165。
从此前保存的一张验证golden trace观察其1,052,672个激活：

- 约98.78%绝对值小于0.5；约99.76%绝对值小于1。
- 按该SDK scale/zp模拟QDQ，约64.84%被舍入到0。
- 该单样本激活QDQ RMSE约0.0721。

这表明per-layer范围很宽、主体小数值分辨率较粗，是后续检查MLP激活量化的明确线索。
快照只有一张验证样本，不是120张校准的完整分布，也不是对板端内部QDQ的逐位证明。
不能据此盲目裁剪大激活：少量大值可能包含重要语义，需在校准集上量化统计和验证。
GELU算子本身保持FP16，并不意味着输入到INT8 c_proj前的激活不会再次量化。

这也解释了为何“权重per-channel”和“保护GELU/LayerNorm”不等于完成高精度PTQ。
RKNN当前channel配置作用于权重，不能当成文章建议的激活per-channel量化已经实现。

## MLP版独立测速

QKV无精度收益、输出投影只有两张改善，不新增独立测速。MLP版验证后，串行执行
同轮FP16对照：两轮正序/反序、每配置8秒、host固定FP16 tensor、热身/初始化除外。
不使用带perf收集标志的运行时间当速度，不同时执行校准、验证或其它NPU任务。

| 配置 | 同轮FP16 | 恢复MLP的混合版 |
| --- | ---: | ---: |
| 1 context/all核，FPS均值 | 2.148 | 2.351 |
| 3 context/spread，FPS均值 | 3.803 | 3.972 |
| 3 context两轮FPS | 3.791 / 3.814 | 3.954 / 3.991 |

恢复MLP后吞吐接近普通FP16，只比同轮基线高约4.5%，远低于此前混合速度版6.25FPS。
两轮短基准不支持宣称显著稳定加速。以上为CLIP-only聚合吞吐，不是单帧延迟、
完整YOLO-CLIP FPS或长期稳定性结果。分类仍比普通FP16低5个百分点。

## 下一步及交付状态

下一轮应先分别检查MLP c_fc/c_proj及其输入激活量化，定位高收益的少数矩阵/层，
再考虑代表性校准、范围优化、可验证的等价重参数化或匹配实际INT8 Attention后端的QAT。
不以整组MLP恢复FP16作为最终加速方案，也不继续增加LayerNorm-only训练步数替代定位。

默认模型、原工程、driver和系统Runtime未替换。原有3个CTest通过。
板端临时槽每次完成后，在验证状态、当前无运行任务及SHA检查通过时仅删除该槽
`quantization/artifacts/group_ptq_20261010/active/clip.rknn`；三组完整模型本机留存，
可以按SHA重新上传恢复。未删除旧候选、用户文件或任何目录。板端剩余约1.8GB。

## 代码与证据

- 构建：`quantization/build_sdpa_safe_full.py`、`run_group_ptq_builds.py`。
- 3组模型和policy：`quantization/artifacts/sdpa_safe_full_20261010/protect_*_except_sdpa/`。
- 120张指标：`quantization/artifacts/accuracy_group_ptq_20261010.json`。
- 激活线索：`quantization/artifacts/mlp_activation_quant_diagnostic_20261010.json`。
- 实际算子表/输入一致核验：`results/sdpa_safe_full_20261010/`。
- 同轮MLP测速：上述目录`speed_group_mlp/speed_summary.json`。
- 临时槽安全清理：`tools/clear_group_active_slot.py`。

分组对照遵循diagnose的固定输入、单变量、可验证算子精度反馈环；由于项目没有该
技能的管理配置，沿用现有诊断目录，没有新增issue系统或项目管理文件。
