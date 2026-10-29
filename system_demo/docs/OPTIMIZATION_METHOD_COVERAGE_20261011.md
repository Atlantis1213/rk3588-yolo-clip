# 系统优化与量化方法覆盖清单

目标仍是：对已收集、与 RK3588 和当前约束兼容的常见方法逐项试验，直到出现有证据的
明显瓶颈。当前尚未达到这一终点。读过论文、准备了脚本、浮点验证或训练损失改善均不算
完成板端验收；没有把整体目标缩成 MMSE 或一组重参数化试验。

## 边界与验收

- 24-block ViT-L/14、224 输入、303,966,208 个逻辑参数，YOLO→ROI→CLIP 语义保持。
- 允许量化精度改变、数学等价参数折叠、既有参数的训练/重建；不增减网络层或文本分类头。
- 原工程、默认模型、系统 Runtime 2.3.0、driver 0.9.6 保留。隔离 Runtime 2.3.2 仅用于实验。
- 真实融合混合 FP16 SDPA 的均匀化异常已最小复现，后续混合版的 24 个 SDPA 保持 INT8 NPU。
- 仅用新开发 144 张挑策略，锁定 144 张在冻结后才使用；新文件与原 360 张来源互斥。
- 指标为同一官方参考权重的 TrashNet 六类零样本分类，不是历史四类+unknown 场景。

上两条是早期研究协议的历史记录。用户之后指定真实200张YOLO裁剪/empty绕过检测、
以及标准数值保真作为两种验收标准；当前推进按`REAL_DATA_EVALUATION_20261011.md`
执行，不再依据旧144张分数挑选部署策略。冻结200张不参与训练、校准或调阈值；
已见测试集上的候选比较不是新的独立盲测。
- 三 context FPS 是 CLIP-only 聚合吞吐，不是单图延迟或完整 YOLO/RGA/DMA 流水线速度。
- 只做 Demo 级两轮短基准，初始化/热身排除、正序/反序、无其它并行 NPU 工作；不追加长期稳定性。

## 已试与进行中

| 方法 | 证据/状态 | 当前判断 |
| --- | --- | --- |
| 多 context 分核、共享权重 | `tools/bench_clip_concurrency.cpp`、历次同轮 speed | FP16 约3.8、KL混合约6.3 CLIP FPS；非理想线性3倍 |
| 逐核准入、YOLO adaptive | `OPTIMIZATION_20261010.md`、`results/npu_adaptive_ieee_20261010` | 有尾延迟收益，未见吞吐提升 |
| batch3 | 同报告、batch基准 | 没有优于分核 context 的吞吐收益 |
| 宿主/DMA输入、归一化位模式 | `OPTIMIZATION_20261010.md`及已有输入对照 | 正确性与速度分别验证，不能把DMA摄像头性能算成CLIP模型性能 |
| MinMax/normal，30张 | `method_sweep_20261011/normal30`，144张板端+同轮speed完成 | 85/144=59.03%，未采用 |
| KL，30张 | `method_sweep_20261011/kl30`，144张板端+同轮speed完成 | 91/144=63.19%，未采用；需与MMSE30而非KL120作算法对照 |
| MMSE，30张 | 同目录`mmse30`，约3.09h构建，144张板端+同轮speed完成 | 89/144=61.81%，约6.42 FPS，未采用 |
| KL，120张 | `sensitive_kl_except_sdpa`，新144张基线完成 | 95/144=65.97%，约6.3 FPS，仍为量化参考 |
| 分组矩阵FP16保护 | `PTQ_GROUP_ABLATION_20261010.md`、`PTQ_MLP_SPLIT_20261010.md` | 已试QKV、输出投影、MLP上下投影组，不等于任意逐层最优搜索 |
| 分段下投影FP16 | `PTQ_KL_PARTIAL_MLP_20261011.md` | 中段验证收益未通过独立划分复测，未采用 |
| RepQ式LN2仿射折叠 | `affine_sources/repq_ln2`、`affine_rknn/repq_ln2`、新144张板端完成 | 94/144=65.28%，平均特征余弦0.8362高于KL的0.8162，但分类少1张，暂不采用；不是完整RepQ-ViT复现 |
| SmoothQuant式LN2 α=.25/.50/.75 | `run_method_sweep_builds.py`串行队列 | α=.25板端93/144=64.58%，暂不采用；α=.50构建中、α=.75待跑 |
| LN1+LN2的RepQ/SmoothQuant | 同串行队列 | 待构建/验收；仍不实现论文log Softmax |
| LN-only及LN+原投影QAT | `QAT_FOLLOWUP_20261010.md`、4060四组300步 | 代理特征改善、分类未改善；代理不匹配实际融合内核，未部署 |
| AdaRound式MLP下投影重建 | `reconstruct_rounding.py`，4060逐层500步，SDK KL120逐通道scales | 24层CUDA训练完成，用120张校准、每图16token；平均层MSE降约5.76%，尚无完整RKNN结果 |
| 偏差补偿 | 与AdaRound同时计算nearest和learned舍入的bias补偿 | AdaRound+补偿的平均层MSE降约9.90%；固定网格和全部SHA已验证，三个单变量完整模型已排队 |
| W8A16/W4A16 | `artifacts/platform_precision_support.json` | 当前SDK的RK3588目标配置实际拒绝，不套用其它芯片结果 |
| 编译/注意力边界控制 | `SDPA_ROOT_DIAGNOSIS_20261010.md` | opt0仍融合；CPU目标key混合阶段拒绝；不是SDK/driver已经修复 |

最新状态补充（2026-10-11）：6个仿射候选全部转换完成，真实ROI板端评估已启动。
三个重建候选修复vendor IR校验后均完成SDK step2、导出和模拟器检查，新目录
`reconstruction_rknn/irfix_20261011`；待同一真实ROI板端验收，不能仅据校准层MSE采用。
历史CLIP在Runtime230/232的200个精确输入上输出逐值相同；原项目FP32和OpenAI FP32
在显式单位范数契约下方向几乎完全相同，原始幅度差不能当作量化损失。

## 未完成的方法，不能写成“方法已经耗尽”

1. 扩充为 240 张校准的 KL /代表性与范围统计对照。
2. 使用校准集的 percentile/MSE 范围搜索，之后核验实际部署是否使用这些参数。
3. Hessian/empirical-Fisher 加权范围及层重建，对比普通 MSE；避免用开发/测试激活拟合。
4. 将 AdaRound/bias 结果用同一量化参数网格导出，核验SDK重舍入没有破坏训练出的网格。
5. BRECQ式 block 联合重建、层间蒸馏，以及更贴近正确INT8 SDPA的QAT代理；现有单矩阵
   500步 pilot 不等于完整 BRECQ，也不证明训练类路线耗尽。
6. 正确SDPA路线上的逐层auto-hybrid/敏感性与速度预算搜索。
7. PTQ4ViT twin quant /FQ-ViT PTF+LIS /RepQ log量化的后端兼容性小模型探针。
   算法存在不意味着RKNN原生支持，未做探针前不标“不支持”或“已经试过”。
8. 实际可用的编译器融合/attention参数优化；CUDA FlashAttention不能直接加速RK3588 NPU。
9. 同一正确模型下的CPU亲和性、频率/功耗与host/DMA I/O消融，按当前证据补缺而非重复长期烧机。

## 对照协议与自动运行

构建端只运行一条CPU队列，避免8 GB WSL里同时转换多个完整模型；MMSE已完成不重启。
板端协调器单独工作，只保留一个候选临时模型，基线KL120独立保留，精度测试和测速串行。
候选退役前核验本机完整模型SHA、取回完成的板端证据；仅删除本轮的临时模型副本，
不删除默认模型、原模型或结果。MMSE30/normal30/kl30/repq_ln2/smooth_ln2_a25临时副本已退役，本机完整模型保留。

入口：`quantization/run_method_sweep_builds.py`、`tools/coordinate_method_sweep.py`、
`tools/run_method_sweep_board.py`、`quantization/evaluate_protocol_v2.py`。
独立状态位于`quantization/artifacts/method_sweep_20261011/cpu_queue_20261011/status.json`
及`results/method_sweep_20261011/coordinator_20261011/status.json`。

下一条串行任务为仅bias、仅AdaRound、AdaRound+bias三个单变量候选：
`quantization/run_reconstruction_builds.py`等待同一CPU互斥锁，
`tools/coordinate_reconstruction_after_sweep.py`等待前一条板端队列和转换队列结束后验收。
固定24,576个SDK weight channel scales；训练输出的最大整数网格误差1.53e-5，
但仍需验证SDK重新量化后的真实NPU输出，不能只靠浮点权重落在网格上便宣称成功。

4060的普通子进程随SSH会话结束而退出，已通过无活动进程/空日志发现并改用独立WMI
supervisor。首次失败没有进入训练或创建run；新进程已验证完成校准缓存并进入逐层训练。
远端状态：`D:/YOLO-CLIP-QAT-20261010/runs/adaround_down_kl120_20261011/status.json`。

## 已读参考，不套用论文精度数字

- [PTQ4ViT](https://arxiv.org/abs/2111.12293)、[官方代码](https://github.com/hahnyuan/PTQ4ViT)
- [RepQ-ViT](https://arxiv.org/abs/2212.08254)、[官方代码](https://github.com/zkkli/RepQ-ViT)
- [SmoothQuant](https://arxiv.org/abs/2211.10438)、[官方代码](https://github.com/mit-han-lab/smoothquant)
- [FQ-ViT](https://arxiv.org/abs/2111.13824)、[官方代码](https://github.com/megvii-research/FQ-ViT)
- [AdaRound](https://arxiv.org/abs/2004.10568)、[BRECQ](https://arxiv.org/abs/2102.05426)、
  [BRECQ官方实现](https://github.com/yhhhli/BRECQ)
- [RepQuant](https://arxiv.org/abs/2402.05628)、[P4Q](https://arxiv.org/abs/2409.17634)

本轮实际读取了RepQ的`quant_model.py`、`quant_modules.py`、`quantizer.py`、`test_quant.py`
和BRECQ的`adaptive_rounding.py`、`layer_recon.py`，区分论文完整算法与RKNN约束下的改编。
