# CLIP 量化精度：逐层诊断与手动混合量化（2026-10-10）

重排查更新：混合FP16融合SDPA在板端输出等效均匀V平均，无权重独立模型复现；
普通FP16正常，取消SDPA的FP16保护后恢复query差异。前期“主要是量化敏感性”的
解释不完整，详见 `SDPA_ROOT_DIAGNOSIS_20261010.md`，不将候选替换为默认模型。

本轮继续使用独立官方 ViT-L/14 参考，不替换历史部署模型。采取固定反馈循环：
先对齐 PyTorch/ONNX/板端 FP16，再隔离输入与输出契约，最后分析网络内的量化误差。
前轮速度、公开分类准确率和系统调度见 `OPTIMIZATION_20261010.md`。

## 1. 固定样本对照：排除错误的 FP32/FP16 基线

固定验证集每类第一个样本，共6张；没有用最终测试集挑策略。
读取同一套板端实际输入 FP16 bit，转为 NCHW FP32给 PyTorch/ONNX。

| 对照 | 6张样本cosine范围 | 判断 |
| --- | ---: | --- |
| PyTorch → ONNX FP32 | 约1.0（误差约1e-11量级） | 导出等价检查通过 |
| PyTorch → 板端FP16 | 0.99984–0.99998 | 这组样本的FP16基线正常 |
| PyTorch → 板端普通INT8 | 0.18768–0.46113 | 大幅偏差可重复 |

脚本：`quantization/diagnose_reference.py`。
证据：`quantization/artifacts/diagnosis_20261010/reference_comparison.json`。
六样本不是全数据集等价证明，完整分类指标仍以前轮120张验证/测试为准。

## 2. 隔离 C API 输入和输出契约

`tools/probe_quant_io.cpp` 对同一个模型、同一输入分别测试：

- FP16宿主输入（SDK转换）。
- 同值FP32宿主输入。
- 手工按 `q=clip(round(x/scale)+zp)` 的INT8宿主输入。
- 同一手工INT8输入的原生pass-through提交。

每种模式同时核对 SDK float output 与原始INT8输出手工反量化。
输入scale=0.01544500608、zp=-12；输出scale=0.08204193413、zp=-5。

结果：FP16/F32宿主输出逐位一致，重复输出与前轮保存的结果逐位一致；
原始输出手工反量化与SDK float output最大绝对差≤2.24e-7。
手工INT8宿主与原生pass-through输出也逐位一致。
因此，本次大偏差不是单由FP16提交类型或输出scale/zp反量化错误造成。

三张样本的手工INT8与SDK浮点提交不同（cosine约0.425–0.514），另外三张相同。
这与前轮native DMA诊断的模式对应，但不能因此声称DMA路径已经验收。
当前数据说明差异并非DMA绑定所独有；舍入方法或量化内核的数值敏感性仍需区分。
默认pipeline继续拒绝未验收的native INT8输入。

证据：`diagnosis_20261010/io_contract_comparison.json`与对应probe manifest/embeddings。

## 3. 只量化输入的负对照

`tools/probe_quant_input.cpp` 用普通INT8模型的scale/zp将输入量化再反量化，
随后送入原完整FP16网络；没有把网络内的权重/激活改为INT8。

输入重建RMSE约0.00445–0.00497；输出embedding cosine为0.98797–0.99879，
远好于完整INT8的0.18768–0.46113。输入量化有影响，但不足以单独解释这组样本的崩溃。
主要排查方向因此收缩到网络内部量化，不能简单更换DMA输入格式后称已修复精度。

证据：`diagnosis_20261010/input_qdq_control.json`。

## 4. 完成 SDK 逐层分析：归一化、残差和注意力路径敏感

官方SDK不能直接对load_rknn后的模型进行逐层分析，因为缺少原始图信息。
因此按相同ONNX、同120张校准、channel-wise W8A8/normal/optimization3重建诊断模型，
再执行 `accuracy_analysis(target=None)`。首张固定验证图生成373条误差记录。
SDK golden与PyTorch的最终输出cosine约1.0，验证了这份浮点golden。

single：给当前层正确的golden输入，观察当前层自身及量化边界的误差。
entire：从量化网络起点累积到当前层的误差。不能把低entire值直接归咎于该层本身。

| 位置 | single cosine | entire cosine | 解释 |
| --- | ---: | ---: | --- |
| ln_pre | 0.97775 | 0.97509 | 第一个single<0.99的记录 |
| block1 / ln1 | 0.98924 | 0.88868 | 早期累积误差已明显放大 |
| block13 / ln1 | 0.53383 | 0.35694 | 局部归一化/量化边界也严重失真 |
| block14 / ln1 | 0.52111 | 0.24921 | 不仅是前层累计偏差 |
| block22 / attention | 0.38659 | 0.00832 | 注意力路径局部误差很大 |
| block23 / 第二次残差Add | 0.68139 | 0.09346 | 残差精度也是敏感点 |
| 最终projection | 0.99943 | 0.46278 | 本层基本准确，但输入特征已失真 |

这些是一个样本的simulator定位证据，不代表所有图像都由相同单一算子导致。
没有证据证明大量CPU fallback；这里的结论是量化数值误差，不是执行设备归属。

### 模拟器与板端还存在差异

同一新导出的诊断RKNN也已传到板端复测。虽然二次构建的RKNN SHA不同，
六个板端宿主输出与原普通INT8逐位一致，输入/输出scale、zp相同。
同一诊断模型的simulator与板端输出cosine仍只有0.36465–0.52105，
而simulator本身对PyTorch也已严重偏离（0.28228–0.46278）。

因此，量化网络在模拟器中就已失败，Runtime不可能是唯一原因；但模拟器/Runtime
差异同样未被排除，不能声称二者完全一致或确定没有硬件内核问题。
后续混合量化需要同时看simulator和板端结果，最终以板端分类与速度验收。

脚本：`quantization/analyze_layers.py`、`summarize_layer_analysis.py`。
证据：`diagnosis_20261010/layer_summary.json`；完整SDK表和映射位于
`layer_analysis/snapshot/error_analysis.txt`及`map_name_to_file.txt`。
完整编译/分析日志：`layer_analysis/analysis.log`，未替换系统Runtime。

## 5. 手动混合量化候选：构建与验收分开

根据实际误差表生成两套可追溯策略：

- 归一化+残差：99个tensor。
- 归一化+残差+注意力：123个tensor，本轮优先测试这一套。

将这些敏感tensor设为float16，不主动把大矩阵Conv/MatMul权重全部切回FP16。
SDK可能扩大相邻的浮点边界，因此“文件大小/推理速度不受影响”不是已有结论。
没有改网络深度、图像尺寸或YOLO→ROI→CLIP的处理语义。

`build_manual_hybrid.py`保存SDK step1的可编辑model/data/quantization.cfg，
将误差表中的策略名逐一核对，禁止默默丢弃无法匹配的敏感tensor。
后续可以复用这套临时IR，不必每次都重新做120张校准。
候选状态及完整日志保存在 `quantization/artifacts/manual_hybrid_20261010`。

当前保护策略由固定验证trace产生；只有完成板端速度、120张验证分类和embedding
比较后，才能判断是否改善。不得把“设置了float16”当作精度已恢复。

### 本轮候选已完成板端验收：拒绝采用

123个tensor均在SDK配置中找到对应项，模型成功导出，大小320,805,658 bytes；
原普通INT8为319,908,186 bytes。模型构建成功不等于实际质量达标。
数据校验记录位于 `manual_hybrid_20261010/ir_provenance.json`，包括ONNX及120个校准
输入的SHA256；重新使用IR时脚本检查版本、源模型、校准文件及配置是否一致。

| 120张验证集 | 新FP16基线 | 手动混合候选 |
| --- | ---: | ---: |
| accuracy | 74.17% | 17.50% |
| macro-F1 | 0.6991 | 0.1189 |
| 对FP16 embedding cosine均值 | reference | 0.4624 |
| mean run_ms（本轮分类计时） | 415.41* | 273.40 |

*FP16分类计时来自保存的同输入基线，独立轮换测速另做新对照，不仅引用旧计时。
比普通INT8的平均embedding cosine0.2972有所提高，但分类没有恢复，不可采用。
没有用最终测试集重新调此策略，也没有将不合格候选部署为默认模型。

独立固定tensor、host FP16提交、同Runtime2.3.2、两轮正序/反序比较：

| 配置 | 第一轮FPS | 第二轮FPS | 范围与限制 |
| --- | ---: | ---: | --- |
| 新FP16 / 1context all | 2.1472 | 2.1818 | 独立对照 |
| 手动混合 / 1context all | 3.2001 | 3.1977 | 约1.48倍完成吞吐 |
| 手动混合 / 3context spread | 7.3446 | 7.2663 | CLIP-only，不是完整YOLO-CLIP FPS |

三context的单worker平均run_ms仍约387ms，7.3 FPS对应聚合完成间隔约137ms，
不能当作单帧延迟137ms。更重要的是当前精度不合格，不能包装成已实现可用7 FPS系统。
此数据只证明该量化候选有吞吐潜力，不证明精度、零拷贝INT8输入或完整流水线验收。

证据：`quantization/artifacts/accuracy_manual_sensitive.json`、
`results/manual_hybrid_20261010/speed_summary.json`及各原始报告。
测速前先完成120张验证，二者不并行执行，避免NPU负载污染。

## 6. 新文章的参考价值与不能照搬的部分

已阅读用户提供的[CLIP INT8量化文章](https://blog.csdn.net/gitblog_00068/article/details/151598138)。
可借鉴的是分层保护、代表性校准、速度/精度双评估，以及后续QAT/蒸馏方向。
本轮试验说明：保护敏感层是可检验假设，不是精度恢复保证。

代码核对发现该文的示例混用了不同CLIP骨干：其ViT示例套用了ResNet的BN/多卷积
融合结构，还将visual子模块当成完整CLIP对象调用。官方VisionTransformer没有这些BN
和三层卷积stem。所称动态量化的示例实际上使用静态prepare/校准/convert路径；
而且prepare的第二个位置参数是inplace，QConfig需要先绑定到相应module。
这些已分别核对OpenAI官方固定commit源码及PyTorch v2.0.1源码，不执行文章中的安装/训练代码。

在本项目中，浮点保护使用当前RKNN支持的FP16；CPU的fbgemm、ONNX Runtime动态
量化和文本INT4不等于RK3588 NPU加速。文本特征在项目中预计算，量化文本网络并不
减少在线图像推理耗时；本环境也已拒绝RK3588的W4A16配置。
权重dequantize不等于恢复完整浮点模块/撤销量化内核，更不可能恢复已丢失的信息。

该文的性能表未明确给出与本项目相符的硬件、ViT-L/14、数据集和测量协议，不能
移用其倍率或精度下降数字。本轮自己的板端数字及验收失败必须保留。

下一步优先做全浮点混合配置的控制实验，确认SDK保护范围与重数量化边界，再检查
CLS token、残差与注意力的局部数据；需要时在同架构上做fake-quant/特征余弦蒸馏。
不能仅把模块dtype设回float，或继续增加名单长度后宣称精度已修复。

主要核对来源：

- [OpenAI CLIP官方VisionTransformer](https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/clip/model.py)
- [PyTorch v2.0.1量化接口源码](https://github.com/pytorch/pytorch/blob/v2.0.1/torch/ao/quantization/quantize.py)
- RKNN Toolkit2 API/用户手册及本地 `platform_precision_support.json`。

## 2026-10-10 下午追加

新增前缀差分：ln_pre与block0 LN1的两条导出路径板端一致，但到block0注意力+残差
已复现混合全浮点板端失真、模拟器正确，范围收敛到首个注意力相关路径，未定位唯一算子。
新增三组QAT均完成，投影矩阵微调提高特征余弦到0.85994，但分类仍未超过64.17%。
详见 `QAT_FOLLOWUP_20261010.md`，所有原模型保持不变。

全浮点含IO保护仍未修复板端精度（120张准确率17.50%）；GELU混合含IO19.17%。
单核/三核、普通FP16输入/native透传的固定样本差分已经完成，不能将此问题
归因为核分配；不同Runtime子进程输出也一致。真实4060 QAT首轮完成，但分类
未改善，模型未替换。详见 `QAT_PILOT_20261010.md` 与新布局/核/Runtime原始记录。

## 复现命令（WSL，原列表）

```sh
cd /mnt/f/YOLO-CLIP/system_demo/quantization
source /home/atlantis/yolo-clip-quant-venv/bin/activate
python diagnose_reference.py --backend torch
python diagnose_reference.py --backend onnx
python diagnose_reference.py --backend compare
python analyze_layers.py
python summarize_layer_analysis.py
python build_manual_hybrid.py
```

板端IO探针的模型使用隔离Runtime2.3.2；原默认Runtime2.3.0/历史模型不动。
源码位置为 `tools/probe_quant_io.cpp` 和 `tools/probe_quant_input.cpp`。
SDK依据：API手册第22页，用户手册第30–31、123–125、133–135页。
