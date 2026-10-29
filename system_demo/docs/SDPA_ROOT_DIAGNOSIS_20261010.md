# 重新排查：混合 FP16 融合注意力的板端执行异常

完整模型后续验收已完成：只让Attention为INT8的版本验证准确率70.83%（FP16为
74.17%），主体INT8混合版59.17%/CLIP三context吞吐6.25FPS；默认模型未替换。
详见 `SDPA_FULL_MODEL_20261010.md`，不把局部0.956余弦当作分类准确率。

## 结论

在当前版本组合（Toolkit2 2.3.2 / 隔离Runtime2.3.2 / RK3588 driver0.9.6）下，
混合量化流程导出、再用`custom_quantize_layers`指定为FP16的`exSDPAttention`，
在板端表现为均匀平均，丢失本应由Q/K决定的query差异。
普通非量化FP16导出的同一计算正常；两条路线的模拟器也都正常。

正确：`softmax(Q Kᵀ / sqrt(64)) V`。

错误分支输出：`≈ mean(V, token_axis)`，同一head的257个query输出逐元素相同。
6组输入对均匀V平均的余弦均值0.9999999799，剩余差异在FP16舍入量级。

这是具体的执行正确性失效，不应先用更多QAT去补偿。
尚未拿到SDK/driver内部源码，不能认定究竟是哪一行、哪个shift/alpha字段或
哪次版本修复导致；严格定位是当前版本组合中的混合FP16融合SDPA路径。
没有直接读取内部Softmax矩阵，因此说的是“输出等效于均匀注意力”，不是已证明
每个内部logit一定为0。

## 决定性对照：不用CLIP权重也复现

独立模型只有Q/K/V三个输入，shape均为`[1,16,257,64]`，不含CLIP权重、图像解码、
归一化、QKV线性层或输出投影。文件约0.45MB。使用真实板端Q/K/V的FP16位模式，
由独立Torch数学运算给出正确参考，逻辑NHWC输入由Runtime转换到native NC1HWC2。
普通FP16结果正确，验证了这个输入提交路径。

| 路线 | 板端→正确Attention余弦均值 | query是否有差异 |
| --- | ---: | --- |
| 普通非量化FP16 | 0.999921130 | 有 |
| 混合流程，全可映射节点FP16 | 0.612027912 | 无，257个输出相同 |
| 同一混合IR，仅取消SDPA的FP16保护 | 0.956091069 | 恢复 |

最后一组算子表明确为`exSDPAttention INT8 NPU`，其它边界保持FP16，必要的dtype
转换由SDK生成；每张图的query最大差异都大于1。触发条件与SDPA的混合FP16保护
分支有关，而不是模型太大、不能多核、一般量化损失或数据集太小。

前两组模拟器对正确计算余弦分别0.9999999438、0.9999999044。模拟器通过并不代表
板端正确。INT8控制仍有真实量化误差，0.956不是完整模型精度恢复，不将其部署。
报告的余弦归约采用FP64；大向量直接用FP32归约存在约1e-4的数值差异，因此以
保存的FP64汇总为准，不将这类微小差异解释成模型收益。

## 排除与缩小范围的证据

- 新进程重放原6张输入，原混合注意力前缀输出与上次逐元素相同，错误确定复现。
- QKV投影、分别导出的Q/K/V，普通FP16与混合FP16板端输出逐元素相同，
  各自对ONNX余弦均大于0.999999。
- 首次显著差异在注意力核心（输出投影、残差之前）：普通FP16对ONNX余弦
  0.999922，混合FP16仅0.612484。独立无权重复现消除了仅裁剪边界才正确的主要疑虑。
- 实际算子表把前两组核心均标为FLOAT16、NPU；最小复现没有大量CPU算子回退，
  CPU主要用于Input/Output边界。该报告不保证所有内部实现细节的dtype。
- 原IR的3072行QKV权重都在原始权重中找到，重排符合head内Q/K/V交错。
- CPU数值对照比较了不同scale、SDK已有scale/zp的QDQ、Q/K交换、head错误重排、
  Softmax轴错误：最佳匹配是均匀平均，并非这些简单错误的近似匹配。
- 在独立IR副本中只把融合scale从0.125改成0、1或0.015625，模拟器明显改变；
  板端仍接近原错误结果。scale0与0.015625逐元素完全相同，scale1最大变化仅
  0.00048828125。IR副本故意改变语义，只用于诊断，不能部署。
- 上轮opt0仍保留融合SDPA，保护内部q_matmul/q_softmax也没有改善。

这些证据针对混合FP16融合SDPA失效。不足以宣布历史未知权重模型、普通全INT8
模型及其它算子都正确；普通INT8原有精度不合格仍要单独验收。

## 前期判断更正与后续顺序

此前主要归因于“量化敏感性、需要QAT恢复”不够准确：这里先有一个部署正确性问题。
训练代理实现了正确Softmax，与错误板端分支不同；训练端改善不能保证板端改善。

本轮按 diagnose 的最小复现与差分方法形成确定性判据，仅诊断，没有覆盖默认模型，
没有升级driver、替换系统Runtime或修改governor。所有训练与诊断进程已经结束。

建议下一步：保留正确普通FP16基线，停止沿用对融合SDPA直接指定FP16的当前策略；
在隔离副本验证不走该融合分支的语义等价导出，或比较明确匹配的SDK/Runtime/driver。
未经批准不升级系统driver。正确性恢复后，再做INT8/混合/QAT的完整速度-精度验收。

## 证据位置（以system_demo/quantization为根）

- `artifacts/recheck_attn_hybrid_20261010/`：重新重放。
- `artifacts/prefix_{qkv,q,k,v,core}_control_20261010/`：逐边界模型与对照。
- `artifacts/core_perf_{plain,hybrid}_20261010/`：算子dtype/target报告。
- `artifacts/core_weight_order_audit_20261010.json`：权重行核验。
- `artifacts/core_numerical_hypotheses_20261010.json`：数值假设。
- `artifacts/sdpa_ir_scale_comparison_20261010.json`：因果scale探针。
- `artifacts/standalone_sdpa_20261010/`：约0.45MB纯Attention复现。
- `artifacts/standalone_sdpa_keep_int8_20261010/`：仅改变SDPA精度保护的控制。

入口：`run_attention_boundaries.py`、`run_standalone_sdpa.py`、
`probe_sdpa_keep_int8.py`、`explain_core_numerics.py`、`probe_sdpa_scale_ir.py`。
板端：`tools/dump_prefix_outputs.cpp`、`tools/dump_standalone_sdpa.cpp`。
`assert_sdpa_diagnosis.py`核验捕获的失效与INT8控制，不声称生产问题已修复。
