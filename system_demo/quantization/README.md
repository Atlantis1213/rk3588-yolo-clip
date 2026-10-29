# CLIP 量化实验：独立参考与速度/精度双评估

本目录不是历史模型转换过程的复原。板端未找到原 checkpoint/ONNX，下载的官方
OpenAI ViT-L/14 与历史 RKNN 输出尚不能证明等价；量化候选只与同一新 ONNX 的新 FP16
基线比较。所有 INT8 候选目前均未通过精度验收，不能替换默认模型。

## 目录与运行环境

- 本机：`F:\YOLO-CLIP\system_demo\quantization`
- WSL：`/mnt/f/YOLO-CLIP/system_demo/quantization`
- Python：`/home/atlantis/yolo-clip-quant-venv/bin/python`
- 板端：`/home/orangepi/yolo_clip_bench_systemopt`
- `assets/ViT-L-14.pt`：下载并校验的官方原始权重。
- `assets/dataset-resized.zip`：TrashNet 原始下载，共2527张。
- `artifacts/clip_visual_reference_fp32.onnx`：固定 NCHW、opset17 的图像塔。
- `artifacts/clip_*.rknn`：各实验转换产物。
- `artifacts/logs/`：完整编译日志；未完成日志不代表量化失败。
- `artifacts/accuracy_*.json`：分类/embedding 指标及混淆矩阵。
- `../results/quant_speed_20261010/`：独立轮换速度比较。

## 复现准备与转换（WSL）

```sh
cd /mnt/f/YOLO-CLIP/system_demo/quantization
bash setup_env.sh
source /home/atlantis/yolo-clip-quant-venv/bin/activate
python download_official_clip.py
python export_reference.py --baseline baseline --experimental-reference
python prepare_data.py
python prepare_eval_text.py
python prepare_final_test.py
python run_conversion_logged.py fp16
python run_conversion_logged.py int8-normal
python run_conversion_logged.py int8-auto-hybrid
python equalize_reference.py
python run_conversion_logged.py int8-equalized
```

不要只量化5张演示图。已固定 calibration/validation/final_test 三组，每组120张，
六类各20张，文件级互不重叠；最终测试集不用于选择量化策略。预处理沿用项目直接
resize224，而不是偷偷换成 CLIP 默认 center crop。校准输入为已标准化 NCHW `.npy`，
配置 mean=0/std=1，避免重复标准化。

MMSE 用30张分层校准样本，运行成本较高，需要单独预留转换时间：

```sh
python prepare_mmse_subset.py
python run_conversion_logged.py int8-mmse
```

该命令从头重新转换，不是断点续算。本轮是否完成以 `artifacts/mmse_status.json`
与模型产物/转换 manifest 为准，不能根据部分进度日志声称已评估。

## 板端评估

板端模型命名与本地产物的对应：

| 本地产物 | 板端文件 |
| --- | --- |
| clip_fp16.rknn | clip_reference_fp16_232.rknn |
| clip_int8-normal.rknn | clip_int8_normal_232.rknn |
| clip_int8-auto-hybrid.rknn | clip_auto_hybrid_232.rknn |
| clip_int8-equalized.rknn | clip_equalized_int8_232.rknn |
| clip_fp16-batch3.rknn | clip_reference_batch3_232.rknn |

为每个进程指定实验 Runtime，不替换系统库。例如已部署的最终测试输入：

```sh
cd /home/orangepi/yolo_clip_bench_systemopt
LD_LIBRARY_PATH=./quant_runtime232 ./build/dump_clip_embeddings \
  clip_reference_fp16_232.rknn final_test_png results/quant_final_fp16_ieee host
LD_LIBRARY_PATH=./quant_runtime232 ./build/dump_clip_embeddings \
  clip_int8_normal_232.rknn final_test_png results/quant_final_int8_ieee host
python3 tools/bench_variants_board.py
```

将板端 `embeddings.f32`、`manifest.json` 下载到对应的本地 artifacts 目录，再运行：

```sh
python evaluate_embeddings.py --split final_test \
  --baseline artifacts/quant_final_fp16_ieee \
  --candidate artifacts/quant_final_int8_ieee \
  --output artifacts/accuracy_final_test.json
```

`host` 为 FP16 宿主输入，`float32` 是同一已舍入张量转 FP32 后提交；两者在120张
验证图上的 INT8 输出一致。`native-int8` 仅为输入契约诊断，不是验收或默认路径。
当前默认 DMA pipeline 仍只接受已验证的 native FP16 输入。

## 方法边界

`equalize_reference.py` 是 SmoothQuant/RepQ 风格的通道重参数化实验，非完整论文复现。
LN γ/β 与唯一后继 linear 的输入通道权重反向缩放，在 FP32 检查中保持等价、无新增
推理算子，但并未恢复本次 RKNN INT8 精度。QAT、PTQ4ViT、P4Q 仅为后续路线，未训练。
本环境对 RK3588 的 config 探针只接受 W8A8，拒绝 W8A16/W4A16。

完整测量、采纳/拒绝决定与下一步见 `../docs/OPTIMIZATION_20261010.md`。

## 逐层诊断与手动混合量化续实验

新增的固定反馈循环与输入契约对照见 `../docs/QUANT_DIAGNOSIS_20261010.md`。
`diagnose_reference.py` 对照PyTorch/ONNX/板端FP16与INT8；`analyze_layers.py`按原配置
重建并保存SDK逐层误差；`summarize_layer_analysis.py`生成敏感tensor策略。
`build_manual_hybrid.py`保存SDK临时IR并验证策略tensor名，不将未映射项静默忽略。

```sh
python diagnose_reference.py --backend torch
python diagnose_reference.py --backend onnx
python diagnose_reference.py --backend compare
python analyze_layers.py
python summarize_layer_analysis.py
python build_manual_hybrid.py
```

如果step1已经完成，可在同一实验目录用 `--reuse-ir` 重用它的临时文件；不能拿别的
模型/校准版本的IR混用。步骤是否完成和候选是否验收必须分别查看，不能只看编译成功。
