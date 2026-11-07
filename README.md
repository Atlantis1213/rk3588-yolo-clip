# RK3588 YOLO-CLIP 多模态垃圾识别与端侧推理优化

面向 RK3588 / Orange Pi 的 C++17 视觉推理项目，包含 USB 摄像头采集、DMA-BUF
共享缓冲、RGA 图像预处理、RKNN 推理、YOLO/CLIP 分阶段线程池、NPU 核资源协调，
以及 CLIP 量化和部署数值一致性诊断工具。

主工程位于 [`system_demo/`](system_demo/README.md)。根目录保留早期摄像头实现与测试说明。

初次阅读建议从[中文代码阅读路线](system_demo/docs/CODE_READING_GUIDE.md)开始，
再结合源码中的中文注释理解数据流、线程协作、资源生命周期与量化实验。

## 数据路径与实现边界

```text
USB camera / V4L2 raw YUYV or NV12
  -> exported DMA-BUF FD
  -> RGA color conversion / resize / letterbox
  -> CPU normalization directly into FP16 DMA tensor
  -> RKNN YOLO -> ROI -> RGA preprocessing
  -> FP16 DMA tensor pool -> RKNN CLIP image encoder
  -> similarity matching with offline text embeddings
```

- 当前 raw 摄像头路径不需要 FFmpeg/MPP；压缩视频的 MPP 解码接入不属于已实现功能。
- DMA 输入绑定避免额外输入张量复制，但归一化仍由 CPU 完成；不宣称全链路完全零拷贝。
- 图像编码器为 OpenAI CLIP ViT-L/14。文本侧使用同一 `openai/clip-vit-large-patch14`
  的文本编码器离线生成特征，板端不逐帧运行文本模型。
- NPU 核资源协调是应用层请求准入，不是修改内核调度器或实现 NPU 抢占。

## 目录

| 目录 | 内容 |
| --- | --- |
| `system_demo/src/` | 程序入口与流水线编排 |
| `system_demo/include/` | V4L2、DMA、RGA、RKNN、缓冲池、队列与调度模块 |
| `system_demo/tests/` | 归一化、资源生命周期与并发测试 |
| `system_demo/tools/` | 摄像头对照、板端探针、性能测试与分类测试脚本 |
| `system_demo/quantization/` | 模型转换、PTQ/QAT 实验和 Attention 最小复现工具 |
| `system_demo/docs/` | 测试协议、模型来源、测量结果与已知限制 |

## 编译与测试

在配置好 OpenCV、RGA 和 RKNN 开发依赖的 RK3588 Linux 上：

```sh
cmake -S system_demo -B system_demo/build -DCMAKE_BUILD_TYPE=Release
cmake --build system_demo/build -j2
ctest --test-dir system_demo/build --output-on-failure
```

不依赖板端硬件库的 Linux 可移植测试：

```sh
cmake -S system_demo -B system_demo/build-portable \
  -DBUILD_HARDWARE_DEMO=OFF -DCMAKE_BUILD_TYPE=Debug
cmake --build system_demo/build-portable -j2
ctest --test-dir system_demo/build-portable --output-on-failure
```

完整参数与运行示例见主工程 README。部分实验脚本保留历史机器路径和地址，使用前
需按实际环境配置；仓库不包含 SSH 私钥、密码或 GitHub 令牌。

## 结果阅读口径

- 摄像头流水线、CLIP-only 聚合吞吐、单请求耗时是不同测量口径，不能互相替代。
- 在固定裁剪的 200 张真实图片上，研究 FP16 为 87.5%，KL/RepQ 混合精度为 82.5%。
  这是绕过 YOLO 的裁剪后分类结果，不是在线端到端准确率。
- 默认 YOLO 的检测置信度问题、Attention 混合精度异常的版本边界，以及未采用的量化
  候选均保留在实验文档中，不将实验候选描述为默认上线模型。

建议从以下文档阅读：

- [摄像头 DMA 说明](CAMERA_DMA.md)
- [系统优化记录](system_demo/docs/OPTIMIZATION_20261010.md)
- [模型来源与输出约定](system_demo/docs/MODEL_PROVENANCE.md)
- [真实数据测试协议](system_demo/docs/REAL_DATA_EVALUATION_20261011.md)
- [绕过 YOLO 的 CLIP 裁剪分类测试](system_demo/docs/CLIP_ONLY_CROP_EVAL_20261011.md)
- [融合 Attention 最小复现与诊断](system_demo/docs/SDPA_ROOT_DIAGNOSIS_20261010.md)

## 仓库内容与本地资产

本仓库保存源码、测试脚本、文本标签/离线文本特征以及实验说明。模型权重、RKNN/ONNX
文件、图片数据集、量化中间产物、SDK 运行库、虚拟环境和完整原始测试输出留在本地，
由 `.gitignore` 排除，不随 Git 上传。

因此，仅克隆仓库不能直接重跑所有历史实验；需准备文档中对应的模型、数据和板端依赖。
