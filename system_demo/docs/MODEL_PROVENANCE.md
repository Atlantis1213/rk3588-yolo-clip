# Model inventory and conversion provenance

The actual deployed artifacts were inspected through RKNN and SHA-256 on 2026-10-09.
The system optimization does not change their bytes. Readable raw inventory is stored in
`results/debug_profiles/model_inventory.json` and can be regenerated with `model_inventory`.

| Property | YOLO | CLIP image encoder |
| --- | --- | --- |
| Filename | yolo_waste_single_class.rknn | clip_image_encoder_vitl14.rknn |
| Size | 7,632,181 bytes | 624,326,207 bytes |
| Logical/native input | FP16 NHWC [1,640,640,3] | FP16 NHWC [1,224,224,3] |
| Input width stride | 640 | 224 |
| Input bytes including stride | 2,457,600 | 301,056 |
| Logical output shape | [1,5,8400] | [1,768] |
| Logical output type | FP16; API requests float output | FP16; API requests float output |

Runtime: `2.3.0 (c949ad889d@2024-11-07T11:35:33)`; driver: `0.9.6`.

```text
YOLO SHA256: e7120761a54e899c635bc67f2c30da593acad102b81581e41fb9277964998f12
CLIP SHA256: 898069fbe1bf71a760f551f0b6bc36c889a5dbedcb15d427070b8923ba1be177
```

SDK tensor metadata reported quantization enum 2 (AFFINE_ASYMMETRIC) for these queried
boundaries. Preserve this recorded fact; do not infer all weights/operator precisions from
FP16 boundary tensors or claim a particular historical INT8 conversion procedure without
the original conversion configuration and graph evidence.

## Known preprocessing contract

YOLO: BGR image placed at the top-left of a square padded to max(H,W), padding value 114,
resize to 640x640, RGB order, pixel/255, NHWC FP16. The code selects the highest-confidence
single-class candidate and maps its box back to original coordinates.

CLIP: ROI resize to 224x224, RGB order, `(pixel/255 - mean)/std`, where
mean=[0.48145466,0.4578275,0.40821073] and std=[0.26862954,0.26130258,0.27577711].
Output embedding dimension is 768. Text embeddings are loaded from the existing .f32 bank
and normalized for cosine matching. Current scalar/LUT implement corrected IEEE binary16;
legacy scalar/LUT retain the old conversion bits. Normalization formulas are unchanged,
but the corrected FP16 rounding can change the actual tensor and resulting predictions.

## Historical records still missing

Original checkpoint/ONNX files, YOLO training version, training dataset/metrics, exporter
command/opset, RKNN Toolkit conversion version/configuration, calibration set and operator
precision policy are not available in this workspace. A .rknn artifact and Runtime version
do not reproduce these steps. Do not conflate Toolkit conversion version with Runtime version.

When the originals become available, archive their hashes and actual commands, compare
ONNX and RKNN outputs on fixed fixtures, record conversion warnings and any precision changes.
This project currently demonstrates deployment/system optimization, not verified training
or independently reproduced historical model conversion.

## Separate official-reference experiment (2026-10-10)

A board-side search did not recover the historical CLIP checkpoint/ONNX/conversion scripts.
The user authorized downloading replacements. OpenAI's official ViT-L/14 checkpoint was
downloaded separately to `quantization/assets/ViT-L-14.pt` and its official SHA verified:

```text
b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836
```

Source: OpenAI CLIP `_MODELS` Azure URL, recorded by `quantization/download_official_clip.py`.
OpenAI CLIP code is pinned at `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`.
The visual tower is 224 pixels, patch14, width1024, 24 blocks, 16 heads, embedding768.
The exported FP32 ONNX uses opset17, fixed NCHW [1,3,224,224], 1,216,291,415 bytes.

Against the historical RKNN on five identical input tensors, embedding cosine was
0.9655–0.9904, insufficient to establish same-weight/same-export equivalence. The exporter
requires an explicit `--experimental-reference` override. This is a new reference;
all quantized candidates compare against a new FP16 conversion from that same ONNX.
It does not replace the historical deployed model or text feature bank.

Conversion environment: independent WSL venv, Toolkit2.3.2, PyTorch2.2 CPU, ONNX1.16.1.
Board measurements use Runtime2.3.2 through child-process `LD_LIBRARY_PATH=./quant_runtime232`;
the installed Runtime2.3.0, driver0.9.6 and system governor are not replaced/modified.
Mean=0/std=1 in RKNN configuration because calibration tensors are already normalized.
Calibration NPY uses NCHW float32 values rounded through IEEE FP16 first, matching the
application's inputs. The SDK rejected the NHWC calibration negative control explicitly.

TrashNet source and checksums are recorded in `quantization/artifacts/dataset_provenance.json`.
Calibration/validation/final-test each contain 20 samples per class (120 per split),
with file-disjoint membership and seed20261009. The six public classes are not the original
four-category-plus-unknown deployment task. Text features use the same official checkpoint,
fixed prompt templates and class-bank normalization across every candidate.

Conversion manifests/logs, reference comparison, embedding manifests and accuracy reports
are retained in `quantization/artifacts`. See `OPTIMIZATION_20261010.md` and
`../quantization/README.md` for acceptance decisions and commands.

## Original assets located on 2026-10-11

The user identified `G:/OrangePi/Yolo_clip` on the 4060 host. Its YOLO and ViT-L/14 RKNN
hashes exactly match the historical board artifacts above. The directory also contains
the original `clip_image_encoder_vitl14.onnx` (SHA256
`a51e3e827dcdcd733fdc20a94726a97828a1dd18521b6034ce9d07b58627fa6b`) and the actual
YOLO best.pt/best.onnx plus training records. Thus the earlier "missing source" statement
describes the earlier search, not the current state. The YOLO custom run was 3 epochs.

The original source is evaluated on the remote machine using byte-identical frozen
200-image inputs; numeric equivalence/quality is established by the new reports, not
by filenames alone. The source recovery does not retroactively make downloaded-reference
INT8/QAT experiments equivalent to the historical model. See `REAL_DATA_EVALUATION_20261011.md`.

## Output-contract correction and Runtime control (2026-10-11)

On all 200 byte-identical frozen primary inputs, the recovered project FP32 ONNX
outputs unit embeddings (mean norm 0.999999996). The downloaded OpenAI FP32 tower
outputs raw embeddings (mean norm 18.6277321). After explicitly normalizing both,
mean cosine is 0.9999999999986545, normalized NRMSE 1.6404e-6 and SQNR 115.70 dB.
Thus the prior large cross-source raw RMSE/SQNR reflects different output contracts,
NOT evidence of quantization damage or incompatible floating visual directions.
This is numerical equivalence on the measured inputs, not a proof of byte-identical weights.

Historical RKNN has mean cosine 0.9732236653 to the recovered floating source,
NRMSE 0.23141484 and SQNR 12.7122 dB. On the same 200 frozen inputs, isolated
Runtime 2.3.2 versus installed Runtime 2.3.0 produces byte-identical FP32 embedding
values (RMSE/max error 0). Runtime replacement therefore does not resolve this
historical directional discrepancy on this test. The historical converter/export
graph and precision choices still require investigation; do not blame a weight mismatch
or normalization scale alone. Default artifacts and installed Runtime/driver unchanged.

Evidence: `results/real_eval_200_20261011/recovered_original_fidelity_unit_contract_v2.json`.
The previous report is retained; v2 distinguishes raw and explicit unit-output contracts.
