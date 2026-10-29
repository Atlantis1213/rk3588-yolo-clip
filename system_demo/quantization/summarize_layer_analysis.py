"""Parse actual SDK error table and compare simulator to board/reference."""
import json
from pathlib import Path
import re
import numpy as np

root = Path(__file__).resolve().parent / "artifacts"
diagnosis = root / "diagnosis_20261010"
analysis = diagnosis / "layer_analysis"
if not json.loads((analysis / "status.json").read_text())["completed"]:
    raise RuntimeError("layer analysis has not completed")
pattern = re.compile(r"^\[(?P<op>[^\]]+)\]\s+(?P<tensor>\S+)\s+"
                     r"(?P<entire_cos>\S+)\s*\|\s*(?P<entire_euc>\S+)\s+"
                     r"(?P<single_cos>\S+)\s*\|\s*(?P<single_euc>\S+)\s*$")
layers = []
for line in (analysis / "snapshot/error_analysis.txt").read_text().splitlines():
    if not line.startswith("["):
        continue
    match = pattern.fullmatch(line)
    if match is None:
        raise RuntimeError("unparsed SDK table row: " + line)
    row = match.groupdict()
    for key in ("entire_cos", "entire_euc", "single_cos", "single_euc"):
        row[key] = float(row[key])
    row["order"] = len(layers)
    layers.append(row)
if not layers:
    raise RuntimeError("no SDK layers parsed")


def metric(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return {"cosine": float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))),
            "rmse": float(np.sqrt(np.mean((a-b)**2)))}


names = [r["image"] for r in json.loads((diagnosis / "reference_comparison.json").read_text())["rows"]]
reference = np.load(diagnosis / "torch.f32.npy")
simulator = np.load(analysis / "simulator_int8.f32.npy")
board_dir = root / "quant_public_int8_normal_ieee"
board_manifest = json.loads((board_dir / "manifest.json").read_text())
embeddings = np.fromfile(board_dir / "embeddings.f32", np.float32).reshape(-1, 768)
board = {s["image"]: value for s, value in zip(board_manifest["samples"], embeddings)}
comparisons = [{"image": name, "torch_to_simulator": metric(reference[i], simulator[i]),
                "board_to_simulator": metric(board[name], simulator[i])} for i, name in enumerate(names)]
paired_probe = diagnosis / "quant_io_diagnostic_20261010"
if (paired_probe / "manifest.json").exists():
    probe_metadata = json.loads((paired_probe / "manifest.json").read_text())
    for i, row in enumerate(comparisons):
        selected = {r["mode"]: np.fromfile(paired_probe / r["file"], np.float32)
                    for r in probe_metadata["rows"] if r["image"] == names[i] + ".bin"}
        row["same_artifact_board_host_to_simulator"] = metric(selected["fp16_host"], simulator[i])
        row["same_artifact_manual_int8_to_simulator"] = metric(selected["int8_host"], simulator[i])
        row["same_artifact_board_host_to_prior_board"] = metric(selected["fp16_host"], board[names[i]])
        row["torch_to_same_artifact_board"] = metric(reference[i], selected["fp16_host"])
golden = np.loadtxt(analysis / "snapshot/golden/image_embedding.txt")
summary = {"sample": names[0], "layer_count": len(layers),
           "sdk_golden_to_torch": metric(reference[0], golden),
           "first_single_cos_below_099": next((r for r in layers if r["single_cos"] < .99), None),
           "first_entire_cos_below_09": next((r for r in layers if r["entire_cos"] < .9), None),
           "worst_single_layers": sorted(layers, key=lambda r: r["single_cos"])[:12],
           "final_layer": layers[-1], "simulator_comparisons": comparisons,
           "layers": layers}
(diagnosis / "layer_summary.json").write_text(json.dumps(summary, indent=2))
policy = [r["tensor"] for r in layers if r["op"] in {"exNorm", "Add"}]
(diagnosis / "hybrid_policy_norm_residual.json").write_text(json.dumps({
    "policy": "protect_norm_and_residual_add", "precision": "float16", "count": len(policy),
    "tensors": policy,
    "reason": "First significant single-layer error is pre-LN; residual add error accumulates. Weighted Conv/MatMul remains quantized where SDK supports it.",
    "note": "Candidate policy derived from one fixed validation trace; accuracy/speed are not yet validated."}, indent=2))
print(json.dumps({k: v for k, v in summary.items() if k != "layers"}, indent=2))
print("Protected norm/residual tensors: " + str(len(policy)))
attention_policy = [r["tensor"] for r in layers if r["op"] in {"exNorm", "Add", "exSDPAttention"}]
(diagnosis / "hybrid_policy_sensitive.json").write_text(json.dumps({
    "policy": "protect_norm_residual_and_attention", "precision": "float16", "count": len(attention_policy),
    "tensors": attention_policy,
    "reason": "Pre-LN first drops; later norm/residual and block22 attention have large single-layer errors. Weighted Conv/MatMul is not explicitly protected.",
    "note": "Candidate, not accepted; SDK may expand FP16 boundaries, so inspect actual model size, accuracy and speed."}, indent=2))
