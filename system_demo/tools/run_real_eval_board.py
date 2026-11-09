# 板端真实数据评估编排：独占 NPU 实验锁，先冻结一次 YOLO ROI，再让各 CLIP 读取同一输入。
# 脚本有实际硬件执行与写文件行为；阅读/语法检查不需要运行整个脚本。

"""Independent real-data pipeline evaluation, with one frozen ROI/input per image."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_method_sweep_board import digest

root = Path(__file__).resolve().parents[1]
output = root / "results/real_eval_200_20261011"
# 非阻塞文件锁协调同协议的实验脚本；进程列表再检查其他已知 NPU 工具，但不能代替系统级设备隔离。
lock = (root / ".method_sweep_npu.lock").open("a")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
processes = subprocess.run(["ps", "-eo", "comm"], capture_output=True, text=True, check=True).stdout
if any(
    name.startswith(
        ("dump_clip", "dump_prefix", "bench_clip", "yolo_clip_syste", "prepare_real", "dump_frozen")
    )
    for name in processes.splitlines()
):
    raise RuntimeError("another NPU workload is active")
# 仅给子进程指定实验 Runtime，不替换系统安装的 RKNN 库。
environment = dict(os.environ, LD_LIBRARY_PATH=str(root / "quant_runtime232"))
models = {
    "deployed_historical": root / "clip_image_encoder_vitl14.rknn",
    "official_fp16": root / "clip_reference_fp16_232.rknn",
    "official_kl120": root
    / "quantization/artifacts/method_sweep_20261011/kl120_reference/clip.rknn",
}
hashes = {name: digest(path) for name, path in models.items()}
info = {
    "completed": False,
    "phase": "started",
    "model_sha": hashes,
    "variants": [],
    "yolo_sha": digest(root / "yolo_waste_single_class.rknn"),
    "yolo_confidence": 0.05,
    "empty_bypass_yolo_only": True,
    "detection_failure_policy": "count unknown in end-to-end task; full-frame tensor only for diagnostic CLIP comparisons",
    "same_frozen_inputs_all_variants": True,
}


# 临时文件写完后原子替换状态，减少中断时读到半个 JSON 的可能。
def update(phase, **details):
    info.update(phase=phase, **details)
    temporary = output / "board_status.tmp"
    temporary.write_text(json.dumps(info, indent=2))
    os.replace(temporary, output / "board_status.json")
    print("REAL200 " + phase, flush=True)


def run(command, log, timeout):
    result = subprocess.run(
        list(map(str, command)),
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    log.write_text(result.stdout + "\n" + result.stderr)
    result.check_returncode()
    print(result.stdout, flush=True)


try:
    if (output / "frozen").exists():
        raise RuntimeError(
            "frozen ROI directory already exists; do not re-run YOLO or overwrite it"
        )
    update("prepare_frozen_yolo_rois")
    run(
        [
            root / "build/prepare_real_clip_inputs",
            root / "yolo_waste_single_class.rknn",
            output / "dataset",
            output / "frozen",
            0.05,
        ],
        output / "prepare_rois.log",
        180,
    )
    crop_manifest = json.loads((output / "frozen/manifest.json").read_text())
    info["detected_nonempty"] = crop_manifest["detected_nonempty"]
    info["empty_count"] = crop_manifest["empty_count"]
    # 冻结每张输入的 SHA，并在所有候选跑完后再次核对，确保没有在比较期间重新裁剪或改写。
    tensors = {
        row["image"]: digest(output / "frozen/inputs_fp16" / (row["image"] + ".bin"))
        for row in crop_manifest["samples"]
    }
    (output / "frozen/input_sha256.json").write_text(json.dumps(tensors, indent=2))
    # 每个候选单独保存日志与 manifest，检查模型哈希未变化；completed 只表示采集流程完成。
    for name, model in models.items():
        update("inference_" + name)
        run(
            [
                root / "build/dump_frozen_clip_inputs",
                model,
                output / "dataset",
                output / "frozen",
                output / name,
            ],
            output / (name + ".log"),
            600,
        )
        if digest(model) != hashes[name]:
            raise RuntimeError("model SHA changed during evaluation")
        info["variants"].append(name)
    if any(
        digest(output / "frozen/inputs_fp16" / (name + ".bin")) != value
        for name, value in tensors.items()
    ):
        raise RuntimeError("frozen input changed during model comparisons")
    update("completed", completed=True)
except Exception as error:
    update("failed", error=repr(error))
    raise
