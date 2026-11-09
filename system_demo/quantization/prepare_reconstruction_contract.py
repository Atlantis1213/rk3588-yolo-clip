# GPU 层重建的参数协议导出：把 SDK 实际使用的 KL120 权重/激活 scale 和 zero_point 固定下来。
# 仅加载已记录的本地产物；pickle 会执行反序列化逻辑，不能用外来不可信文件替代。

"""Export actual SDK KL120 weight scales for bounded GPU layer reconstruction."""
import json
from pathlib import Path
import pickle
import onnx
from record_ir_provenance import digest

root = Path(__file__).resolve().parent
sweep = root / "artifacts/method_sweep_20261011"
directory = root / "artifacts/calibration_algorithms_20261010/kl_divergence"
with (directory / "clip_visual_reference_fp32.data").open("rb") as handle:
    sdk = pickle.load(handle)  # Our own completed, recorded SDK artifact only.
if sdk["quantizer.param"]["qalgorithm"] != "kl_divergence":
    raise RuntimeError("KL120 SDK calibration required")
# Node metadata is unchanged by the existing fold; avoid loading its 1.2 GB external weights.
# 这里只需要节点与参数名映射，不读取 1.2GB 外部权重，避免为元数据查询分配整塔参数内存。
model = onnx.load(sweep / "affine_sources/repq_ln2/source.onnx", load_external_data=False)
nodes = {node.name: node for node in model.graph.node}
records = []
for index in range(24):
    prefix = f"/transformer/resblocks/resblocks.{index}"
    weight_name = nodes[prefix + "/mlp/c_proj/MatMul"].input[1]
    weight = sdk["quant_tab"][weight_name]
    activation_name = prefix + "/mlp/gelu/Mul_1_output_0_tp_rs"
    activation = sdk["quant_tab"][activation_name]
    # 每层降维矩阵有 1024 个输出通道参数；激活按层仅一个 scale。方向与通道数必须先核对。
    if (
        weight["dtype"] != "int8"
        or weight["qmethod"] != "channel"
        or len(weight["scale"]) != 1024
        or len(weight["zero_point"]) != 1024
        or len(activation["scale"]) != 1
    ):
        raise RuntimeError("unexpected matrix/activation scale contract")
    records.append(
        {
            "block": index,
            "source_weight": weight_name,
            "torch_weight_orientation": "output,input",
            "source_onnx_orientation": "input,output",
            "weight": {
                key: weight[key]
                for key in ["scale", "zero_point", "min", "max", "qtype", "qmethod", "dtype"]
            },
            "activation": {
                key: activation[key] for key in ["scale", "zero_point", "min", "max", "dtype"]
            },
        }
    )
# 记录源 ONNX 与 SDK data 的哈希，训练和回填候选时据此确认使用同一量化网格。
contract = {
    "source_sha": digest(root / "artifacts/clip_visual_reference_fp32.onnx"),
    "sdk_data_sha": digest(directory / "clip_visual_reference_fp32.data"),
    "blocks": records,
    "scope": "actual SDK step1 KL120 parameters; not a claim of bit-exact fused-NPU simulation",
}
with (sweep / "reconstruction_contract_kl120.json").open("x") as handle:
    json.dump(contract, handle, indent=2)
print(
    json.dumps(
        {
            "blocks": len(records),
            "weight_scale_count": 24 * 1024,
            "sdk_algorithm": sdk["quantizer.param"]["qalgorithm"],
        },
        indent=2,
    )
)
