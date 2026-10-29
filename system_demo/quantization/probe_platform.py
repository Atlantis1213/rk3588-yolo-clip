"""Record supported precision configurations; do not assume generic docs imply RK3588 support."""
import json
from pathlib import Path
from rknn.api import RKNN

results = []
for precision in ("w8a8", "w8a16", "w4a16"):
    context = RKNN(verbose=False)
    try:
        status = context.config(target_platform="rk3588", quantized_dtype=precision)
        results.append({"platform": "rk3588", "precision": precision, "config_supported": status == 0})
    except Exception as error:
        results.append({"platform": "rk3588", "precision": precision, "config_supported": False,
                        "reason": str(error).splitlines()[0]})
    finally: context.release()
path = Path(__file__).resolve().parent / "artifacts/platform_precision_support.json"
path.write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
