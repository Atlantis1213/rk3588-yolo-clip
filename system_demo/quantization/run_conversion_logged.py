"""Keep compiler progress in an artifact instead of flooding the chat/terminal."""
import argparse
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("mode")
args = parser.parse_args()
root = Path(__file__).resolve().parent
logs = root / "artifacts/logs"; logs.mkdir(exist_ok=True)
start = time.monotonic()
with (logs / (args.mode + ".log")).open("w") as log:
    process = subprocess.Popen([__import__("sys").executable, str(root / "convert_rknn.py"), "--mode", args.mode],
                               stdout=log, stderr=subprocess.STDOUT)
    while process.poll() is None:
        time.sleep(10)
        elapsed = time.monotonic() - start
        if int(elapsed) // 30 != int(elapsed - 10) // 30:
            print(f"RKNN {args.mode} build running: {elapsed:.0f}s (full log retained)", flush=True)
if process.returncode:
    tail = (logs / (args.mode + ".log")).read_text(errors="replace")[-6000:].replace("\r", "\n")
    print(tail, flush=True)
    raise SystemExit(process.returncode)
print(f"RKNN {args.mode} conversion completed in {time.monotonic()-start:.1f}s", flush=True)
