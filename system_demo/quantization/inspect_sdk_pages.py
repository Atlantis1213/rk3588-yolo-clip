"""Read-only SDK page routing; --pages prints complete selected pages."""
import argparse
from pathlib import Path
from pypdf import PdfReader

parser = argparse.ArgumentParser()
parser.add_argument("--document", choices=["api", "guide"], required=True)
parser.add_argument("--search", default="accuracy_analysis")
parser.add_argument("--pages", default="")
args = parser.parse_args()
filename = "RKNN_Toolkit2_API_2.3.2_EN.pdf" if args.document == "api" else "RKNN_UserGuide_2.3.2_EN.pdf"
reader = PdfReader(Path(__file__).resolve().parent / "assets" / filename)
selected = {int(value) for value in args.pages.split(",") if value}
for number, page in enumerate(reader.pages, start=1):
    content = page.extract_text() or ""
    if selected:
        if number in selected:
            print(f"DOCUMENT {filename}, PAGE {number}\n{content}")
    elif args.search.lower() in content.lower():
        print(f"PAGE {number}")
        print("\n".join(line for line in content.splitlines() if args.search.lower() in line.lower()))
