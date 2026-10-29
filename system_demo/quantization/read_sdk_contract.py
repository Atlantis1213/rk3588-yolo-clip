from pathlib import Path
from pypdf import PdfReader

root = Path(__file__).resolve().parent
for filename in ("RKNN_Toolkit2_API_2.3.2_EN.pdf", "RKNN_UserGuide_2.3.2_EN.pdf"):
    reader = PdfReader(root / "assets" / filename)
    for index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        lower = text.lower()
        if ".npy" in lower or ("dataset" in lower and "nchw" in lower):
            print(f"DOCUMENT {filename}, PAGE {index+1}")
            print(text)
