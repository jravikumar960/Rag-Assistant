
import fitz
from pathlib import Path

print("DEBUG: Script started")

pdf_path = Path("documents") / "DEEP LEARNING ( R20A06610 ).pdf"

print("DEBUG: PDF path:", pdf_path.resolve())
print("DEBUG: PDF exists:", pdf_path.exists())

if not pdf_path.exists():
    raise FileNotFoundError(f"PDF not found: {pdf_path.resolve()}")

with fitz.open(pdf_path) as pdf:
    print("DEBUG: Total PDF pages:", len(pdf))

    page = pdf[1]  # Physical page 2
    text = page.get_text("text", sort=True)

    print("\n--- EXTRACTED PAGE 2 TEXT ---")
    print(repr(text))
    print("\n--- FORMATTED TEXT ---")
    print(text if text.strip() else "No selectable text found on page 2.")
