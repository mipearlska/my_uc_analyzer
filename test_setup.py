"""Quick test to verify setup works."""
import pdfplumber
from pathlib import Path

pdf_path = Path("data/etsi_use_case_aiagent.pdf")

if not pdf_path.exists():
    print(f"ERROR: PDF not found at {pdf_path}")
    exit(1)

with pdfplumber.open(pdf_path) as pdf:
    print(f"PDF loaded successfully!")
    print(f"Total pages: {len(pdf.pages)}")
    
    # Extract first page text
    first_page = pdf.pages[0]
    text = first_page.extract_text()
    print(f"\n--- First page preview (first 500 chars) ---")
    print(text[:500] if text else "No text extracted")