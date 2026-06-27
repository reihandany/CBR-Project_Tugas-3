import os
import pytesseract
from pdf2image import convert_from_path

# Lokasi tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

pdf_folder = "data/pdf"
output_folder = "data/raw"

os.makedirs(output_folder, exist_ok=True)

pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

print(f"Jumlah PDF: {len(pdf_files)}")

for i, pdf_file in enumerate(pdf_files, start=1):

    print(f"[{i}/{len(pdf_files)}] {pdf_file}")

    pdf_path = os.path.join(pdf_folder, pdf_file)

    pages = convert_from_path(
        pdf_path,
        dpi=300,
        poppler_path=r"C:\poppler-26.02.0\Library\bin"
    )

    full_text = ""

    for page in pages:

        text = pytesseract.image_to_string(
            page,
            lang="ind+eng"
        )

        full_text += text + "\n"

    txt_name = os.path.splitext(pdf_file)[0] + ".txt"

    with open(
        os.path.join(output_folder, txt_name),
        "w",
        encoding="utf-8"
    ) as f:

        f.write(full_text.lower())

print("Selesai.")