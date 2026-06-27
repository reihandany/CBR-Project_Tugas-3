import os
import re
import json
from collections import Counter
import pandas as pd

# Folder
raw_folder = "data/raw"
output_folder = "data/processed"

os.makedirs(output_folder, exist_ok=True)

txt_files = sorted([
    f for f in os.listdir(raw_folder)
    if f.endswith(".txt")
])

cases = []

def clean_text(text):

    text = text.lower()

    # satukan spasi & enter
    text = re.sub(r"\s+", " ", text)

    # ----------------------------
    # Hapus watermark Mahkamah Agung
    # ----------------------------

    patterns = [

        r"direktori putusan",

        r"putusan\.mahkamahagung\.go\.id",

        r"mahkamah agung republik indonesia",

        r"pid\.\d+\.[a-z]\.\d+",

        r"halaman\s+\d+",

        r"www\.\S+",

        r"\S+@\S+",

        r"\d{3}-\d{7,}",

        r"disclaimer kepaniteraan.*?pelayanan publik",

        r"transparansi dan akuntabilitas pelaksanaan fungsi peradilan",

        r"informasi paling kini.*?waktu",

        r"demi keadilan berdasarkan ketuhanan yang maha esa",

    ]

    for p in patterns:
        text = re.sub(
            p,
            " ",
            text,
            flags=re.IGNORECASE
        )

    # ----------------------------
    # Hilangkan huruf OCR acak
    # contoh:
    # e h a
    # r a i
    # l m b
    # ----------------------------

    text = re.sub(
        r"(?:\b[a-z]\b\s*){3,}",
        " ",
        text
    )

    # hilangkan karakter aneh
    text = re.sub(
        r"[^a-z0-9(),./:%\- ]",
        " ",
        text
    )

    # rapikan spasi
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# Fungsi Ekstraksi
def extract_no_perkara(text):

    match = re.search(
        r"nomor\s+([\w./ -]+)",
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1).strip()

    return ""


def extract_terdakwa(text):
    match = re.search(
        r'nama lengkap\s*:\s*(.*?)\s*(?:tempat lahir|umur)',
        text,
        re.IGNORECASE | re.DOTALL
    )

    if match:
        return " ".join(match.group(1).split())

    return ""


def extract_parties(text):
    matches = re.findall(
        r'(penggugat|tergugat)\s*[:\-]\s*(.+?)(?=\s*(?:penggugat|tergugat|kuasa hukum|alamat|tempat lahir|umur|tanggal|nomor|pasal|mengingat|menimbang|mengadili|demikian|$))',
        text,
        re.IGNORECASE | re.DOTALL
    )

    parties = {
        "penggugat": "",
        "tergugat": ""
    }

    for role, value in matches:
        parties[role.lower()] = " ".join(value.split())

    return parties["penggugat"], parties["tergugat"]


def extract_argumen_hukum_utama(text):
    patterns = [
        r'menimbang(.*?)(?=mengadili|mengingat|demikian diputuskan|putusan)',
        r'mengingat(.*?)(?=mengadili|demikian diputuskan|putusan)',
    ]

    text = re.sub(r"\s+", " ", text)

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            re.IGNORECASE | re.DOTALL
        )

        if match:
            hasil = match.group(1).strip()
            hasil = re.sub(r"\s+", " ", hasil)
            return hasil[:800]

    return ""


def extract_pasal(text):

    match = re.search(
        r"pasal\s+\d+\s+ayat\s*\(\d+\)",
        text,
        re.IGNORECASE
    )

    if match:
        return match.group().title()

    match = re.search(
        r"pasal\s+\d+",
        text,
        re.IGNORECASE
    )

    if match:
        return match.group().title()

    return ""

def extract_jenis_perkara(text):

    text = text.lower()

    # Narkotika
    if "narkotika" in text:
        return "Narkotika"

    # Psikotropika
    elif "psikotropika" in text:
        return "Psikotropika"

    # Perlindungan Anak
    elif "perlindungan anak" in text:
        return "Perlindungan Anak"

    # Korupsi
    elif "korupsi" in text or "tipikor" in text:
        return "Korupsi"

    # Pencurian
    elif "pencurian" in text:
        return "Pencurian"

    # Penggelapan
    elif "penggelapan" in text:
        return "Penggelapan"

    # Penipuan
    elif "penipuan" in text:
        return "Penipuan"

    # Penganiayaan
    elif "penganiayaan" in text:
        return "Penganiayaan"

    # Pembunuhan
    elif "pembunuhan" in text:
        return "Pembunuhan"

    # Default
    else:
        return "Lainnya"


def extract_tanggal(text):

    match = re.search(
        r"tanggal\s+(\d+\s+\w+\s+\d{4})",
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1)

    return ""


def extract_ringkasan(text):

    match = re.search(
        r"bahwa(.*?)(?=menimbang)",
        text,
        re.IGNORECASE | re.DOTALL
    )

    if match:
        hasil = re.sub(r"\s+", " ", match.group(1))
        return hasil[:800]

    return text[:800]

def extract_amar_putusan(text):

    text = re.sub(r"\s+", " ", text)

    patterns = [

        r'mengadili\s*:?(.*?)(?=menetapkan barang bukti)',

        r'mengadili\s*:?(.*?)(?=membebankan biaya perkara)',

        r'menjatuhkan pidana.*?(?=menetapkan barang bukti)',

        r'menjatuhkan pidana.*?(?=membebankan biaya perkara)',

        r'menjatuhkan pidana.*?(?=demikian diputuskan)',

    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            hasil = match.group()

            hasil = re.sub(r"\s+", " ", hasil)

            # Batasi panjang agar tidak terlalu panjang
            return hasil[:350]

    return ""

def extract_lama_pidana(text):

    text = re.sub(r"\s+", " ", text)

    patterns = [

    r'pidana penjara selama\s+(.+?)\s+dikurangi',

    r'pidana penjara selama\s+(.+?)\s+dan',

    r'pidana penjara selama\s+(.+?)\.',

    r'pidana penjara selama\s+(.+?)\;',

    r'penjara selama\s+(.+?)\s+dikurangi',

]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            return match.group(1).strip()

    return ""

# Proses Semua TXT
for i, file in enumerate(txt_files, start=1):
    path = os.path.join(raw_folder, file)

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Bersihkan text terlebih dahulu
    text = clean_text(text)

    no_perkara = extract_no_perkara(text)
    terdakwa = extract_terdakwa(text)
    pasal = extract_pasal(text)
    jenis_perkara = extract_jenis_perkara(text)
    tanggal = extract_tanggal(text)
    ringkasan = extract_ringkasan(text)
    amar_putusan = extract_amar_putusan(text)
    lama_pidana = extract_lama_pidana(text)
    penggugat, tergugat = extract_parties(text)
    argumen_hukum_utama = extract_argumen_hukum_utama(text)

    tokens = re.findall(r"\b[a-z0-9]+\b", text)
    token_counts = Counter(tokens)
    top_words = [f"{word}:{count}" for word, count in token_counts.most_common(20)]

    combined_pihak = ""
    if penggugat and tergugat:
        combined_pihak = f"{penggugat} vs {tergugat}"
    elif penggugat:
        combined_pihak = penggugat
    elif tergugat:
        combined_pihak = tergugat
    else:
        combined_pihak = terdakwa

    cases.append({
        "case_id": i,
        "no_perkara": no_perkara,
        "tanggal": tanggal,
        "jenis_perkara": jenis_perkara,
        "ringkasan_fakta": ringkasan,
        "argumen_hukum_utama": argumen_hukum_utama,
        "pasal": pasal,
        "pihak": combined_pihak,
        "pihak_penggugat": penggugat,
        "pihak_tergugat": tergugat,
        "pihak_terdakwa": terdakwa,
        "amar_putusan": amar_putusan,
        "lama_pidana": lama_pidana,
        "word_count": len(tokens),
        "unique_word_count": len(token_counts),
        "top_words": ";".join(top_words),
        "text_full": text
    })

# DataFrame
df = pd.DataFrame(cases)

json_output_file = os.path.join(output_folder, "cases.json")
with open(json_output_file, "w", encoding="utf-8") as f:
    json.dump(cases, f, ensure_ascii=False, indent=2)

print("=" * 60)
print("CASE REPRESENTATION BERHASIL")
print("=" * 60)

print(df.head())
print(f"\nJumlah kasus : {len(df)}")
print(f"Hasil disimpan di : {json_output_file}")