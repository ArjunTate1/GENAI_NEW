"""
Feature extraction, multimodal text parsers, and sample data for the GenAI Day classifier.
"""
import io
import os
import re
import string
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from PIL import Image, ImageOps, ImageStat

# The exact 10 features in order expected by linguistic_scaler.joblib
ORDERED_FEATURES = [
    "char_count",
    "word_count",
    "avg_word_length",
    "sentence_count",
    "comma_count",
    "punctuation_count",
    "punctuation_ratio",
    "uppercase_ratio",
    "digit_ratio",
    "type_token_ratio",
]

FEATURE_META = {
    "char_count": {"name": "Character Count", "desc": "Total characters in text", "unit": "chars"},
    "word_count": {"name": "Word Count", "desc": "Total word tokens", "unit": "words"},
    "avg_word_length": {"name": "Avg Word Length", "desc": "Mean characters per word", "unit": "chars/word"},
    "sentence_count": {"name": "Sentence Count", "desc": "Total recognized sentences", "unit": "sentences"},
    "comma_count": {"name": "Comma Count", "desc": "Frequency of commas used", "unit": "commas"},
    "punctuation_count": {"name": "Total Punctuation", "desc": "Count of punctuation marks", "unit": "marks"},
    "punctuation_ratio": {"name": "Punctuation Density", "desc": "Punctuation marks per character", "unit": "ratio"},
    "uppercase_ratio": {"name": "Uppercase Density", "desc": "Uppercase letters per character", "unit": "ratio"},
    "digit_ratio": {"name": "Numeric Density", "desc": "Digit characters per character", "unit": "ratio"},
    "type_token_ratio": {"name": "Lexical Diversity (TTR)", "desc": "Unique words divided by total words", "unit": "ratio"},
}


def compute_linguistic_dict(text: str) -> Dict[str, float]:
    """Computes stylometric and linguistic features from raw text."""
    clean_text = text.strip()
    words = re.findall(r"\b\w+\b", clean_text)
    sentences = [s for s in re.split(r"[.!?]+", clean_text) if s.strip()]

    n_chars = max(len(clean_text), 1)
    n_words = max(len(words), 1)
    n_sent = max(len(sentences), 1)
    punct = sum(c in string.punctuation for c in clean_text)
    unique_words = {w.lower() for w in words}

    return {
        "char_count": float(len(clean_text)),
        "word_count": float(n_words),
        "sentence_count": float(n_sent),
        "avg_word_length": float(np.mean([len(w) for w in words])) if words else 0.0,
        "avg_sentence_length": float(n_words / n_sent),
        "type_token_ratio": float(len(unique_words) / n_words),
        "lexical_diversity": float(len(unique_words) / n_words),
        "punctuation_count": float(punct),
        "punctuation_ratio": float(punct / n_chars),
        "comma_count": float(clean_text.count(",")),
        "uppercase_ratio": float(sum(c.isupper() for c in clean_text) / n_chars),
        "digit_ratio": float(sum(c.isdigit() for c in clean_text) / n_chars),
        "long_word_ratio": float(sum(len(w) > 6 for w in words) / n_words),
        "sentence_len_std": float(np.std([len(re.findall(r"\b\w+\b", x)) for x in sentences])) if sentences else 0.0,
        "newline_count": float(clean_text.count("\n")),
        "question_count": float(clean_text.count("?")),
        "exclamation_count": float(clean_text.count("!")),
    }


def linguistic_vector(text: str, scaler) -> np.ndarray:
    """Transforms raw text into a scaled 10-dimensional linguistic feature vector."""
    feats = compute_linguistic_dict(text)
    names = getattr(scaler, "feature_names_in_", None)
    if names is not None:
        vec = [feats[n] for n in names]
    else:
        vec = [feats[n] for n in ORDERED_FEATURES]

    X = np.array(vec, dtype=float).reshape(1, -1)
    return scaler.transform(X)


# ----------------------------------------------------------------------------
# MULTIMODAL EXTRACTION: PDF & IMAGE OCR
# ----------------------------------------------------------------------------
def extract_text_from_pdf(uploaded_file) -> Tuple[str, Dict[str, Any]]:
    """Extracts text from PDF files using pypdf with fallback image OCR if scanned."""
    from pypdf import PdfReader

    file_bytes = uploaded_file.read() if hasattr(uploaded_file, "read") else uploaded_file
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)

    reader = PdfReader(io.BytesIO(file_bytes))
    num_pages = len(reader.pages)
    extracted_pages = []

    for idx, page in enumerate(reader.pages):
        page_text = (page.extract_text() or "").strip()
        if page_text:
            extracted_pages.append(page_text)

    full_text = "\n\n".join(extracted_pages).strip()
    is_scanned = len(full_text) < 20 and num_pages > 0

    metadata = {
        "pages": num_pages,
        "extracted_chars": len(full_text),
        "word_count": len(full_text.split()),
        "is_scanned": is_scanned,
    }
    return full_text, metadata


def get_default_tesseract_path() -> Optional[str]:
    """Finds installed tesseract binary on Windows or system PATH."""
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def preprocess_image_for_ocr(img: Image.Image) -> Image.Image:
    """Enhances image contrast and scale for better OCR recognition."""
    gray = ImageOps.grayscale(img)
    # If the image has dark background, invert it
    if ImageStat.Stat(gray).mean[0] < 128:
        gray = ImageOps.invert(gray)
    gray = ImageOps.autocontrast(gray, cutoff=2)
    # Upscale small images for clearer character recognition
    if max(gray.size) < 2000:
        factor = 2 if max(gray.size) < 1000 else 1.5
        new_size = (int(gray.width * factor), int(gray.height * factor))
        gray = gray.resize(new_size, Image.Resampling.LANCZOS)
    return gray.convert("RGB")


def extract_text_from_image(
    uploaded_file,
    engine: str = "auto",
    tesseract_cmd: Optional[str] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Extracts text from an image using Tesseract or EasyOCR.
    """
    if hasattr(uploaded_file, "read"):
        img = Image.open(uploaded_file).convert("RGB")
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
    else:
        img = Image.open(io.BytesIO(uploaded_file)).convert("RGB")

    orig_size = img.size
    processed_img = preprocess_image_for_ocr(img)

    used_engine = None
    extracted_text = ""
    errors = []

    # Try Tesseract if requested or auto
    if engine in ["auto", "tesseract"]:
        try:
            import pytesseract
            tess_path = tesseract_cmd or get_default_tesseract_path()
            if tess_path and os.path.exists(tess_path):
                pytesseract.pytesseract.tesseract_cmd = tess_path
            raw_text = pytesseract.image_to_string(processed_img)
            extracted_text = raw_text.strip()
            used_engine = f"Tesseract OCR ({'custom' if tesseract_cmd else 'auto-detected'})"
        except Exception as e:
            errors.append(f"Tesseract error: {e}")

    # Fallback to EasyOCR if Tesseract failed or if selected
    if (not extracted_text and engine == "auto") or engine == "easyocr":
        try:
            import easyocr
            reader = easyocr.Reader(["en"], gpu=False)
            results = reader.readtext(np.array(processed_img), detail=0)
            extracted_text = " ".join(results).strip()
            used_engine = "EasyOCR (Deep Learning Engine)"
        except Exception as e:
            errors.append(f"EasyOCR error: {e}")

    if not extracted_text and not used_engine:
        raise RuntimeError("OCR Failed. " + " | ".join(errors))

    metadata = {
        "engine": used_engine or "None",
        "dimensions": f"{orig_size[0]}x{orig_size[1]} px",
        "char_count": len(extracted_text),
        "word_count": len(extracted_text.split()),
    }
    return extracted_text, metadata


# ----------------------------------------------------------------------------
# CURATED SAMPLE TEXTS (FOR 1-CLICK TESTING)
# ----------------------------------------------------------------------------
SAMPLE_TEXTS = {
    "AI-Generated (Scientific Summary)": (
        "Artificial intelligence and large language models have fundamentally reshaped modern natural language "
        "processing architectures. Furthermore, by leveraging self-attention mechanisms and multi-headed transformer "
        "layers, these systems achieve remarkable semantic generalization across diverse textual benchmarks. "
        "In addition, statistical representations such as term-frequency inverse document frequency (TF-IDF) provide "
        "interpretable sparse baselines that complement continuous representation spaces. Consequently, rigorous "
        "empirical evaluation remains paramount to differentiate machine-generated synthesis from genuine human authorship."
    ),
    "Human-Written (Personal Perspective)": (
        "I was sitting on my front porch yesterday evening, drinking some lukewarm coffee and listening to the neighborhood dogs bark. "
        "It's funny how you never really notice the small things until you're forced to sit completely still for an hour or two. "
        "My phone had died about twenty minutes earlier, and instead of plugging it in, I just decided to watch the clouds drift over "
        "the pine trees. Honestly, I think we spend way too much time staring at glowing rectangles and arguing over nonsense online."
    ),
    "Hybrid / Edited (Technical Review)": (
        "Recent evaluations show that while deep sentence embeddings capture broad contextual semantics, stylometric "
        "analysis uncovers crucial discrepancies in syntactic cadence and punctuation usage. We took several drafts generated "
        "by automated assistants and manually tweaked the transitions, yet random forest classifiers trained on lexical diversity "
        "still flagged key irregularities in paragraph-level token variance."
    ),
}
