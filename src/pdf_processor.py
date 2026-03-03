from pathlib import Path
from typing import Dict, List, Any
from pypdf import PdfReader
import logging
import re

# -----------------------------------------------------------------------------
# Logger configuration
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )


def extract_text_from_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Extract text page by page from a PDF using pypdf.
    Returns structured output for downstream processing.
    """
    pdf_path = Path(pdf_path)
    logger.info(f"Starting PDF extraction for file: {pdf_path.name}")

    result = {
        "file_name": pdf_path.name,
        "file_path": str(pdf_path),
        "num_pages": 0,
        "pages": [],
        "full_text": "",
        "extraction_status": "success",
        "warnings": [],
        "errors": []
    }

    try:
        reader = PdfReader(str(pdf_path))
        result["num_pages"] = len(reader.pages)
        logger.info(f"Loaded PDF successfully: {pdf_path.name} | pages={result['num_pages']}")

        full_text_parts: List[str] = []

        for i, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text()
                text = text.strip() if text else ""

                if not text:
                    warning_msg = f"Page {i} returned no text."
                    logger.warning(f"{pdf_path.name} | {warning_msg}")
                    result["warnings"].append(warning_msg)
                else:
                    logger.info(f"{pdf_path.name} | Extracted text from page {i} | chars={len(text)}")

                result["pages"].append({
                    "page_number": i,
                    "text": text
                })

                full_text_parts.append(f"\n--- PAGE {i} ---\n{text}")

            except Exception as page_error:
                warning_msg = f"Failed to extract text from page {i}: {str(page_error)}"
                logger.exception(f"{pdf_path.name} | {warning_msg}")
                result["warnings"].append(warning_msg)
                result["pages"].append({
                    "page_number": i,
                    "text": ""
                })

        result["full_text"] = "\n".join(full_text_parts).strip()
        logger.info(
            f"Completed extraction for {pdf_path.name} | total_chars={len(result['full_text'])}"
        )

    except Exception as e:
        error_msg = f"Could not read PDF: {str(e)}"
        logger.exception(f"{pdf_path.name} | {error_msg}")
        result["extraction_status"] = "failed"
        result["errors"].append(error_msg)

    return result


def assess_text_quality(text: str) -> Dict[str, Any]:
    """
    Simple text-quality checks for routing and validation.
    """
    logger.info("Assessing extracted text quality")

    warnings = []
    char_count = len(text)
    word_count = len(text.split())

    suspicious_chars = len(re.findall(r"[�□◊�]", text))
    suspicious_ratio = suspicious_chars / char_count if char_count > 0 else 0

    if char_count == 0:
        warnings.append("No text extracted from PDF.")
    elif word_count < 30:
        warnings.append("Very little text extracted; document may be image-based or corrupted.")

    if suspicious_ratio > 0.01:
        warnings.append("Extracted text contains suspicious characters and may be corrupted.")

    for warning in warnings:
        logger.warning(f"Text quality warning: {warning}")

    logger.info(
        f"Text quality assessed | chars={char_count} | words={word_count} "
        f"| suspicious_chars={suspicious_chars} | suspicious_ratio={suspicious_ratio:.4f}"
    )

    return {
        "char_count": char_count,
        "word_count": word_count,
        "suspicious_char_count": suspicious_chars,
        "suspicious_ratio": suspicious_ratio,
        "warnings": warnings,
        "is_usable": char_count > 0
    }


def process_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Full PDF processing pipeline for a single file.
    """
    logger.info(f"Processing started for PDF: {pdf_path}")

    extraction_result = extract_text_from_pdf(pdf_path)

    if extraction_result["extraction_status"] == "failed":
        logger.error(f"Processing failed for PDF: {pdf_path}")
        return extraction_result

    quality = assess_text_quality(extraction_result["full_text"])
    extraction_result["text_quality"] = quality
    extraction_result["warnings"].extend(quality["warnings"])

    logger.info(
        f"Processing completed for PDF: {pdf_path} | "
        f"status={extraction_result['extraction_status']} | "
        f"warnings={len(extraction_result['warnings'])}"
    )

    return extraction_result