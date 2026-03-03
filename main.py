from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from datetime import datetime, timezone, date
from typing import Any, Dict, List
from bson import ObjectId

from src.workflow import run_workflow


# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
def configure_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pipeline.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def safe_get(d: Dict[str, Any], *keys, default=None):
    current = d
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def sanitize_filename(name: str) -> str:
    """
    Create a filesystem-safe filename stem.
    """
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def make_json_safe(obj):
    """
    Recursively convert objects into JSON-serializable values.
    """
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, ObjectId):
        return str(obj)

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]

    # fallback for unknown objects
    return str(obj)


def save_json(data: Dict[str, Any] | List[Dict[str, Any]], output_path: Path) -> None:
    """
    Save JSON-safe data to disk.

    Args:
        data: The Python object to save.
        output_path: Destination JSON file path.
    """
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_data = make_json_safe(data)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(safe_data, f, indent=4, ensure_ascii=False)


def build_result_summary(result: Dict[str, Any], pdf_file: Path) -> Dict[str, Any]:
    triage_decision = safe_get(result, "triage", "triage_decision", default={}) or {}
    metadata = safe_get(result, "extraction", "metadata", default={}) or {}
    email_send = safe_get(result, "email", "send_result", default={}) or {}

    return {
        "file_name": pdf_file.name,
        "file_path": str(pdf_file),
        "final_status": result.get("final_status"),
        "final_message": result.get("final_message"),
        "triage_status": safe_get(result, "triage", "triage_status"),
        "route": triage_decision.get("route"),
        "document_type": metadata.get("document_type"),
        "policy_number": metadata.get("policy_number"),
        "date_of_loss": metadata.get("date_of_loss"),
        "recipient": metadata.get("recipient"),
        "claimant": metadata.get("claimant"),
        "defendant": metadata.get("defendant"),
        "case_court_reference_number": metadata.get("case_court_reference_number"),
        "email_status": safe_get(result, "email", "email_status"),
        "email_send_status": email_send.get("send_status"),
        "errors": {
            "triage": safe_get(result, "triage", "errors", default=[]),
            "extraction": safe_get(result, "extraction", "errors", default=[]),
            "storage": safe_get(result, "storage", "errors", default=[]),
            "email": safe_get(result, "email", "errors", default=[]),
            "email_send": email_send.get("error"),
        },
    }


def save_individual_result(
    result: Dict[str, Any],
    summary: Dict[str, Any],
    pdf_file: Path,
    results_dir: Path,
) -> Path:
    """
    Save one full workflow result per file as JSON.
    """
    safe_name = sanitize_filename(pdf_file.stem)
    output_file = results_dir / f"{safe_name}.json"

    payload = {
        "file_name": pdf_file.name,
        "file_path": str(pdf_file),
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "result": result,
    }

    save_json(payload, output_file)
    return output_file


# -----------------------------------------------------------------------------
# Batch runner
# -----------------------------------------------------------------------------
def run_batch_workflow(
    input_dir: str,
    source: str = "uploaded_pdf",
    output_dir: str = "outputs",
) -> List[Dict[str, Any]]:
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    logs_path = output_path / "logs"
    results_path = output_path / "results"
    reports_path = output_path / "reports"

    configure_logging(logs_path)

    logger.info("Starting batch workflow")
    logger.info("Input directory: %s", input_path.resolve())
    logger.info("Output directory: %s", output_path.resolve())

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")

    pdf_files = sorted(input_path.glob("*.pdf"))

    if not pdf_files:
        logger.warning("No PDF files found in directory: %s", input_path)
        return []

    logger.info("Found %d PDF file(s) to process", len(pdf_files))

    all_results: List[Dict[str, Any]] = []
    succeeded = 0
    partial = 0
    failed = 0

    for index, pdf_file in enumerate(pdf_files, start=1):
        logger.info("Processing file %d/%d | %s", index, len(pdf_files), pdf_file.name)

        try:
            result = run_workflow(pdf_path=str(pdf_file), source=source)
            summary = build_result_summary(result, pdf_file)

            output_file = save_individual_result(
                result=result,
                summary=summary,
                pdf_file=pdf_file,
                results_dir=results_path,
            )

            record = {
                "file_name": pdf_file.name,
                "output_json": str(output_file),
                "summary": summary,
            }
            all_results.append(record)

            final_status = result.get("final_status")
            if final_status == "success":
                succeeded += 1
            elif final_status == "partial_success":
                partial += 1
            else:
                failed += 1

            logger.info(
                "Finished file | name=%s | final_status=%s | route=%s | doc_type=%s | saved=%s",
                pdf_file.name,
                summary.get("final_status"),
                summary.get("route"),
                summary.get("document_type"),
                output_file.name,
            )

        except Exception as e:
            failed += 1
            logger.exception("Unhandled failure while processing file: %s", pdf_file.name)

            summary = {
                "file_name": pdf_file.name,
                "file_path": str(pdf_file),
                "final_status": "failed",
                "final_message": f"Unhandled batch runner failure: {str(e)}",
                "triage_status": None,
                "route": None,
                "document_type": None,
                "policy_number": None,
                "date_of_loss": None,
                "recipient": None,
                "claimant": None,
                "defendant": None,
                "case_court_reference_number": None,
                "email_status": None,
                "email_send_status": None,
                "errors": {
                    "triage": [],
                    "extraction": [],
                    "storage": [],
                    "email": [],
                    "email_send": str(e),
                },
            }

            failure_result = {
                "final_status": "failed",
                "final_message": f"Unhandled batch runner failure: {str(e)}",
            }

            output_file = save_individual_result(
                result=failure_result,
                summary=summary,
                pdf_file=pdf_file,
                results_dir=results_path,
            )

            all_results.append(
                {
                    "file_name": pdf_file.name,
                    "output_json": str(output_file),
                    "summary": summary,
                }
            )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_report_file = reports_path / f"batch_results_summary_{timestamp}.json"

    save_json(all_results, summary_report_file)

    logger.info("Batch workflow completed")
    logger.info("Success: %d | Partial: %d | Failed: %d", succeeded, partial, failed)
    logger.info("Per-file JSON results saved in: %s", results_path.resolve())
    logger.info("Batch summary report saved to: %s", summary_report_file.resolve())

    return all_results


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    input_dir = r"C:\Users\tanne\UK_interviews\Lloyds\Lloyd’s of London - AI Engineer - Technical Assignment\Mock PDF Documents"

    results = run_batch_workflow(
        input_dir=input_dir,
        source="uploaded_pdf",
        output_dir="outputs",
    )

    print(f"\nProcessed {len(results)} file(s).")
    for item in results:
        summary = item["summary"]
        print(
            f"- {summary['file_name']} | "
            f"status={summary['final_status']} | "
            f"route={summary['route']} | "
            f"doc_type={summary['document_type']} | "
            f"json={item['output_json']}"
        )