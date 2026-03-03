"""
Streamlit UI for the AI-assisted legal document processing workflow.

Features:
- Accepts either:
  1. a single PDF file path
  2. a folder path containing multiple PDF files
- Runs the LangGraph-based workflow
- Displays per-file:
  - final status
  - triage route
  - extracted metadata
  - email result
- Saves one JSON output per file
- Saves a batch summary JSON for folder runs

Expected existing modules:
- src.workflow
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from src.workflow import run_workflow

try:
    from bson import ObjectId
except ImportError:
    ObjectId = None


# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Legal Document Processing Pipeline",
    page_icon="📄",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def configure_logging() -> logging.Logger:
    """
    Configure and return the application logger.

    Returns:
        A configured logger instance.
    """
    logger = logging.getLogger("streamlit_app")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


LOGGER = configure_logging()


# -----------------------------------------------------------------------------
# JSON safety helpers
# -----------------------------------------------------------------------------
def make_json_safe(obj: Any) -> Any:
    """
    Recursively convert Python objects into JSON-serializable values.

    This is useful because workflow outputs may contain values such as:
    - datetime
    - pathlib.Path
    - MongoDB ObjectId
    - other non-JSON-native types

    Args:
        obj: Any Python object.

    Returns:
        A JSON-safe representation of the object.
    """
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if ObjectId is not None and isinstance(obj, ObjectId):
        return str(obj)

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]

    return str(obj)


def save_json(data: Any, output_path: Path) -> None:
    """
    Save JSON-safe data to disk.

    Args:
        data: The Python object to save.
        output_path: Destination JSON file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(make_json_safe(data), f, indent=4, ensure_ascii=False)


# -----------------------------------------------------------------------------
# Path and file helpers
# -----------------------------------------------------------------------------
def sanitize_filename(name: str) -> str:
    """
    Convert a filename stem into a filesystem-safe string.

    Args:
        name: Raw filename stem.

    Returns:
        A sanitized filename stem.
    """
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def resolve_input_pdfs(input_path_str: str) -> List[Path]:
    """
    Resolve a user-provided input path into one or more PDF files.

    Behavior:
    - If input path is a file, it must be a .pdf file and is returned as a single-item list.
    - If input path is a directory, all .pdf files inside it are returned.
    - If invalid, an exception is raised.

    Args:
        input_path_str: User-provided file path or folder path.

    Returns:
        A list of PDF file paths.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file is not a PDF or folder has no PDFs.
    """
    input_path = Path(input_path_str)

    if not input_path.exists():
        raise FileNotFoundError(f"Path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError("The selected file is not a PDF.")
        return [input_path]

    if input_path.is_dir():
        pdf_files = sorted(input_path.glob("*.pdf"))
        if not pdf_files:
            raise ValueError("The selected folder does not contain any PDF files.")
        return pdf_files

    raise ValueError("The provided path is neither a valid file nor a valid folder.")


# -----------------------------------------------------------------------------
# Result helpers
# -----------------------------------------------------------------------------
def safe_get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Safely retrieve a nested value from a dictionary.

    Args:
        d: Source dictionary.
        *keys: Sequence of nested keys.
        default: Value returned if lookup fails.

    Returns:
        The nested value or the default.
    """
    current = d
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def build_result_summary(result: Dict[str, Any], pdf_file: Path) -> Dict[str, Any]:
    """
    Build a concise summary of a workflow result for one file.

    Args:
        result: Full workflow result.
        pdf_file: Source PDF path.

    Returns:
        A smaller summary dictionary.
    """
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
    }


def save_individual_result(
    result: Dict[str, Any],
    summary: Dict[str, Any],
    pdf_file: Path,
    output_dir: Path,
) -> Path:
    """
    Save one full workflow result as a separate JSON file.

    Args:
        result: Full workflow result.
        summary: Summary dictionary for the file.
        pdf_file: Source PDF path.
        output_dir: Output directory where per-file JSON files are stored.

    Returns:
        The path of the written JSON file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_filename(pdf_file.stem)
    output_file = output_dir / f"{safe_name}.json"

    payload = {
        "file_name": pdf_file.name,
        "file_path": str(pdf_file),
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "result": result,
    }

    save_json(payload, output_file)
    return output_file


def save_batch_summary(
    summaries: List[Dict[str, Any]],
    output_dir: Path,
) -> Path:
    """
    Save a batch summary JSON file.

    Args:
        summaries: List of summary dictionaries.
        output_dir: Output directory for summary reports.

    Returns:
        The path of the written summary JSON file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_file = output_dir / f"batch_results_summary_{timestamp}.json"
    save_json(summaries, output_file)
    return output_file


# -----------------------------------------------------------------------------
# Workflow runner helpers
# -----------------------------------------------------------------------------
def process_single_pdf(
    pdf_file: Path,
    source: str,
    results_dir: Path,
) -> Dict[str, Any]:
    """
    Process a single PDF through the workflow and save a separate JSON output.

    Args:
        pdf_file: PDF path to process.
        source: Source label for workflow storage/tracing.
        results_dir: Directory for per-file JSON output.

    Returns:
        A dictionary containing:
        - file_name
        - output_json
        - summary
        - result
    """
    LOGGER.info("Processing single PDF | file=%s", pdf_file.name)

    result = run_workflow(pdf_path=str(pdf_file), source=source)
    summary = build_result_summary(result, pdf_file)
    output_json = save_individual_result(result, summary, pdf_file, results_dir)

    return {
        "file_name": pdf_file.name,
        "output_json": str(output_json),
        "summary": summary,
        "result": result,
    }


def process_multiple_pdfs(
    pdf_files: List[Path],
    source: str,
    output_base_dir: Path,
) -> Dict[str, Any]:
    """
    Process multiple PDFs one by one and save:
    - one JSON file per PDF
    - one batch summary JSON file

    Args:
        pdf_files: List of PDF paths to process.
        source: Source label for workflow storage/tracing.
        output_base_dir: Base output directory.

    Returns:
        A dictionary containing:
        - items: per-file results
        - summary_json: batch summary JSON path
    """
    results_dir = output_base_dir / "results"
    reports_dir = output_base_dir / "reports"

    items: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    progress_bar = st.progress(0)
    status_placeholder = st.empty()

    for idx, pdf_file in enumerate(pdf_files, start=1):
        status_placeholder.info(f"Processing {idx}/{len(pdf_files)}: {pdf_file.name}")

        try:
            item = process_single_pdf(
                pdf_file=pdf_file,
                source=source,
                results_dir=results_dir,
            )
            items.append(item)
            summaries.append(item["summary"])

        except Exception as e:
            LOGGER.exception("Unhandled error while processing PDF | file=%s", pdf_file.name)

            failure_result = {
                "final_status": "failed",
                "final_message": f"Unhandled processing failure: {str(e)}",
            }

            failure_summary = {
                "file_name": pdf_file.name,
                "file_path": str(pdf_file),
                "final_status": "failed",
                "final_message": f"Unhandled processing failure: {str(e)}",
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
            }

            output_json = save_individual_result(
                result=failure_result,
                summary=failure_summary,
                pdf_file=pdf_file,
                output_dir=results_dir,
            )

            item = {
                "file_name": pdf_file.name,
                "output_json": str(output_json),
                "summary": failure_summary,
                "result": failure_result,
            }
            items.append(item)
            summaries.append(failure_summary)

        progress_bar.progress(idx / len(pdf_files))

    summary_json = save_batch_summary(summaries, reports_dir)
    status_placeholder.success("Processing complete.")

    return {
        "items": items,
        "summary_json": str(summary_json),
    }


# -----------------------------------------------------------------------------
# UI rendering helpers
# -----------------------------------------------------------------------------
def render_summary_cards(items: List[Dict[str, Any]]) -> None:
    """
    Render a compact summary table for processed files.

    Args:
        items: List of per-file result dictionaries.
    """
    rows = []
    for item in items:
        summary = item["summary"]
        rows.append(
            {
                "file_name": summary.get("file_name"),
                "final_status": summary.get("final_status"),
                "route": summary.get("route"),
                "document_type": summary.get("document_type"),
                "policy_number": summary.get("policy_number"),
                "email_send_status": summary.get("email_send_status"),
                "output_json": item.get("output_json"),
            }
        )

    st.dataframe(rows, use_container_width=True)


def render_detailed_result(item: Dict[str, Any]) -> None:
    """
    Render detailed workflow output for a single processed item.

    Args:
        item: A per-file workflow result dictionary.
    """
    summary = item["summary"]
    result = item["result"]

    st.subheader(summary["file_name"])
    st.write(f"**Final status:** {summary.get('final_status')}")
    st.write(f"**Route:** {summary.get('route')}")
    st.write(f"**Document type:** {summary.get('document_type')}")
    st.write(f"**JSON output:** `{item.get('output_json')}`")

    with st.expander("Triage result", expanded=False):
        st.json(make_json_safe(result.get("triage", {})))

    with st.expander("Extraction result", expanded=True):
        st.json(make_json_safe(result.get("extraction", {})))

    with st.expander("Email result", expanded=False):
        st.json(make_json_safe(result.get("email", {})))

    with st.expander("Full workflow result", expanded=False):
        st.json(make_json_safe(result))


def render_sidebar_content() -> tuple[str, str]:
    """
    Render sidebar content describing the business goal and solution approach.

    Returns:
        A tuple containing:
        - source label
        - output directory
    """
    with st.sidebar:
        st.header("Business Goal")
        st.markdown(
            """
            The goal is to reduce the manual effort involved in handling legal documents
            such as notices, lawsuits, court orders, and formal correspondence.

            Today, these documents are processed across multiple teams and regions, which
            can lead to slow turnaround times, inconsistent handling, and limited visibility.
            The wider Service of Suits vision is to build a centralised, AI-assisted workflow
            that improves speed, consistency, traceability, and scalability, while still
            keeping a human in the loop for oversight.
            """
        )

        st.header("How This Prototype Solves It")
        st.markdown(
            """
            This proof of concept implements a thin end-to-end AI-assisted document pipeline
            using the provided mock PDF files.

            The workflow:

            - **loads and processes PDF documents**
            - **checks whether the extracted text is usable**
            - **uses an LLM to triage the document**
            - **routes the document through either full-text extraction or RAG fallback**
            - **extracts key metadata fields**
            - **flags missing or inconsistent information**
            - **stores structured results as JSON and in MongoDB**
            - **generates and sends a draft email notification**

            This demonstrates how legal document intake can be made faster, more structured,
            and more transparent, while still supporting manual review when needed.
            """
        )

        st.header("Key Metadata Extracted")
        st.markdown(
            """
            - **Document Type**
            - **Date of Loss**
            - **Policy Number**
            - **Recipient**
            - **Claimant / Defendant**
            - **Case / Court Reference Number**
            """
        )

        st.header("Input")
        st.markdown(
            """
            You can provide either:

            - a **single PDF file path**
            - or a **folder path** containing multiple PDF files
            """
        )

    source = "uploaded_pdf"
    output_dir_str = "outputs"
    return source, output_dir_str

# -----------------------------------------------------------------------------
# Main UI
# -----------------------------------------------------------------------------
def main() -> None:
    """
    Main Streamlit entrypoint.

    This UI accepts either:
    - a single PDF file path
    - or a folder path containing PDF files

    It then runs the workflow and writes:
    - one JSON result per file
    - one batch summary JSON for multi-file runs
    """
    st.title("AI-Assisted Legal Document Processing")
    st.caption("Process a single PDF or an entire folder of PDFs through the workflow.")

    with st.sidebar:
        source, output_dir_str = render_sidebar_content()

    input_path_str = st.text_input(
        "Enter PDF file path or folder path",
        value="",
        placeholder=r"C:\path\to\file.pdf or C:\path\to\folder",
    )

    run_button = st.button("Run workflow", type="primary")

    if run_button:
        if not input_path_str.strip():
            st.error("Please provide a valid file path or folder path.")
            return

        try:
            pdf_files = resolve_input_pdfs(input_path_str.strip())
        except Exception as e:
            st.error(str(e))
            return

        output_base_dir = Path(output_dir_str)

        st.success(f"Found {len(pdf_files)} PDF file(s) to process.")

        batch_result = process_multiple_pdfs(
            pdf_files=pdf_files,
            source=source,
            output_base_dir=output_base_dir,
        )

        items = batch_result["items"]
        summary_json = batch_result["summary_json"]

        st.header("Run Summary")
        st.write(f"**Batch summary JSON:** `{summary_json}`")
        render_summary_cards(items)

        st.header("Detailed Results")
        for item in items:
            render_detailed_result(item)


if __name__ == "__main__":
    main()