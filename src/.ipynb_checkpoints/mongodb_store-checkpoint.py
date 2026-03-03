from __future__ import annotations

import os
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def get_mongo_collection(
    uri: Optional[str] = None,
    db_name: str = "legal_doc_pipeline",
    collection_name: str = "documents",
) -> Collection:
    """
    Create and return a MongoDB collection handle.
    """
    mongo_uri = uri or os.getenv("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MongoDB URI is missing. Set MONGODB_URI in environment variables.")

    logger.info("Connecting to MongoDB | db=%s | collection=%s", db_name, collection_name)

    client = MongoClient(mongo_uri)
    db = client[db_name]
    return db[collection_name]


def build_storage_record(
    processed_pdf: Dict[str, Any],
    triage_result: Dict[str, Any],
    extraction_result: Dict[str, Any],
    source: str = "uploaded_pdf",
) -> Dict[str, Any]:
    """
    Build a normalized MongoDB document for one processed file.
    """
    now_utc = datetime.now(timezone.utc).isoformat()

    record = {
        "file_name": processed_pdf.get("file_name"),
        "file_path": processed_pdf.get("file_path"),
        "source": source,
        "processed_at": now_utc,
        "pdf_processing": {
            "extraction_status": processed_pdf.get("extraction_status"),
            "num_pages": processed_pdf.get("num_pages"),
            "warnings": processed_pdf.get("warnings", []),
            "errors": processed_pdf.get("errors", []),
            "text_quality": processed_pdf.get("text_quality", {}),
        },
        "triage": triage_result,
        "extraction": extraction_result,
    }

    return record


def save_processing_result(
    processed_pdf: Dict[str, Any],
    triage_result: Dict[str, Any],
    extraction_result: Dict[str, Any],
    *,
    uri: Optional[str] = None,
    db_name: str = "legal_doc_pipeline",
    collection_name: str = "documents",
    source: str = "uploaded_pdf",
) -> Dict[str, Any]:
    """
    Save one complete pipeline result into MongoDB.
    """
    try:
        collection = get_mongo_collection(
            uri=uri,
            db_name=db_name,
            collection_name=collection_name,
        )

        record = build_storage_record(
            processed_pdf=processed_pdf,
            triage_result=triage_result,
            extraction_result=extraction_result,
            source=source,
        )

        logger.info(
            "Saving processing result to MongoDB | file=%s",
            record.get("file_name"),
        )

        result = collection.insert_one(record)

        logger.info(
            "MongoDB insert successful | file=%s | inserted_id=%s",
            record.get("file_name"),
            str(result.inserted_id),
        )

        return {
            "storage_status": "success",
            "inserted_id": str(result.inserted_id),
            "record": record,
            "errors": [],
        }

    except PyMongoError as e:
        logger.exception("MongoDB save failed | error=%s", str(e))
        return {
            "storage_status": "failed",
            "inserted_id": None,
            "record": None,
            "errors": [f"MongoDB save failed: {str(e)}"],
        }

    except Exception as e:
        logger.exception("Unexpected storage error | error=%s", str(e))
        return {
            "storage_status": "failed",
            "inserted_id": None,
            "record": None,
            "errors": [f"Unexpected storage error: {str(e)}"],
        }