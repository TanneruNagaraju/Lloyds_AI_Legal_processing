from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


from src.rag import run_rag_fields

from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Shared schema: same output shape as full-text extraction
# -----------------------------------------------------------------------------
class ExtractedMetadata(BaseModel):
    is_relevant: bool = Field(
        ...,
        description="Whether the document is relevant to the legal/claims workflow."
    )
    document_type: str = Field(
        ...,
        description="Normalized document type."
    )
    date_of_loss: Optional[str] = Field(
        default=None,
        description="Date of loss or incident in ISO format YYYY-MM-DD when available."
    )
    policy_number: Optional[str] = Field(
        default=None,
        description="Policy number if present."
    )
    recipient: Optional[str] = Field(
        default=None,
        description="Recipient of the correspondence or intended recipient."
    )
    claimant: Optional[str] = Field(
        default=None,
        description="Claimant name if present."
    )
    defendant: Optional[str] = Field(
        default=None,
        description="Defendant or responding party if present."
    )
    case_court_reference_number: Optional[str] = Field(
        default=None,
        description="Court, case, or regulatory reference number if present."
    )
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Fields that could not be found or are not provided."
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Important ambiguities, inconsistencies, or caveats."
    )
    summary: str = Field(
        ...,
        description="Brief summary of the document."
    )

    @field_validator("document_type")
    @classmethod
    def normalize_document_type(cls, value: str) -> str:
        return value.strip().lower().replace(" ", "_")


# -----------------------------------------------------------------------------
# Prompt for extraction from retrieved evidence
# -----------------------------------------------------------------------------
def build_rag_extraction_prompt() -> ChatPromptTemplate:
    system_message = """
You are a legal document metadata extraction agent.

You will receive retrieved evidence chunks from a legal/claims document. The evidence was selected using field-specific retrieval queries. Your task is to extract the required metadata only from the provided evidence.

Rules:
- Use only the evidence provided.
- Do not invent values.
- If a field is not present, unclear, or explicitly not applicable, return null for that field.
- Add field names to missing_fields when they cannot be extracted.
- If multiple conflicting values appear, choose the most likely one only if strongly supported, and mention the conflict in warnings.
- Normalize date_of_loss to YYYY-MM-DD when possible.
- document_type should be concise and normalized, such as:
  settlement_offer, court_scheduling_order, first_notice_of_loss,
  coverage_position_letter, regulatory_inquiry, legal_correspondence,
  irrelevant_brochure, unknown

Extract these fields:
- is_relevant
- document_type
- date_of_loss
- policy_number
- recipient
- claimant
- defendant
- case_court_reference_number
- missing_fields
- warnings
- summary
"""

    human_message = """
File name:
{file_name}

Triage decision:
{triage_decision}

Retrieved evidence:
\"\"\"
{evidence}
\"\"\"

Return the structured extraction output.
"""
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_message.strip()),
            ("human", human_message.strip()),
        ]
    )


# -----------------------------------------------------------------------------
# LLM factory
# -----------------------------------------------------------------------------

@traceable(name="get_extraction_llm")
def get_extraction_llm(model_name: str = "gpt-4.1-mini", temperature: float = 0.0):
    logger.info("Initializing RAG extraction LLM | model=%s", model_name)
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
    )


# -----------------------------------------------------------------------------
# Main RAG fallback extraction function
# -----------------------------------------------------------------------------

@traceable(name="extract_metadata_with_rag_fallback")
def extract_metadata_with_rag_fallback(
    processed_pdf: Dict[str, Any],
    triage_result: Dict[str, Any],
    *,
    llm_for_rewrite: Optional[Any] = None,
    embedding_model: str = "text-embedding-3-small",
    collection: str = "legal_doc_rag",
    extraction_model_name: str = "gpt-4.1-mini",
    source: Optional[str] = None,
    parent_chunk_size: int = 300,
    parent_chunk_overlap: int = 30,
    child_chunk_size: int = 50,
    child_chunk_overlap: int = 8,
    child_k: int = 10,
    rerank_top_n: int = 8,
    parent_top_k: int = 5,
    field_top_n_chunks: int = 5,
) -> Dict[str, Any]:
    """
    RAG fallback metadata extraction.

    Steps:
    1. Run field-specific retrieval using your existing RAG pipeline
    2. Feed merged retrieved evidence to the LLM
    3. Return the same schema as full-text extraction
    """
    file_name = processed_pdf.get("file_name", "unknown.pdf")
    full_text = (processed_pdf.get("full_text") or "").strip()

    logger.info("Starting RAG fallback extraction | file=%s", file_name)

    if not full_text:
        logger.warning("RAG extraction skipped because text is empty | file=%s", file_name)
        return {
            "file_name": file_name,
            "extraction_status": "failed",
            "extraction_mode": "rag_fallback",
            "metadata": None,
            "retrieved_evidence": None,
            "errors": ["No extracted text available for RAG fallback extraction."],
        }

    try:
        logger.info("Running field-specific RAG retrieval | file=%s", file_name)

        evidence = run_rag_fields(
            text=full_text,
            collection=collection,
            embedding_model=embedding_model,
            llm=llm_for_rewrite,  # can be None if you don't want paraphrase rewrite
            source=source or file_name,
            parent_chunk_size=parent_chunk_size,
            parent_chunk_overlap=parent_chunk_overlap,
            child_chunk_size=child_chunk_size,
            child_chunk_overlap=child_chunk_overlap,
            child_k=child_k,
            rerank_top_n=rerank_top_n,
            parent_top_k=parent_top_k,
            field_top_n_chunks=field_top_n_chunks,
        )

        if not evidence or not evidence.strip():
            logger.warning("RAG retrieval returned no evidence | file=%s", file_name)
            return {
                "file_name": file_name,
                "extraction_status": "failed",
                "extraction_mode": "rag_fallback",
                "metadata": None,
                "retrieved_evidence": "",
                "errors": ["RAG retrieval returned no evidence."],
            }

        logger.info(
            "RAG evidence retrieved | file=%s | evidence_chars=%s",
            file_name,
            len(evidence),
        )

        llm = get_extraction_llm(model_name=extraction_model_name, temperature=0.0)
        prompt = build_rag_extraction_prompt()
        structured_llm = llm.with_structured_output(ExtractedMetadata)

        chain = prompt | structured_llm

        logger.info("Invoking LLM on retrieved evidence | file=%s", file_name)

        metadata: ExtractedMetadata = chain.invoke(
            {
                "file_name": file_name,
                "triage_decision": triage_result.get("triage_decision", {}),
                "evidence": evidence,
            }
        )

        logger.info(
            "RAG extraction complete | file=%s | doc_type=%s | policy=%s",
            file_name,
            metadata.document_type,
            metadata.policy_number,
        )

        if metadata.warnings:
            for warning in metadata.warnings:
                logger.warning("RAG extraction warning | file=%s | warning=%s", file_name, warning)

        return {
            "file_name": file_name,
            "extraction_status": "success",
            "extraction_mode": "rag_fallback",
            "metadata": metadata.model_dump(),
            "retrieved_evidence": evidence,
            "errors": [],
        }

    except Exception as e:
        logger.exception("RAG fallback extraction failed | file=%s | error=%s", file_name, str(e))
        return {
            "file_name": file_name,
            "extraction_status": "failed",
            "extraction_mode": "rag_fallback",
            "metadata": None,
            "retrieved_evidence": None,
            "errors": [f"RAG fallback extraction failed: {str(e)}"],
        }