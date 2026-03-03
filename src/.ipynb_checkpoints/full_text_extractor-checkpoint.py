from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Pydantic schema for extracted metadata
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
# Prompt builder
# -----------------------------------------------------------------------------
def build_extraction_prompt() -> ChatPromptTemplate:
    system_message = """
You are a legal document metadata extraction agent.

You will receive text extracted from a PDF. Extract the key metadata fields and return structured output.

Required behavior:
- Use only the document text provided.
- Do not invent values.
- If a field is not present, unclear, or explicitly not applicable, return null for that field.
- Add the field name to missing_fields if it cannot be extracted.
- If multiple conflicting values appear, choose the most likely one only if strongly supported, and explain the ambiguity in warnings.
- Normalize the date_of_loss to YYYY-MM-DD when possible.
- Keep document_type concise and normalized, such as:
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

Document text:
\"\"\"
{text}
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
    logger.info("Initializing extraction LLM | model=%s", model_name)
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
    )


# -----------------------------------------------------------------------------
# Main extraction function
# -----------------------------------------------------------------------------
@traceable(name="extract_metadata_full_text")   
def extract_metadata_full_text(
    processed_pdf: Dict[str, Any],
    triage_result: Dict[str, Any],
    model_name: str = "gpt-4.1-mini",
) -> Dict[str, Any]:
    file_name = processed_pdf.get("file_name", "unknown.pdf")
    full_text = processed_pdf.get("full_text", "") or ""

    logger.info("Starting full-text metadata extraction | file=%s", file_name)

    if not full_text.strip():
        logger.warning("Extraction skipped because text is empty | file=%s", file_name)
        return {
                "file_name": file_name,
                "extraction_status": "failed",
                "extraction_mode": "full_text",
                "metadata": None,
                "retrieved_evidence": None,
                "errors": ["No extracted text available for metadata extraction."],
            }

    try:
        llm = get_extraction_llm(model_name=model_name, temperature=0.0)
        prompt = build_extraction_prompt()
        structured_llm = llm.with_structured_output(ExtractedMetadata)

        chain = prompt | structured_llm

        logger.info(
            "Invoking extraction LLM | file=%s | chars=%s",
            file_name,
            len(full_text),
        )

        metadata: ExtractedMetadata = chain.invoke(
            {
                "file_name": file_name,
                "triage_decision": triage_result.get("triage_decision", {}),
                "text": full_text,
            }
        )

        logger.info(
            "Extraction complete | file=%s | doc_type=%s | policy=%s",
            file_name,
            metadata.document_type,
            metadata.policy_number,
        )

        if metadata.warnings:
            for warning in metadata.warnings:
                logger.warning("Extraction warning | file=%s | warning=%s", file_name, warning)

        return {
                "file_name": file_name,
                "extraction_status": "success",
                "extraction_mode": "full_text",
                "metadata": metadata.model_dump(),
                "retrieved_evidence": None,
                "errors": [],
            }

    except Exception as e:
        logger.exception("Full-text extraction failed | file=%s | error=%s", file_name, str(e))
        return {
                "file_name": file_name,
                "extraction_status": "failed",
                "extraction_mode": "full_text",
                "metadata": None,
                "retrieved_evidence": None,
                "errors": [f"Full-text extraction failed: {str(e)}"],
            }