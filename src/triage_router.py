from __future__ import annotations

import logging
from typing import Any, Dict, Literal, Optional
import os
from pydantic import BaseModel, Field, field_validator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Pydantic schema for LLM triage output
# -----------------------------------------------------------------------------
class TriageDecision(BaseModel):
    is_relevant: bool = Field(
        ...,
        description="Whether the document is relevant to the legal/claims workflow."
    )
    document_type: str = Field(
        ...,
        description="High-level document type, e.g. settlement_offer, court_order, fnol, coverage_letter, regulatory_inquiry, irrelevant_brochure."
    )
    route: Literal["full_text", "rag_fallback", "manual_review", "reject"] = Field(
        ...,
        description="Routing decision for downstream workflow."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1."
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Problems, risks, ambiguities, or warnings found in the document."
    )
    reason: str = Field(
        ...,
        description="Short explanation for the chosen routing decision."
    )

    @field_validator("document_type")
    @classmethod
    def normalize_document_type(cls, value: str) -> str:
        return value.strip().lower().replace(" ", "_")


# -----------------------------------------------------------------------------
# Prompt builder
# -----------------------------------------------------------------------------
def build_triage_prompt() -> ChatPromptTemplate:
    """
    Prompt for LLM-based document triage.
    """
    system_message = """
You are an intake triage agent for an AI-assisted legal document processing workflow.

You will receive extracted text from a PDF document. Your task is to:
1. Decide whether the document is relevant to a legal, claims, regulatory, coverage, or litigation workflow.
2. Classify the document type.
3. Assess whether the extracted text appears sufficiently usable.
4. Choose the best route for downstream processing.

You must return structured output with these fields:
- is_relevant: boolean
- document_type: string
- route: one of ["full_text", "rag_fallback", "manual_review", "reject"]
- confidence: number between 0 and 1
- issues: array of strings
- reason: short explanation

Routing guidance:
- full_text:
  Use when the document is relevant and the extracted text appears suitable for direct metadata extraction.
- rag_fallback:
  Use when the document is relevant, but long, noisy, partially inconsistent, or difficult enough that retrieval-assisted extraction would be safer.
- manual_review:
  Use when the document may be relevant but the extracted text is too poor, too ambiguous, or too incomplete for reliable automated extraction.
- reject:
  Use when the document is clearly outside the legal/claims workflow.

Important rules:
- Do not invent facts.
- If the document is not a court case, that is acceptable; it may still be relevant.
- If fields such as date of loss or case reference are absent or marked not applicable, mention that in issues if relevant.
- If the text contains corruption, duplication, OCR-like noise, broken formatting, or inconsistent values, reflect that in issues and routing.
- Be conservative when confidence is low.
"""

    human_message = """
Document file name:
{file_name}

Number of pages:
{num_pages}

Extracted text:
\"\"\"
{text}
\"\"\"

Return the structured triage decision.
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
@traceable(name="get_triage_llm")
def get_triage_llm(model_name: str = "gpt-4.1-mini", temperature: float = 0.0):
    """
    Create the LLM client for triage.
    """
    logger.info("Initializing triage LLM | model=%s", model_name)
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
    )


# -----------------------------------------------------------------------------
# Main triage function
# -----------------------------------------------------------------------------
@traceable(name="triage_document_with_llm")
def triage_document_with_llm(
    processed_pdf: Dict[str, Any],
    model_name: str = "gpt-4.1-mini",
) -> Dict[str, Any]:
    """
    Run LLM-based validation + routing on extracted PDF text.

    Expected input:
    {
        "file_name": "...",
        "num_pages": 2,
        "full_text": "...",
        "extraction_status": "success",
        ...
    }
    """
    file_name = processed_pdf.get("file_name", "unknown.pdf")
    extraction_status = processed_pdf.get("extraction_status", "failed")
    full_text = processed_pdf.get("full_text", "") or ""
    num_pages = processed_pdf.get("num_pages", 0)

    logger.info("Starting LLM triage for file: %s", file_name)

    # Minimal deterministic sanity guard
    if extraction_status != "success":
        logger.error("Skipping triage because extraction failed | file=%s", file_name)
        return {
            "file_name": file_name,
            "triage_status": "failed",
            "triage_decision": None,
            "errors": ["PDF extraction failed before triage."],
        }

    if not full_text.strip():
        logger.warning("Skipping triage because extracted text is empty | file=%s", file_name)
        return {
            "file_name": file_name,
            "triage_status": "failed",
            "triage_decision": None,
            "errors": ["Extracted text is empty."],
        }

    try:
        llm = get_triage_llm(model_name=model_name, temperature=0.0)
        prompt = build_triage_prompt()

        structured_llm = llm.with_structured_output(TriageDecision)

        chain = prompt | structured_llm

        logger.info(
            "Invoking triage LLM | file=%s | pages=%s | chars=%s",
            file_name,
            num_pages,
            len(full_text),
        )

        decision: TriageDecision = chain.invoke(
            {
                "file_name": file_name,
                "num_pages": num_pages,
                "text": full_text,
            }
        )

        logger.info(
            "Triage complete | file=%s | route=%s | relevant=%s | confidence=%.2f",
            file_name,
            decision.route,
            decision.is_relevant,
            decision.confidence,
        )

        if decision.issues:
            for issue in decision.issues:
                logger.warning("Triage issue | file=%s | issue=%s", file_name, issue)

        return {
            "file_name": file_name,
            "triage_status": "success",
            "triage_decision": decision.model_dump(),
            "errors": [],
        }

    except Exception as e:
        logger.exception("LLM triage failed | file=%s | error=%s", file_name, str(e))
        return {
            "file_name": file_name,
            "triage_status": "failed",
            "triage_decision": None,
            "errors": [f"LLM triage failed: {str(e)}"],
        }


# -----------------------------------------------------------------------------
# Routing helper for LangGraph
# -----------------------------------------------------------------------------
def get_route_from_triage(state: Dict[str, Any]) -> str:
    """
    LangGraph-compatible router function.

    Expects:
    state["triage"]["triage_decision"]["route"]
    """
    triage = state.get("triage", {})
    triage_status = triage.get("triage_status")

    if triage_status != "success":
        logger.warning("Routing to manual_review because triage failed.")
        return "manual_review"

    route = triage.get("triage_decision", {}).get("route", "manual_review")
    logger.info("Resolved graph route: %s", route)
    return route