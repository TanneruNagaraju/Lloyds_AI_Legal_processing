from __future__ import annotations

import os
import logging
from typing import Any, Dict, Optional, TypedDict

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from IPython.display import Image, display


from src.pdf_processor import process_pdf
from src.triage_router import triage_document_with_llm, get_route_from_triage
from src.full_text_extractor import extract_metadata_full_text
from src.rag_fallback_extractor import extract_metadata_with_rag_fallback
from src.mongodb_store import save_processing_result
from src.email_drafter import generate_email_draft
from src.email_sender import send_email

from dotenv import load_dotenv
load_dotenv()


logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Graph state
# -----------------------------------------------------------------------------
class WorkflowState(TypedDict, total=False):
    pdf_path: str
    source: str

    processed_pdf: Dict[str, Any]
    triage: Dict[str, Any]
    extraction: Dict[str, Any]
    storage: Dict[str, Any]
    email: Dict[str, Any]

    final_status: str
    final_message: str


# -----------------------------------------------------------------------------
# Node 1: PDF processing
# -----------------------------------------------------------------------------
def pdf_processing_node(state: WorkflowState) -> WorkflowState:
    pdf_path = state["pdf_path"]
    logger.info("Workflow node: pdf_processing | pdf_path=%s", pdf_path)

    processed_pdf = process_pdf(pdf_path)

    return {
        **state,
        "processed_pdf": processed_pdf,
    }


# -----------------------------------------------------------------------------
# Node 2: LLM triage / routing
# -----------------------------------------------------------------------------
def triage_node(state: WorkflowState) -> WorkflowState:
    logger.info("Workflow node: triage")

    processed_pdf = state["processed_pdf"]

    triage_result = triage_document_with_llm(
        processed_pdf=processed_pdf,
        model_name="gpt-4.1-mini",
    )

    return {
        **state,
        "triage": triage_result,
    }


# -----------------------------------------------------------------------------
# Conditional router for LangGraph
# -----------------------------------------------------------------------------
def route_after_triage(state: WorkflowState) -> str:
    logger.info("Workflow router: route_after_triage")

    triage = state.get("triage", {})
    triage_status = triage.get("triage_status")

    if triage_status != "success":
        logger.warning("Triage failed; routing to manual_review")
        return "manual_review"

    route = triage.get("triage_decision", {}).get("route", "manual_review")
    logger.info("Route selected by triage: %s", route)
    return route


# -----------------------------------------------------------------------------
# Node 3A: Full-text extraction
# -----------------------------------------------------------------------------
def full_text_extraction_node(state: WorkflowState) -> WorkflowState:
    logger.info("Workflow node: full_text_extraction")

    processed_pdf = state["processed_pdf"]
    triage_result = state["triage"]

    extraction_result = extract_metadata_full_text(
        processed_pdf=processed_pdf,
        triage_result=triage_result,
        model_name="gpt-4.1-mini",
    )

    return {
        **state,
        "extraction": extraction_result,
    }


# -----------------------------------------------------------------------------
# Node 3B: RAG fallback extraction
# -----------------------------------------------------------------------------
def rag_fallback_node(state: WorkflowState) -> WorkflowState:
    logger.info("Workflow node: rag_fallback")

    processed_pdf = state["processed_pdf"]
    triage_result = state["triage"]

    rewrite_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    extraction_result = extract_metadata_with_rag_fallback(
        processed_pdf=processed_pdf,
        triage_result=triage_result,
        llm_for_rewrite=rewrite_llm,
        embedding_model="text-embedding-3-small",
        collection="legal_doc_rag",
        extraction_model_name="gpt-4.1-mini",
        source=state.get("source") or processed_pdf.get("file_name"),
    )

    return {
        **state,
        "extraction": extraction_result,
    }


# -----------------------------------------------------------------------------
# Node 3C: Manual review
# -----------------------------------------------------------------------------
def manual_review_node(state: WorkflowState) -> WorkflowState:
    logger.info("Workflow node: manual_review")

    triage = state.get("triage", {})
    triage_decision = triage.get("triage_decision", {})

    extraction_result = {
        "file_name": state.get("processed_pdf", {}).get("file_name"),
        "extraction_status": "failed",
        "extraction_mode": "manual_review",
        "metadata": None,
        "retrieved_evidence": None,
        "errors": [
            "Document routed to manual review.",
            triage_decision.get("reason", "Triage was unsuccessful or document quality was insufficient."),
        ],
    }

    return {
        **state,
        "extraction": extraction_result,
        "final_status": "manual_review",
        "final_message": "Document requires manual review.",
    }


# -----------------------------------------------------------------------------
# Node 3D: Reject
# -----------------------------------------------------------------------------
def reject_node(state: WorkflowState) -> WorkflowState:
    logger.info("Workflow node: reject")

    triage = state.get("triage", {})
    triage_decision = triage.get("triage_decision", {})

    extraction_result = {
        "file_name": state.get("processed_pdf", {}).get("file_name"),
        "extraction_status": "failed",
        "extraction_mode": "reject",
        "metadata": None,
        "retrieved_evidence": None,
        "errors": [
            "Document rejected as irrelevant or out of scope.",
            triage_decision.get("reason", "Document is not relevant to the workflow."),
        ],
    }

    return {
        **state,
        "extraction": extraction_result,
        "final_status": "rejected",
        "final_message": "Document was rejected as out of scope.",
    }


# -----------------------------------------------------------------------------
# Node 4: MongoDB storage
# -----------------------------------------------------------------------------
def storage_node(state: WorkflowState) -> WorkflowState:
    logger.info("Workflow node: storage")

    processed_pdf = state.get("processed_pdf", {})
    triage_result = state.get("triage", {})
    extraction_result = state.get("extraction", {})

    storage_result = save_processing_result(
        processed_pdf=processed_pdf,
        triage_result=triage_result,
        extraction_result=extraction_result,
        source=state.get("source", "uploaded_pdf"),
    )

    return {
        **state,
        "storage": storage_result,
    }


# -----------------------------------------------------------------------------
# Node 5: Email draft generation
# -----------------------------------------------------------------------------
def email_draft_node(state: WorkflowState) -> WorkflowState:
    logger.info("Workflow node: email_draft")

    extraction_result = state.get("extraction", {})

    if extraction_result.get("extraction_status") != "success":
        logger.info("Skipping email draft because extraction was not successful")

        return {
            **state,
            "email": {
                "file_name": extraction_result.get("file_name"),
                "email_status": "skipped",
                "email_draft": None,
                "send_result": None,
                "errors": ["Email draft skipped because extraction was not successful."],
            },
        }

    email_result = generate_email_draft(
        extraction_result=extraction_result,
        model_name="gpt-5-mini",
    )

    return {
        **state,
        "email": {
            **email_result,
            "send_result": None,
        },
    }



def email_send_node(state: WorkflowState) -> WorkflowState:
    logger.info("Workflow node: email_send")

    email_state = state.get("email", {})
    email_draft = email_state.get("email_draft")

    if email_state.get("email_status") != "success" or not email_draft:
        logger.info("Skipping email sending because no successful email draft is available")

        updated_email_state = {
            **email_state,
            "send_result": {
                "send_status": "skipped",
                "error": "Email sending skipped because no successful draft is available.",
            },
        }

        return {
            **state,
            "email": updated_email_state,
            "final_status": "partial_success",
            "final_message": "Workflow completed, but email was not sent.",
        }

    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    default_receiver_email = os.getenv("RECEIVER_EMAIL")
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "465"))


    if not sender_email or not sender_password:
        logger.warning("Email sending skipped because sender credentials are missing")

        updated_email_state = {
            **email_state,
            "send_result": {
                "send_status": "skipped",
                "error": "Sender credentials are missing in environment variables.",
            },
        }

        return {
            **state,
            "email": updated_email_state,
            "final_status": "partial_success",
            "final_message": "Workflow completed, but email credentials are missing.",
        }

    if not default_receiver_email:
        logger.warning("Email sending skipped because no recipient email is available")

        updated_email_state = {
            **email_state,
            "send_result": {
                "send_status": "skipped",
                "error": "No recipient email address available.",
            },
        }

        return {
            **state,
            "email": updated_email_state,
            "final_status": "partial_success",
            "final_message": "Workflow completed, but no recipient email was available.",
        }

    send_result = send_email(
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        smtp_username=sender_email,
        smtp_password=sender_password,
        sender_email=sender_email,
        to_email=default_receiver_email,
        subject=email_draft["subject"],
        body=email_draft["body"],
    )

    updated_email_state = {
        **email_state,
        "send_result": send_result,
    }

    final_status = "success" if send_result.get("send_status") == "success" else "partial_success"
    final_message = (
        "Workflow completed and email was sent successfully."
        if send_result.get("send_status") == "success"
        else "Workflow completed, but email sending failed."
    )

    return {
        **state,
        "email": updated_email_state,
        "final_status": final_status,
        "final_message": final_message,
    }

# -----------------------------------------------------------------------------
# Build graph
# -----------------------------------------------------------------------------
def build_workflow_graph():
    logger.info("Building LangGraph workflow")

    workflow = StateGraph(WorkflowState)

    workflow.add_node("pdf_processing", pdf_processing_node)
    workflow.add_node("triage", triage_node)
    workflow.add_node("full_text_extraction", full_text_extraction_node)
    workflow.add_node("rag_fallback", rag_fallback_node)
    workflow.add_node("manual_review", manual_review_node)
    workflow.add_node("reject", reject_node)
    workflow.add_node("storage", storage_node)
    workflow.add_node("email_draft", email_draft_node)
    workflow.add_node("email_send", email_send_node)

    workflow.set_entry_point("pdf_processing")

    workflow.add_edge("pdf_processing", "triage")

    workflow.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "full_text": "full_text_extraction",
            "rag_fallback": "rag_fallback",
            "manual_review": "manual_review",
            "reject": "reject",
        },
    )

    workflow.add_edge("full_text_extraction", "storage")
    workflow.add_edge("rag_fallback", "storage")
    workflow.add_edge("manual_review", "storage")
    workflow.add_edge("reject", "storage")

    workflow.add_edge("storage", "email_draft")
    workflow.add_edge("email_draft", "email_send")
    workflow.add_edge("email_send", END)

    return workflow.compile()


# -----------------------------------------------------------------------------
# One easy runner
# -----------------------------------------------------------------------------
def run_workflow(
    pdf_path: str,
    source: str = "uploaded_pdf",
) -> Dict[str, Any]:
    logger.info("Running full workflow | pdf_path=%s", pdf_path)

    graph = build_workflow_graph()

    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))

    initial_state: WorkflowState = {
        "pdf_path": pdf_path,
        "source": source,
    }

    result = graph.invoke(initial_state)
    logger.info("Workflow finished | final_status=%s", result.get("final_status"))

    return result