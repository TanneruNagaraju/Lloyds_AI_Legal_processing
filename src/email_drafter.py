from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import smtplib
from email.message import EmailMessage


from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Pydantic schema
# -----------------------------------------------------------------------------
class EmailDraft(BaseModel):
    to: Optional[str] = Field(
        default=None,
        description="Primary recipient email address if available."
    )
    subject: str = Field(
        ...,
        description="Email subject line."
    )
    body: str = Field(
        ...,
        description="Plain-text email body."
    )


# -----------------------------------------------------------------------------
# Prompt
# -----------------------------------------------------------------------------
def build_email_prompt() -> ChatPromptTemplate:
    system_message = """
You are an assistant drafting a professional internal legal/claims email.

You will receive structured metadata extracted from a document. Draft a concise, professional email summarizing the document for a legal or claims handling team.

Rules:
- The email must always begin with a formal salutation, such as "Dear Team," or another appropriate formal opening.
- Use only the provided metadata.
- Do not invent missing facts.
- If the recipient is missing, set `to` to null.
- Mention warnings or missing fields when relevant.
- The email should be clear, professional, and practical.
- The body should be plain text, suitable for internal use.
- End the email with: Regards,\nTanneru
"""

    human_message = """
File name:
{file_name}

Extraction result:
{metadata}

Draft the email.
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

@traceable(name="get_email_llm")
def get_email_llm(model_name: str = "gpt-4.1-mini", temperature: float = 0.0):
    logger.info("Initializing email drafting LLM | model=%s", model_name)
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
    )


# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------
@traceable(name="generate_email_draft")
def generate_email_draft(
    extraction_result: Dict[str, Any],
    model_name: str = "gpt-4.1-mini",
) -> Dict[str, Any]:
    file_name = extraction_result.get("file_name", "unknown.pdf")
    metadata = extraction_result.get("metadata")

    logger.info("Starting email draft generation | file=%s", file_name)

    if extraction_result.get("extraction_status") != "success" or not metadata:
        logger.warning("Skipping email draft because extraction failed | file=%s", file_name)
        return {
            "file_name": file_name,
            "email_status": "failed",
            "email_draft": None,
            "errors": ["Extraction result is not available for email drafting."],
        }

    try:
        llm = get_email_llm(model_name=model_name, temperature=0.0)
        prompt = build_email_prompt()
        structured_llm = llm.with_structured_output(EmailDraft)

        chain = prompt | structured_llm

        logger.info("Invoking email drafting LLM | file=%s", file_name)

        email_draft: EmailDraft = chain.invoke(
            {
                "file_name": file_name,
                "metadata": metadata,
            }
        )

        logger.info("Email draft created | file=%s | subject=%s", file_name, email_draft.subject)

        return {
            "file_name": file_name,
            "email_status": "success",
            "email_draft": email_draft.model_dump(),
            "errors": [],
        }

    except Exception as e:
        logger.exception("Email drafting failed | file=%s | error=%s", file_name, str(e))
        return {
            "file_name": file_name,
            "email_status": "failed",
            "email_draft": None,
            "errors": [f"Email drafting failed: {str(e)}"],
        }