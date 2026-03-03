# AI-Assisted Legal Document Processing (Service of Suits PoC)

This repository contains an end-to-end proof of concept for processing legal documents (mock PDFs) using an AI-assisted workflow with LLM triage, optional RAG fallback, structured extraction, storage, tracing, and email notification.

## What it does

Given a **single PDF** or a **folder of PDFs**, the pipeline:

* extracts text from PDFs (open-source PDF parsing)
* performs **LLM-based triage + routing** (full-text vs RAG fallback vs manual review vs reject)
* extracts structured metadata:

  * Document Type
  * Date of Loss
  * Policy Number
  * Recipient
  * Claimant / Defendant
  * Case / Court Reference Number
* stores results in **MongoDB** and saves **one JSON output per document**
* generates an email draft and can optionally **send** it
* traces runs with **LangSmith** for observability

---

## Project structure

```
.
├── app.py                # Streamlit UI (path-based: file or folder)
├── main.py               # Batch runner (process a folder, per-file JSON outputs)
├── src/
│   ├── pdf_processor.py
│   ├── triage_router.py
│   ├── full_text_extractor.py
│   ├── rag.py        #  RAG code (Qdrant hybrid + rerank + parent/child)
│   ├── rag_fallback_extractor.py
│   ├── mongodb_store.py
│   ├── email_drafter.py
│   ├── email_sender.py
│   └── workflow.py            # LangGraph workflow orchestration
└── outputs/
    ├── results/               # one JSON per PDF
    ├── reports/               # batch summary JSON
    └── logs/                  # pipeline.log
```

---

## Prerequisites

* Python **3.10+** recommended
* An **OpenAI API key**
* **LangSmith** API key for tracing
* **MongoDB** (Atlas or local)
* SMTP credentials if you want to actually send emails

---

## Installation

### 1) Create and activate a virtual environment

**Windows (PowerShell)**

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt` yet, you can start with:

```bash
pip install streamlit python-dotenv pypdf pymongo \
  langchain langchain-openai langgraph langsmith \
  qdrant-client langchain-qdrant torch transformers huggingface_hub
```

---

## Environment setup

Create a `.env` file in the project root.

### Minimum required

```bash
OPENAI_API_KEY=your_openai_key
```

### Recommended (LangSmith tracing)

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=legal-doc-intake
LANGCHAIN_API_KEY=your_langsmith_key
```

### MongoDB (for storage)

```bash
MONGODB_URI=mongodb+srv://<user>:<pass>@<cluster>/?retryWrites=true&w=majority
```

### Hugging Face token (optional, only if needed in your environment)

```bash
HF_TOKEN=your_hf_token
```

### Email sending (optional)

> **Important:** Use a Gmail **App Password** (not your normal password).

```bash
SENDER_EMAIL=your_sender@gmail.com
SENDER_PASSWORD=your_gmail_app_password
DEFAULT_RECEIVER_EMAIL=your_test_inbox@gmail.com
SMTP_HOST=smtp.gmail.com
SMTP_PORT=465
```

---

## Running the pipeline

### Option A — Streamlit UI (single file path or folder path)

```bash
streamlit run app.py
```

In the UI, paste either:

* a **single PDF path**, or
* a **folder path** containing multiple PDFs

Outputs will be saved to:

* `outputs/results/<one-json-per-pdf>.json`
* `outputs/reports/batch_results_summary_<timestamp>.json`

---

### Option B — Batch runner (folder)

Edit `input_dir` inside `main.py` (or pass your folder path if you added CLI args), then run:

```bash
python main.py
```

---

### Option C — Run one file in Python

```python
import logging
from src.workflow import run_workflow

logging.basicConfig(level=logging.INFO)

result = run_workflow(
    pdf_path=r"C:\path\to\file.pdf",
    source="uploaded_pdf"
)
print(result["final_status"])
print(result["extraction"])
print(result["email"])
```

---

## Outputs

For each PDF, the pipeline writes a separate JSON file containing:

* summary (route, doc type, key fields)
* full workflow result (triage, extraction, storage, email)
* timestamps and file metadata

Example:

```
outputs/results/2 - Settlement Offer – Harper v CityTaxi Ltd.json
```

---

## Troubleshooting

### LangSmith project not showing

Make sure:

* `LANGCHAIN_TRACING_V2=true`
* `LANGCHAIN_API_KEY` is set
* the workflow actually invokes an LLM call
* in notebooks, restart the kernel and set env vars **before** imports

### JSON save error (ObjectId not serializable)

The Streamlit and batch runner include `make_json_safe()` to convert Mongo `ObjectId` values to strings before saving.


