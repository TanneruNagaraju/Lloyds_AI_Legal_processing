import logging
from src.workflow import run_workflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

pdf_path = r"C:\Users\tanne\UK_interviews\Lloyds\Lloyd’s of London - AI Engineer - Technical Assignment\Mock PDF Documents\1 - Regulatory Inquiry Notice – Zenith Plastics Ltd.pdf"

result = run_workflow(pdf_path=pdf_path, source="uploaded_pdf")

print(result.keys())
print(result["final_status"])
print(result["final_message"])
print(result["triage"])
print(result["extraction"])
print(result["email"])