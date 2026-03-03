import smtplib
import ssl
import logging
from email.message import EmailMessage
from typing import Dict

logger = logging.getLogger(__name__)


def send_email(
    smtp_host: str,
    smtp_port: int,
    smtp_username: str,
    smtp_password: str,
    sender_email: str,
    to_email: str,
    subject: str,
    body: str,
) -> Dict[str, str]:
    try:
        msg = EmailMessage()
        msg["From"] = sender_email
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.set_content(body)


        with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
            server.login(smtp_username, smtp_password)
            server.send_message(msg)

        logger.info("Email sent successfully | to=%s | subject=%s", to_email, subject)

        return {
            "send_status": "success",
            "msg": "Email sent successfully!"
        }

    except Exception as e:
        logger.exception("Email sending failed | error=%s", str(e))
        return {
            "send_status": "failed",
            "error": str(e),
        }