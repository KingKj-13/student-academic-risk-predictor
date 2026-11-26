"""Early-warning rules and optional email notification for high-risk students."""

from __future__ import annotations

import logging
import os
import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage

logger = logging.getLogger("early_warning")

STATUS = {"Low": "On Track", "Medium": "Monitor", "High": "At Risk"}

BASE_ADVICE = {
    "Low": ["No immediate academic intervention is indicated."],
    "Medium": ["Keep this student under academic monitoring."],
    "High": ["Arrange a one-to-one discussion with the student soon."],
}
FACTOR_ADVICE = {
    "Low attendance": "Review the attendance record and ask about any difficulty in attending.",
    "Low internal marks": "Offer remedial sessions for the subjects with low internal marks.",
    "Low assignment score": "Check which assignments are missing or weak and agree on a plan.",
    "Insufficient study hours": "Help the student draw up a weekly study timetable.",
    "Previous academic failures": "Revisit the topics from previously failed subjects.",
}


def status_for(risk_level: str) -> str:
    return STATUS[risk_level]


def recommendations_for(risk_level: str, factors: list[str]) -> list[str]:
    advice = list(BASE_ADVICE[risk_level])
    if risk_level != "Low":
        advice += [FACTOR_ADVICE[f] for f in factors if f in FACTOR_ADVICE]
    return advice


def is_new_alert(risk_level: str, previous_level: str | None) -> bool:
    """Alert when a student enters High risk, not every time they remain there."""
    return risk_level == "High" and previous_level != "High"


@dataclass(frozen=True)
class EmailSettings:
    host: str
    port: int
    username: str
    password: str
    sender: str
    recipient: str

    @classmethod
    def from_env(cls) -> "EmailSettings | None":
        """Read SMTP settings from the environment. Email stays off unless complete."""
        host = os.getenv("SMTP_HOST", "").strip()
        recipient = os.getenv("ALERT_TO", "").strip()
        username = os.getenv("SMTP_USERNAME", "").strip()
        password = os.getenv("SMTP_PASSWORD", "")
        sender = os.getenv("ALERT_FROM", "").strip() or username
        if not (host and recipient and sender):
            return None
        return cls(host, int(os.getenv("SMTP_PORT", "587")), username, password, sender, recipient)


def build_message(students: list[dict], settings: EmailSettings) -> EmailMessage:
    """One message for any number of students, so a bulk upload does not flood the inbox."""
    message = EmailMessage()
    message["Subject"] = f"Early warning: {len(students)} student(s) newly at high academic risk"
    message["From"] = settings.sender
    message["To"] = settings.recipient
    lines = ["The following students were newly classified as High risk:", ""]
    for student in students:
        factors = ", ".join(student["contributing_factors"]) or "no single dominant factor"
        lines.append(f"- {student['student_id']}  {student['name']}  ({factors})")
    lines += ["", "This is a model estimate to support your judgement, not a confirmed outcome."]
    message.set_content("\n".join(lines))
    return message


def send_alert_email(students: list[dict], settings: EmailSettings | None = None) -> bool:
    """Send the alert digest. Returns False (and logs why) instead of raising."""
    settings = settings or EmailSettings.from_env()
    if not students or settings is None:
        return False
    try:
        context = ssl.create_default_context()
        if settings.port == 465:
            server: smtplib.SMTP = smtplib.SMTP_SSL(settings.host, settings.port, timeout=15, context=context)
        else:
            server = smtplib.SMTP(settings.host, settings.port, timeout=15)
        with server:
            if settings.port != 465:
                server.starttls(context=context)
            if settings.username:
                server.login(settings.username, settings.password)
            server.send_message(build_message(students, settings))
        logger.info("Alert email sent for %d student(s)", len(students))
        return True
    except (smtplib.SMTPException, OSError) as error:
        logger.error("Alert email could not be sent: %s", error)
        return False


def raise_alerts(students: list[dict]) -> None:
    """Log each new alert and email them if email is configured."""
    for student in students:
        # IDs only: names are personal data and do not belong in server logs.
        logger.warning("HIGH RISK alert for student_id=%s", student["student_id"])
    send_alert_email(students)
