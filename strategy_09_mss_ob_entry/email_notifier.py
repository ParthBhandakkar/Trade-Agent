"""
Email Notifier for Strategy 09 (MSS + Order Block) Live Signals.

Sends formatted email alerts when a final signal is generated.

Required environment variables (in the repository root .env):
  SMTP_SERVER   (default: smtp.gmail.com)
  SMTP_PORT     (default: 587)
  SENDER_EMAIL
  SENDER_PASSWORD
  RECIPIENT_EMAIL
"""

from __future__ import annotations

import os
import smtplib
import logging
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, Optional

import pytz

IST = pytz.timezone("Asia/Kolkata")
logger = logging.getLogger(__name__)


class EmailNotifier:
    """SMTP email notifier for live trading signals."""

    def __init__(self) -> None:
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.sender_email = os.getenv("SENDER_EMAIL", "")
        self.sender_password = os.getenv("SENDER_PASSWORD", "")
        self.recipient_email = os.getenv("RECIPIENT_EMAIL", "")

        missing: list[str] = []
        if not self.sender_email:
            missing.append("SENDER_EMAIL")
        if not self.sender_password:
            missing.append("SENDER_PASSWORD")
        if not self.recipient_email:
            missing.append("RECIPIENT_EMAIL")
        self.missing_env_vars = missing
        self.enabled = len(missing) == 0

    def send(self, subject: str, body: str) -> bool:
        """Send a plain-text email. Returns True on success."""
        if not self.enabled:
            if self.missing_env_vars:
                logger.warning(
                    f"Email disabled — missing env vars: "
                    f"{', '.join(self.missing_env_vars)}"
                )
            return False

        try:
            msg = MIMEMultipart()
            msg["From"] = self.sender_email
            msg["To"] = self.recipient_email
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)

            logger.info(f"📧 Email sent: {subject}")
            return True
        except Exception as e:
            logger.error(f"Email send failed: {type(e).__name__}: {e}")
            return False

    def send_signal(self, sig_json: Dict[str, Any], market: str = "CRYPTO") -> bool:
        """
        Send a formatted signal email.

        Parameters
        ----------
        sig_json : dict
            The output of ``format_signal_for_jsonl(signal)``.
        market : str
            'CRYPTO' or 'FOREX' — shown in the subject line.
        """
        symbol = sig_json.get("symbol", "???")
        direction = sig_json.get("direction", "???").upper()
        entry = sig_json.get("entry_price", 0)
        sl = sig_json.get("sl_price", 0)
        tp = sig_json.get("tp_price", 0)
        rr = sig_json.get("risk_reward", 0)
        quality = sig_json.get("quality_score", 0)

        # Nested helpers
        def _g(path: str) -> str:
            """Dot-path getter into sig_json."""
            cur: Any = sig_json
            for p in path.split("."):
                if isinstance(cur, dict):
                    cur = cur.get(p, "")
                else:
                    return ""
            return str(cur) if cur else ""

        subject = (
            f"[S09 {market}] SIGNAL: {symbol} {direction} "
            f"(Q{quality}, {rr:.1f}R)"
        )

        body = (
            f"═══════════════════════════════════════════\n"
            f"  STRATEGY 09 — MSS + Order Block\n"
            f"  Market: {market}\n"
            f"═══════════════════════════════════════════\n\n"
            f"Symbol:       {symbol}\n"
            f"Direction:    {direction}\n"
            f"Quality:      {quality}/100\n"
            f"Signal Time:  {sig_json.get('signal_datetime_ist', '')}\n\n"
            f"── ENTRY ──────────────────────────────────\n"
            f"Entry Price:  {entry}\n"
            f"Stop Loss:    {sl}\n"
            f"Take Profit:  {tp}  (2R)\n"
            f"Risk:Reward:  {rr}\n\n"
            f"── PHASE 1: 4H BIAS ───────────────────────\n"
            f"Direction:    {_g('daily_bias.direction')}\n"
            f"Confidence:   {_g('daily_bias.confidence')}\n"
            f"Reason:       {_g('daily_bias.reason')}\n"
            f"Sweep Time:   {_g('daily_bias.sweep_time_ist')}\n\n"
            f"── PHASE 2: 1H MSS ────────────────────────\n"
            f"Direction:    {_g('mss_confirmation.direction')}\n"
            f"Break Price:  {_g('mss_confirmation.break_price')}\n"
            f"Close Price:  {_g('mss_confirmation.confirmation_close')}\n"
            f"MSS Time:     {_g('mss_confirmation.time_ist')}\n"
            f"Details:      {_g('mss_confirmation.details')}\n\n"
            f"── PHASE 3: 15M OB + 5M TAP ────────────────\n"
            f"OB Time:      {_g('ob_entry.ob_time_ist')}\n"
            f"OB Top:       {_g('ob_entry.ob_top')}\n"
            f"OB Bottom:    {_g('ob_entry.ob_bottom')}\n"
            f"Fib Level:    {_g('ob_entry.fib_level')}\n"
            f"In OTE Zone:  {_g('ob_entry.in_ote_zone')}\n\n"
            f"── DETECTED ───────────────────────────────\n"
            f"Detected At:  {sig_json.get('detected_at_ist', '')}\n"
            f"═══════════════════════════════════════════\n"
        )

        return self.send(subject, body)
