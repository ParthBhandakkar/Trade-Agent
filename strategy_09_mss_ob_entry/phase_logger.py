"""
Phase-based Event Logger for Strategy 09 (MSS + Order Block).

Creates separate JSONL log files for **each phase** so events can be
tracked independently:

  phase_logs/
    4H_bias_log.jsonl           ← Phase 1: 4H Liquidity Sweep bias
    1H_mss_log.jsonl            ← Phase 2: 1H Market Structure Shift
    15M_ob_log.jsonl            ← Phase 3: 15M Order Block in OTE
    5M_tap_log.jsonl            ← Phase 4: 5M OB Tap entry
    signals_log.jsonl           ← Final signals generated
    cycle_summary_log.jsonl     ← Per-cycle aggregate counts
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytz

IST = pytz.timezone("Asia/Kolkata")
UTC = pytz.UTC
logger = logging.getLogger(__name__)


class PhaseLogger:
    """Append-only JSONL logger with one file per strategy phase."""

    def __init__(self, log_dir: Optional[Path] = None) -> None:
        if log_dir is None:
            log_dir = Path(__file__).resolve().parent / "phase_logs"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.bias_log = self.log_dir / "4H_bias_log.jsonl"
        self.mss_log = self.log_dir / "1H_mss_log.jsonl"
        self.ob_log = self.log_dir / "15M_ob_log.jsonl"
        self.tap_log = self.log_dir / "5M_tap_log.jsonl"
        self.signals_log = self.log_dir / "signals_log.jsonl"
        self.cycle_log = self.log_dir / "cycle_summary_log.jsonl"

    # ------------------------------------------------------------------ utils
    def _append(self, filepath: Path, entry: Dict[str, Any]) -> None:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    @staticmethod
    def _now_ist() -> str:
        return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")

    # ======================================================================
    # Phase 1 — 4H Bias (Liquidity Sweep)
    # ======================================================================
    def log_bias(
        self,
        symbol: str,
        biases_found: int,
        bias_details: List[Dict[str, Any]],
        lookback_hours: int = 72,
    ) -> None:
        self._append(self.bias_log, {
            "timestamp_ist": self._now_ist(),
            "symbol": symbol,
            "phase": "Phase1_4H_Bias",
            "status": "PASS" if biases_found > 0 else "FAIL",
            "biases_found": biases_found,
            "lookback_hours": lookback_hours,
            "details": bias_details,
        })

    # ======================================================================
    # Phase 2 — 1H MSS Confirmation
    # ======================================================================
    def log_mss(
        self,
        symbol: str,
        bias_direction: str,
        sweep_time: Any,
        confirmed: bool,
        mss_break_price: Optional[float] = None,
        mss_time: Any = None,
        details: str = "",
    ) -> None:
        self._append(self.mss_log, {
            "timestamp_ist": self._now_ist(),
            "symbol": symbol,
            "phase": "Phase2_1H_MSS",
            "status": "PASS" if confirmed else "FAIL",
            "bias_direction": bias_direction,
            "sweep_time": str(sweep_time),
            "confirmed": confirmed,
            "mss_break_price": mss_break_price,
            "mss_time": str(mss_time) if mss_time else None,
            "details": details,
        })

    # ======================================================================
    # Phase 3 — 15M Order Block in OTE Zone
    # ======================================================================
    def log_ob(
        self,
        symbol: str,
        bias_direction: str,
        mss_time: Any,
        ob_found: bool,
        ob_time: Any = None,
        ob_top: Optional[float] = None,
        ob_bottom: Optional[float] = None,
        fib_level: Optional[float] = None,
        in_ote: bool = False,
    ) -> None:
        self._append(self.ob_log, {
            "timestamp_ist": self._now_ist(),
            "symbol": symbol,
            "phase": "Phase3_15M_OB",
            "status": "PASS" if ob_found else "FAIL",
            "bias_direction": bias_direction,
            "mss_time": str(mss_time) if mss_time else None,
            "ob_found": ob_found,
            "ob_time": str(ob_time) if ob_time else None,
            "ob_top": ob_top,
            "ob_bottom": ob_bottom,
            "fib_level": fib_level,
            "in_ote": in_ote,
        })

    # ======================================================================
    # Phase 4 — 5M OB Tap (Entry)
    # ======================================================================
    def log_tap(
        self,
        symbol: str,
        bias_direction: str,
        ob_time: Any,
        tapped: bool,
        tap_time: Any = None,
        entry_price: Optional[float] = None,
    ) -> None:
        self._append(self.tap_log, {
            "timestamp_ist": self._now_ist(),
            "symbol": symbol,
            "phase": "Phase4_5M_Tap",
            "status": "PASS" if tapped else "FAIL",
            "bias_direction": bias_direction,
            "ob_time": str(ob_time) if ob_time else None,
            "tapped": tapped,
            "tap_time": str(tap_time) if tap_time else None,
            "entry_price": entry_price,
        })

    # ======================================================================
    # Final Signal
    # ======================================================================
    def log_signal(
        self,
        symbol: str,
        direction: str,
        signal_time_ist: str,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        quality_score: int,
        risk_reward: float = 0.0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._append(self.signals_log, {
            "timestamp_ist": self._now_ist(),
            "symbol": symbol,
            "phase": "Signal",
            "status": "GENERATED",
            "direction": direction,
            "signal_time_ist": signal_time_ist,
            "entry_price": entry_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "quality_score": quality_score,
            "risk_reward": risk_reward,
            "extra": extra or {},
        })

    # ======================================================================
    # Cycle Summary
    # ======================================================================
    def log_cycle(
        self,
        cycle_time_ist: str,
        symbols_scanned: int,
        phase1_pass: int = 0,
        phase2_pass: int = 0,
        phase3_pass: int = 0,
        phase4_pass: int = 0,
        signals_generated: int = 0,
    ) -> None:
        self._append(self.cycle_log, {
            "timestamp_ist": self._now_ist(),
            "type": "CYCLE_SUMMARY",
            "cycle_time_ist": cycle_time_ist,
            "symbols_scanned": symbols_scanned,
            "phase1_4h_bias_pass": phase1_pass,
            "phase2_1h_mss_pass": phase2_pass,
            "phase3_15m_ob_pass": phase3_pass,
            "phase4_5m_tap_pass": phase4_pass,
            "signals_generated": signals_generated,
        })
