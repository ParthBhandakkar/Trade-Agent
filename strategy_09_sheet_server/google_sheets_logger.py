from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Sequence

import gspread
from google.oauth2.service_account import Credentials
from gspread.exceptions import WorksheetNotFound
from gspread.utils import rowcol_to_a1

DEFAULT_TRADE_HEADERS: List[str] = [
    "record_id",
    "logged_at_ist",
    "detected_at_ist",
    "signal_time_ist",
    "market",
    "symbol",
    "direction",
    "trade_status",
    "source_strategy",
    "delivery_target",
    "quality_score",
    "risk_reward",
    "entry_price",
    "stop_loss",
    "take_profit",
    "risk_per_unit",
    "sl_reason",
    "killzone_status",
    "killzone_reason",
    "ema_filter_status",
    "ema_filter_detail",
    "bias_direction",
    "bias_confidence",
    "bias_reason",
    "bias_sweep_time_ist",
    "bias_source_time_ist",
    "mss_time_ist",
    "mss_break_price",
    "mss_confirmation_close",
    "mss_details",
    "ob_time_ist",
    "ob_top",
    "ob_bottom",
    "ob_body_top",
    "ob_body_bottom",
    "ob_fib_level",
    "ob_in_ote_zone",
    "vm_hostname",
    "raw_signal_json",
]

_SHEETS_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


class GoogleSheetsTradeLogger:
    """Append Strategy 09 trade rows into a Google Sheet worksheet."""

    def __init__(
        self,
        spreadsheet_id: str,
        worksheet_title: str,
        service_account_file: Optional[str] = None,
        service_account_json: Optional[str] = None,
        headers: Optional[Sequence[str]] = None,
        max_retries: int = 3,
        retry_delay_seconds: float = 2.0,
    ) -> None:
        self.spreadsheet_id = spreadsheet_id
        self.worksheet_title = worksheet_title
        self.service_account_file = service_account_file
        self.service_account_json = service_account_json
        self.expected_headers = list(headers or DEFAULT_TRADE_HEADERS)
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

        self._client: Optional[gspread.Client] = None
        self._spreadsheet = None
        self._worksheet = None
        self._headers_cache: Optional[List[str]] = None

    def connect(self) -> None:
        credentials = self._build_credentials()
        self._client = gspread.authorize(credentials)
        self._spreadsheet = self._client.open_by_key(self.spreadsheet_id)
        self._worksheet = self._get_or_create_worksheet()
        self._headers_cache = self._ensure_headers()

    def append_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        if self._worksheet is None or self._headers_cache is None:
            self.connect()

        headers = self._headers_cache or self.expected_headers
        row = [self._normalize_value(record.get(header, "")) for header in headers]

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                self._worksheet.append_row(
                    row,
                    value_input_option="USER_ENTERED",
                    insert_data_option="INSERT_ROWS",
                )
                return {
                    "success": True,
                    "worksheet": self.worksheet_title,
                    "columns_used": headers,
                }
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
                self._worksheet = None
                self._headers_cache = None
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_seconds * attempt)
                    self.connect()

        raise RuntimeError(
            f"Failed appending record to Google Sheets worksheet "
            f"'{self.worksheet_title}': {last_error}"
        )

    def _build_credentials(self) -> Credentials:
        if self.service_account_json:
            info = json.loads(self.service_account_json)
            return Credentials.from_service_account_info(
                info,
                scopes=_SHEETS_SCOPES,
            )
        if self.service_account_file:
            return Credentials.from_service_account_file(
                self.service_account_file,
                scopes=_SHEETS_SCOPES,
            )
        raise RuntimeError(
            "Missing Google service account credentials. Set "
            "GOOGLE_SERVICE_ACCOUNT_FILE or GOOGLE_SERVICE_ACCOUNT_JSON."
        )

    def _get_or_create_worksheet(self):
        if self._spreadsheet is None:
            raise RuntimeError("Google spreadsheet client not initialized")

        try:
            return self._spreadsheet.worksheet(self.worksheet_title)
        except WorksheetNotFound:
            cols = max(26, len(self.expected_headers) + 6)
            return self._spreadsheet.add_worksheet(
                title=self.worksheet_title,
                rows="2000",
                cols=str(cols),
            )

    def _ensure_headers(self) -> List[str]:
        if self._worksheet is None:
            raise RuntimeError("Google worksheet not initialized")

        current_headers = [value.strip() for value in self._worksheet.row_values(1)]
        current_headers = [value for value in current_headers if value]

        if not current_headers:
            self._write_headers(self.expected_headers)
            return list(self.expected_headers)

        missing_headers = [
            header for header in self.expected_headers if header not in current_headers
        ]
        if not missing_headers:
            return current_headers

        self._worksheet.add_cols(len(missing_headers))
        final_headers = current_headers + missing_headers
        self._write_headers(final_headers)
        return final_headers

    def _write_headers(self, headers: Sequence[str]) -> None:
        if self._worksheet is None:
            raise RuntimeError("Google worksheet not initialized")

        end_cell = rowcol_to_a1(1, len(headers))
        self._worksheet.update(f"A1:{end_cell}", [list(headers)])

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value, default=str)
        return value
