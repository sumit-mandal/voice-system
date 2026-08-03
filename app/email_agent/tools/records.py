"""History-records tool stub. Replace body with real parameterized SQL later."""

from __future__ import annotations 
from typing import Any 

def fetch_history_records(*, patient_id: str | None, from_address: str) -> dict[str, Any]:
    # TODO: map from_address → patient_id, then SELECT ... with bound params (never raw LLM SQL)
    if not patient_id:
        return {
            "ok": False,
            "error": "patient_id_unresolved",
            "records":[],
        }
    
    return {
        "ok": True,
        "patient_id": patient_id,
        "records": [
            {"date": "2025-01-10", "type": "visit", "summary": "Annual checkup"},
            {"date": "2024-08-02", "type": "lab", "summary": "CBC within normal limits"},
        ],
    }