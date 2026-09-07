# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 08:25:55 2025

@author: user
"""

# ./CALI_flow/nodes/building_data_extractor.py

from __future__ import annotations

import os
from pathlib import Path

from state_schema import SimulationState

def extract_calibration_inputs(state: SimulationState) -> SimulationState:
    measured = state.get("measured_monthly_kwh") or {}
    if not measured:
        return {"errors": ["No measured_monthly_kwh found in state."]}

    idf_path = state.get("idf_path")
    if not idf_path:
        return {
            "errors": [
                "No input IDF path was provided. Please run the building generation or RFH step first."
            ]
        }

    p = Path(idf_path)
    if not p.exists():
        return {"errors": [f"Input IDF not found: {p.resolve()}"]}

    months = sorted(int(k) for k in measured.keys())
    if len(months) < 2:
        return {"errors": ["At least two measured months are required for calibration."]}

    return {
        "idf_path": str(p.resolve()),
        "coverage_months": months,
        "message": "Calibrating the above building model..."
    }