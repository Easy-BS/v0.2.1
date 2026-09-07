# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 08:30:09 2025

@author: Xiguan Liang @SKKU
"""

# ./CALI_flow/state_schema.py

from typing import TypedDict, Optional, List, Dict, Any, Literal

class SimulationState(TypedDict, total=False):
    # weather
    measured_year: int
    tmy_epw_path: str
    weather_outdir: str
    weather_report: Dict[str, Any]
    
    structure: str
    construction_year: int
    windows_replaced: bool
    
    user_input: str

    # file IO
    idf_path: str
    idd_path: str
    epw_path: str
    output_dir: str

    # agent fields
    intent: Literal["calibrate_building", "ask_clarification", "unknown"]
    calibration_level: Literal["basic", "unknown"]
    building_type: Literal["residential", "commercial", "public", "unknown"]
    measured_monthly_kwh: Dict[str, float]
    coverage_months: List[int]
    clarification_question: str
    agent_json: Dict[str, Any]
    model: str

    # outputs
    message: str
    metrics: Dict[str, Any]
    stage_outputs: Dict[str, Any]
    errors: Optional[List[str]]


