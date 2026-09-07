# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 07:09:03 2025

@author: Xiguan Liang @SKKU
"""
#./CALI_flow/nodes/user_query_parser.py

from __future__ import annotations
from state_schema import SimulationState

def user_query_parser(state: SimulationState) -> SimulationState:
    text = (state.get("user_input") or "").strip()
    if not text:
        return {"errors": ["No user input received."]}
    return {"user_input": text}