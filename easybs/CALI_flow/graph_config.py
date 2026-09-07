# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 10:31:18 2025

@author: Xiguan Liang @SKKU
"""

# ./CALI_flow/graph_config.py

from langgraph.graph import StateGraph, END
from state_schema import SimulationState
from nodes.amy_weather import amy_weather
from nodes.user_query_parser import user_query_parser
from nodes.llm_router import llm_router
from nodes.building_data_extractor import extract_calibration_inputs
from nodes.calibration_runner import calibration_runner

def _route(state: SimulationState) -> str:
    intent = state.get("intent", "unknown")
    if intent == "calibrate_building":
        return "building_data_extractor"
    if intent == "ask_clarification":
        return "clarify"
    return "unknown"

def clarify_node(state: SimulationState) -> SimulationState:
    q = state.get("clarification_question") or (
        "Please provide your measured monthly heating energy in kWh."
    )
    state["message"] = q
    return state

def unknown_node(state: SimulationState) -> SimulationState:
    state["message"] = (
        "I can calibrate the building model if you provide monthly measured heating energy in kWh."
    )
    return state

def define_graph():
    g = StateGraph(SimulationState)
    g.add_node("amy_weather", amy_weather)

    g.add_node("user_query_parser", user_query_parser)
    g.add_node("llm_router", llm_router)
    g.add_node("building_data_extractor", extract_calibration_inputs)
    g.add_node("calibration_runner", calibration_runner)
    g.add_node("clarify", clarify_node)
    g.add_node("unknown", unknown_node)

    g.set_entry_point("user_query_parser")
    g.add_edge("user_query_parser", "llm_router")

    g.add_conditional_edges(
        "llm_router",
        _route,
        {
            "building_data_extractor": "building_data_extractor",
            "clarify": "clarify",
            "unknown": "unknown",
        },
    )

    g.add_edge("building_data_extractor", "amy_weather")
    g.add_edge("amy_weather", "calibration_runner")
    g.add_edge("calibration_runner", END)
    g.add_edge("clarify", END)
    g.add_edge("unknown", END)

    return g.compile()

