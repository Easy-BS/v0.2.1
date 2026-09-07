# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 08:28:09 2025

@author: Xiguan Liang @SKKU
"""

# ./CALI_flow/nodes/llm_router.py

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Dict

from state_schema import SimulationState

OPENAI_URL = "https://api.openai.com/v1/chat/completions"

SYSTEM_PROMPT = """You are an AI agent for building energy model calibration.
Read the user's text and output STRICT JSON only (no markdown).

Schema:
{
  "intent": "calibrate_building" | "ask_clarification" | "unknown",
  "calibration_level": "basic" | "unknown",
  "building_type": "residential" | "commercial" | "public" | "unknown",
  "measured_monthly_kwh": {
    "1": number,
    "2": number
  },
  "measured_year": 2024,
  "coverage_months": [1,2,3],
  "clarification_question": string
}

Rules:
- If the user asks to calibrate a building model and provides monthly measured heating energy, use intent="calibrate_building".
- Convert month names to month numbers as strings: January->"1", February->"2", ... December->"12".
- calibration_level="basic" if the user says "basic calibration".
- If monthly measured data is missing or too incomplete, use intent="ask_clarification".
- If unrelated, use intent="unknown".
- If the user states the calendar year of the measured data (for example "for each month of 2024"), report it in measured_year. If no year is stated, omit the field.
Return JSON only.
"""

import re

MONTH_MAP = {
    "january": "1",
    "february": "2",
    "march": "3",
    "april": "4",
    "may": "5",
    "june": "6",
    "july": "7",
    "august": "8",
    "september": "9",
    "october": "10",
    "november": "11",
    "december": "12",
}

def parse_monthly_kwh_from_text(text: str) -> dict[str, float]:
    out = {}
    for name, num in MONTH_MAP.items():
        m = re.search(
            rf"\b{name}\b\s*[-–—:]?\s*([0-9]+(?:\.[0-9]+)?)",
            text,
            flags=re.IGNORECASE
        )
        if m:
            out[num] = float(m.group(1))
    return out

def _openai_chat_json(user_text: str, model: str) -> Dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment.")

    payload = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
    }

    req = urllib.request.Request(
        OPENAI_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8")

    data = json.loads(raw)
    content = data["choices"][0]["message"]["content"]
    return json.loads(content)

def llm_router(state: SimulationState) -> SimulationState:
    text = (state.get("user_input") or "").strip()
    if not text:
        return {"errors": ["No user input received."]}

    model = state.get("model") or "gpt-4o-mini"

    try:
        agent = _openai_chat_json(text, model=model)
    except Exception as e:
        return {"errors": [f"LLM router failed: {e}"]}

    intent = agent.get("intent", "unknown")
    out: SimulationState = {
        "agent_json": agent,
        "intent": intent,
        "calibration_level": agent.get("calibration_level", "unknown"),
        "building_type": agent.get("building_type", "unknown"),
    }

    if intent == "calibrate_building":
        measured_llm = agent.get("measured_monthly_kwh") or {}
        measured_regex = parse_monthly_kwh_from_text(text)
        y = agent.get("measured_year")
        if y:
            try:
                y = int(str(y)[:4])
                if 1900 < y < 2100:
                    out["measured_year"] = y
            except Exception:
                pass
        cleaned = {}
        for k, v in measured_llm.items():
            ks = str(k).strip()
            if ks.isdigit():
                cleaned[ks] = float(v)
        
        # raw-text regex extraction fills any missing months
        for k, v in measured_regex.items():
            cleaned[k] = v
        
        months = sorted(int(k) for k in cleaned.keys())
        if len(months) < 2:
            out["intent"] = "ask_clarification"
            out["clarification_question"] = (
                "Please provide at least two months of measured monthly heating energy in kWh."
            )
        else:
            out["measured_monthly_kwh"] = cleaned
            out["coverage_months"] = months

    elif intent == "ask_clarification":
        out["clarification_question"] = agent.get("clarification_question") or (
            "Please provide your monthly measured heating energy in kWh, for example: "
            "January - 5530.4, February - 5716.2."
        )

    return out
