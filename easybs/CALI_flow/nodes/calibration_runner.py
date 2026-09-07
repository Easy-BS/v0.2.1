# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 16:32:56 2025

@author: Xiguan Liang @SKKU
"""

# ./CALI_flow/nodes/calibration_runner.py

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple, List

from state_schema import SimulationState


def _run_stage(script_path: Path, config_json_path: Path, cwd: Path) -> Tuple[str, str]:
    cmd = [sys.executable, str(script_path), "--config", str(config_json_path)]
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8"
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Stage failed: {script_path.name}\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )
    return proc.stdout, proc.stderr


def _extract_last_idf_path(text: str) -> str | None:
    patterns = [
        r"Calibrated IDF saved to:\s*(.+?\.idf)",
        r"\[OK\].*?(\S+After2_Cali_RFH\.idf)",
        r"Saved updated schedules to:\s*(.+?\.idf)",
        r"Saved with occupancy/internal gains:\s*(.+?\.idf)",
        r"Saved:\s*(.+?\.idf)",
        r"Output IDF:\s*(.+?\.idf)",
    ]

    matches = []
    for pat in patterns:
        matches.extend(re.findall(pat, text, flags=re.IGNORECASE))

    if matches:
        return matches[-1].strip()   # take the LAST path, not the first
    return None


def _extract_metrics(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    cvr = re.findall(r"CVRMSE\s*[:=]\s*([+-]?\d+(?:\.\d+)?)\s*%", text, flags=re.IGNORECASE)
    nb = re.findall(r"NMBE\s*[:=]\s*([+-]?\d+(?:\.\d+)?)\s*%", text, flags=re.IGNORECASE)

    if cvr:
        out["cvrmse_percent"] = float(cvr[-1])   # take LAST occurrence
    if nb:
        out["nmbe_percent"] = float(nb[-1])      # take LAST occurrence

    return out


def calibration_runner(state: SimulationState) -> SimulationState:
    try:
        repo_root = Path(__file__).resolve().parents[2]   # project root
        cali_nodes_dir = repo_root / "CALI_flow" / "nodes"

        # Existing deterministic scripts
        scripts: List[str] = [
            "Add_infil.py",
            "Add_internal_heatgain.py",
            "Add_schedule.py",
            "Add_Other.py",
            "Cali_Envelope.py",
            "Cali_Tset_Occ.py",
            "Cali_Tset_Detail.py",
        ]

        config = {
            "idf_path": state["idf_path"],
            "idd_path": state.get("idd_path", r"C:/EnergyPlusV8-9-0/Energy+.idd"),
            "epw_path": state.get("epw_path", r"C:\EnergyPlusV8-9-0\WeatherData\KOR_INCH'ON_IWEC.epw"),
            "measured_monthly_kwh": state["measured_monthly_kwh"],
            "coverage_months": state.get("coverage_months", []),
            "building_type": state.get("building_type", "residential"),
            "calibration_level": state.get("calibration_level", "basic"),
            "structure": state.get("structure", "masonry"),
            "construction_year": state.get("construction_year", 0),
            "windows_replaced": state.get("windows_replaced", None),
        }

        stage_outputs: Dict[str, Any] = {}
        final_idf_path = state["idf_path"]
        final_metrics: Dict[str, Any] = {}

        with tempfile.TemporaryDirectory(prefix="cali_cfg_") as td:
            cfg_path = Path(td) / "cali_runtime_config.json"

            for script_name in scripts:
                config["idf_path"] = final_idf_path
                cfg_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

                script_path = cali_nodes_dir / script_name
                stdout, stderr = _run_stage(script_path, cfg_path, cwd=repo_root)

                stage_outputs[script_name] = {
                    "stdout_tail": stdout[-3000:],
                    "stderr_tail": stderr[-1000:] if stderr else ""
                }

                maybe_idf = _extract_last_idf_path(stdout)
                if maybe_idf:
                    final_idf_path = maybe_idf

                maybe_metrics = _extract_metrics(stdout)
                if maybe_metrics:
                    final_metrics.update(maybe_metrics)

        msg = "Calibration finished successfully."
        if final_metrics:
            msg += (
                f" Final metrics: CVRMSE={final_metrics.get('cvrmse_percent')} %, "
                f"NMBE={final_metrics.get('nmbe_percent')} %."
            )

        return {
            "idf_path": final_idf_path,
            "metrics": final_metrics,
            "stage_outputs": stage_outputs,
            "message": msg,
        }

    except Exception as e:
        return {"errors": [f"Calibration runner failed: {e}"]}