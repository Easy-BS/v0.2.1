# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 17:11:44 2026

@author: Xiguan Liang @SKKU
"""

# ./CALI_flow/nodes/Cali_Tset_Detail.py


from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import shutil
import subprocess
import time
import re
import csv
import math

import numpy as np
import pandas as pd
from eppy.modeleditor import IDF

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize

from cali_runtime_config import load_runtime_config
_RUNTIME = load_runtime_config()


# ============================================================
# Experiment inputs. No fallbacks except the IDD.
# ============================================================
def _require(key: str, what: str):
    """Fetch a runtime value, or fail loudly.

    A missing measured dataset, weather file or model must stop the run.
    Substituting a default would produce a calibrated model and a reported
    CVRMSE against data nobody supplied, with nothing in the output to
    reveal it.
    """
    val = _RUNTIME.get(key)
    if val is None or (isinstance(val, (str, dict, list)) and len(val) == 0):
        raise RuntimeError(
            f"Missing '{key}' in the runtime configuration ({what}). "
            "Pass it via --config from calibration_runner. "
            "Calibration will not proceed with a default."
        )
    return val


# Measured monthly heating energy in kWh, keyed by month number as a string.
# Keys are normalized so 1, "1" and "01" all become "1", which matters because
# align_months() looks up measured_kwh[str(month)].
MEASURED_MONTHLY_KWH: Dict[str, float] = {
    str(int(k)): float(v)
    for k, v in _require("measured_monthly_kwh",
                         "measured monthly heating energy in kWh").items()
}
if len(MEASURED_MONTHLY_KWH) < 2:
    raise RuntimeError(
        "At least two measured months are required for CVRMSE and NMBE; got "
        f"{sorted(int(m) for m in MEASURED_MONTHLY_KWH)}."
    )

MEASURED_MONTHS: List[int] = sorted(int(m) for m in MEASURED_MONTHLY_KWH)

# The IDD is an installation path, not experiment data, so it keeps a default.
IDD_PATH = Path(_RUNTIME.get("idd_path", r"C:/EnergyPlusV8-9-0/Energy+.idd"))
EPW_PATH = Path(_require("epw_path", "weather file for the measured year"))
INPUT_IDF_PATH = Path(_require("idf_path", "model to calibrate"))
OUTPUT_IDF_PATH = Path("./Calibration/After2_Cali_RFH.idf")

# Report written by the amy_weather node, when present
WEATHER_REPORT = _RUNTIME.get("weather_report") or {}

# Working directory for repeated simulations
WORK_DIR: Path = Path("./Calibration/_cali_runs_tset_detail")

# Cleanup policy
CLEAN_WORKSPACE_EACH_START: bool = True
CLEANUP_EACH_RUN: bool = True
KEEP_FAILED_RUNS: bool = True

# Target schedules (monthly-dependent)
RADIANT_SETPOINT_SCHED_NAME = "RADIANT HEATING SETPOINTS"
AIR_HEATING_SETPOINT_SCHED_NAME = "HEATING SETPOINTS"

# Heating meters priority (Monthly, J) summed -> kWh
HEATING_METERS_PRIORITY: List[str] = [
    "DistrictHeating:Facility",
    "Electricity:Heating",
    "Gas:Heating",
]
#%%

# Read the boiler efficiency selected in stage 1. If not available the
# midpoint of the prior is used, which is a safe fallback.
_STAGE1 = {}
try:
    import json as _json
    _STAGE1 = _json.loads(
        Path("./Calibration/stage1_result.json").read_text(encoding="utf-8"))
except Exception:
    pass
BOILER_EFF: float = _STAGE1.get("params", {}).get("boiler_eff", 0.815)


def heat_to_fuel(heat_kwh: Dict[int, float], efficiency: float) -> Dict[int, float]:
    """Convert delivered heat into metered fuel.

    The measured record is fuel; EnergyPlus reports heat. Comparing them
    directly understates the model by the boiler loss.
    """
    eta = max(float(efficiency), 1e-6)
    return {mm: v / eta for mm, v in heat_kwh.items()}
#%%
# ============================================================
# Bounds
# ============================================================
# The setpoint domain is deliberately wider here than in stage 2, where
# setpoints are optimized jointly with occupancy density inside a plausible
# residential band. Relaxing it at this stage lets the monthly residuals drive
# the schedule.
#
# Constraint enforced by clipping: T_home >= T_away.
BOUNDS = {
    "t_home": (17.0, 27.0),
    "t_away": (8.0, 22.0),
}

MIN_HOME_AWAY_GAP = 0.0   # set to e.g. 0.5 to force a strict separation

# ============================================================
# Heuristic tuning controls
# ============================================================
MAX_OUTER_ITERS = 18          # number of heuristic rounds
BASE_STEP_HOME = 0.7          # degC initial step for T_home
BASE_STEP_AWAY = 0.8          # degC initial step for T_away
STEP_DECAY = 0.96             # step reduction per outer iteration
MONTH_WEIGHT_MODE = "relative"   # "relative" or "absolute"
CLAMP_MAX_PER_ITER = 2.0      # max |delta| per month per outer iteration, degC

# Local NSGA-II polish after the heuristic
ENABLE_NSAA_POLISH = True
NSGA_POP = 32
NSGA_GEN = 12
NSGA_SEED = 42
NSGA_LOCAL_SPAN_HOME = 2.5    # +/- range around the current best, degC
NSGA_LOCAL_SPAN_AWAY = 4.0

MONTH_END_DAY = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
                 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}


def echo_inputs() -> None:
    """Print the inputs actually received, so the log records what was
    calibrated against rather than what was assumed."""
    print("[INFO] Stage          : 3, 4 - residual-guided refinement and local NSGA-II")
    print(f"[INFO] Input model    : {INPUT_IDF_PATH}")
    print(f"[INFO] Weather file   : {EPW_PATH.name}")
    if WEATHER_REPORT:
        pct = WEATHER_REPORT.get("amy_vs_tmy_pct")
        extra = f", HDD18 {pct:+.1f}% vs typical year" if pct is not None else ""
        print(f"[INFO] Weather source : AMY {WEATHER_REPORT.get('year')}, "
              f"status {WEATHER_REPORT.get('status')}{extra}")
    elif "AMY" not in EPW_PATH.name.upper():
        print("[WARN] Weather file does not look like an AMY file. "
              "Check that the amy_weather node ran.")
    print(f"[INFO] Measured months: {MEASURED_MONTHS}  "
          f"(total {sum(MEASURED_MONTHLY_KWH.values()):,.0f} kWh)")


# ============================================================
# Workspace management
# ============================================================
def reset_workspace(work_dir: Path) -> None:
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# Guideline 14-style monthly metrics
# ============================================================
def nmbe_percent(meas: np.ndarray, sim: np.ndarray) -> float:
    n = len(meas)
    if n < 2:
        return float("nan")
    denom = (n - 1) * np.mean(meas)
    if denom == 0:
        return float("nan")
    return 100.0 * np.sum(sim - meas) / denom


def cvrmse_percent(meas: np.ndarray, sim: np.ndarray) -> float:
    n = len(meas)
    if n < 2:
        return float("nan")
    denom = np.mean(meas)
    if denom == 0:
        return float("nan")
    rmse = np.sqrt(np.sum((sim - meas) ** 2) / (n - 1))
    return 100.0 * rmse / denom


def align_months(measured_kwh: Dict[str, float],
                 simulated_kwh: Dict[int, float]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    months = sorted(int(k) for k in measured_kwh.keys())
    meas, sim, kept = [], [], []
    for mm in months:
        if mm in simulated_kwh:
            meas.append(float(measured_kwh[str(mm)]))
            sim.append(float(simulated_kwh[mm]))
            kept.append(mm)
    if len(meas) < 2:
        raise RuntimeError(
            f"Not enough overlapping months. Measured={months}, "
            f"SimAvailable={sorted(simulated_kwh.keys())}")
    return np.array(meas), np.array(sim), kept


def compute_metrics(measured_kwh: Dict[str, float],
                    simulated_kwh: Dict[int, float]) -> Tuple[float, float, Dict[int, float]]:
    meas, sim, months = align_months(measured_kwh, simulated_kwh)
    cvr = cvrmse_percent(meas, sim)
    nb = nmbe_percent(meas, sim)
    sim_used = {m: float(simulated_kwh[m]) for m in months}
    return cvr, nb, sim_used


# ============================================================
# EnergyPlus runner
# ============================================================
def guess_energyplus_exe(idd_path: Path) -> Path:
    candidate = idd_path.parent / "energyplus.exe"
    return candidate if candidate.exists() else Path("energyplus")


def run_energyplus(energyplus_exe: Path, idf_path: Path, epw_path: Path,
                   out_dir: Path, timeout_s: int = 3600) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [str(energyplus_exe), "-w", str(epw_path), "-d", str(out_dir), str(idf_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        msg = (proc.stdout[-4000:] if proc.stdout else "") + "\n" + (proc.stderr[-4000:] if proc.stderr else "")
        raise RuntimeError(f"EnergyPlus failed (code {proc.returncode}). Tail output:\n{msg}")


# ============================================================
# Monthly meter parsing
# ============================================================
MONTH_NAME_TO_INT = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def _month_cell_to_int(x) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in MONTH_NAME_TO_INT:
        return MONTH_NAME_TO_INT[s]
    if s.isdigit():
        mm = int(s)
        return mm if 1 <= mm <= 12 else None
    for k, v in MONTH_NAME_TO_INT.items():
        if k.startswith(s) and len(s) >= 3:
            return v
    return None


def read_monthly_meter_j_from_mtr(mtr_path: Path) -> Dict[str, Dict[int, float]]:
    """Parse EnergyPlus eplusout.mtr for Monthly meters.

    Returns { meter_name: { month -> value_J } }.
    """
    lines = mtr_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    end_idx = None
    for i, line in enumerate(lines):
        if "End of Data Dictionary" in line:
            end_idx = i
            break
    if end_idx is None:
        raise RuntimeError("Could not find 'End of Data Dictionary' in eplusout.mtr")

    dict_lines = lines[: end_idx + 1]
    data_lines = lines[end_idx + 1:]

    monthly_idx_to_name: Dict[int, str] = {}
    for line in dict_lines:
        if "!Monthly" not in line and "! Monthly" not in line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
        except Exception:
            continue
        monthly_idx_to_name[idx] = parts[2].split("[", 1)[0].strip()

    if not monthly_idx_to_name:
        raise RuntimeError("No Monthly meters detected in eplusout.mtr dictionary "
                           "(check Output:Meter Monthly).")

    meters_monthly: Dict[str, Dict[int, float]] = {n: {} for n in set(monthly_idx_to_name.values())}
    current_month: Optional[int] = None

    for line in data_lines:
        s = line.strip()
        if not s:
            continue
        parts = [p.strip() for p in s.split(",")]

        # Monthly header: "4, <cum_days>, <month>"
        if parts[0] == "4" and len(parts) >= 3:
            try:
                mm = int(parts[2])
                current_month = mm if 1 <= mm <= 12 else None
            except Exception:
                current_month = None
            continue

        try:
            idx = int(parts[0])
        except Exception:
            continue

        if idx in monthly_idx_to_name and current_month is not None and len(parts) >= 2:
            try:
                val_j = float(parts[1])
            except Exception:
                continue
            meters_monthly.setdefault(monthly_idx_to_name[idx], {})[current_month] = val_j

    meters_monthly = {k: v for k, v in meters_monthly.items() if v}
    if not meters_monthly:
        raise RuntimeError("Monthly meter indices found, but no monthly values parsed from data section.")
    return meters_monthly


def read_monthly_meter_j_from_meter_csv(meter_csv: Path) -> Dict[str, Dict[int, float]]:
    """Parse '*Meter.csv':

        Date/Time    DistrictHeating:Facility [J](Monthly)
        January      6966423054
    """
    df = pd.read_csv(meter_csv)

    month_col = None
    for c in df.columns:
        if str(c).strip().lower() in ("date/time", "date", "time"):
            month_col = c
            break
    if month_col is None:
        month_col = df.columns[0]

    meters: Dict[str, Dict[int, float]] = {}
    for col in df.columns:
        if col == month_col:
            continue
        base_name = str(col).split("[", 1)[0].strip()
        for _, row in df.iterrows():
            mm = _month_cell_to_int(row[month_col])
            if mm is None:
                continue
            try:
                val_j = float(row[col])
            except Exception:
                continue
            meters.setdefault(base_name, {})[mm] = val_j

    if not meters:
        raise RuntimeError(f"No meter columns parsed from {meter_csv.name}. Columns={list(df.columns)}")
    return meters


def read_sim_monthly_heating_kwh(out_dir: Path) -> Dict[int, float]:
    """Return { month -> heating_kWh } by summing the available heating meters."""
    meter_csvs = sorted(out_dir.glob("*Meter.csv"))
    if meter_csvs:
        meters = read_monthly_meter_j_from_meter_csv(meter_csvs[0])
    else:
        mtr = out_dir / "eplusout.mtr"
        if not mtr.exists():
            raise RuntimeError(f"No *Meter.csv and no eplusout.mtr in {out_dir}")
        meters = read_monthly_meter_j_from_mtr(mtr)

    monthly_j: Dict[int, float] = {}
    found_any = False
    for meter_name in HEATING_METERS_PRIORITY:
        key = next((k for k in meters if k.strip().lower() == meter_name.strip().lower()), None)
        if key is None:
            continue
        found_any = True
        for mm, val_j in meters[key].items():
            monthly_j[mm] = monthly_j.get(mm, 0.0) + float(val_j)

    if not found_any:
        raise RuntimeError(f"None of HEATING_METERS_PRIORITY found. Available: {list(meters.keys())}")

    return {mm: val_j / 3.6e6 for mm, val_j in monthly_j.items()}


# ============================================================
# Schedule parsing and writing
# Pattern assumed:
#   Through: mm/dd
#   For: AllDays
#   Until: 08:00 -> T_home
#   Until: 18:00 -> T_away
#   Until: 24:00 -> T_home
# ============================================================
def _find_schedule_compact(idf: IDF, sched_name: str):
    for s in idf.idfobjects.get("SCHEDULE:COMPACT", []):
        if getattr(s, "Name", "").strip().upper() == sched_name.upper():
            return s
    return None


def _delete_schedule_compact(idf: IDF, sched_name: str) -> None:
    seq = idf.idfobjects.get("SCHEDULE:COMPACT", [])
    for s in [s for s in seq if getattr(s, "Name", "").strip().upper() == sched_name.upper()]:
        seq.remove(s)


def ensure_schedule_type_limits(idf: IDF, name: str, lower, upper, numeric_type: str) -> None:
    for stl in idf.idfobjects.get("SCHEDULETYPELIMITS", []):
        if getattr(stl, "Name", "").strip().upper() == name.upper():
            return
    stl = idf.newidfobject("SCHEDULETYPELIMITS")
    stl.Name = name
    if lower is not None and hasattr(stl, "Lower_Limit_Value"):
        stl.Lower_Limit_Value = float(lower)
    if upper is not None and hasattr(stl, "Upper_Limit_Value"):
        stl.Upper_Limit_Value = float(upper)
    if hasattr(stl, "Numeric_Type"):
        stl.Numeric_Type = numeric_type


def read_monthly_home_away_from_schedule(idf: IDF, sched_name: str) -> Dict[int, Tuple[float, float]]:
    """Return { month -> (T_home, T_away) } for the months present."""
    sc = _find_schedule_compact(idf, sched_name)
    if sc is None:
        raise KeyError(f"Schedule:Compact not found: {sched_name}")

    fields = []
    i = 1
    while True:
        fn = f"Field_{i}"
        if not hasattr(sc, fn):
            break
        v = getattr(sc, fn)
        if v is None or str(v).strip() == "":
            break
        fields.append(str(v).strip())
        i += 1

    month_map: Dict[int, Tuple[float, float]] = {}
    mm = None
    j = 0
    while j < len(fields):
        token = fields[j]
        if token.lower().startswith("through:"):
            m = re.search(r"(\d{1,2})/(\d{1,2})", token)
            if m:
                mm = int(m.group(1))
            j += 1
            continue

        if token.lower().startswith("for:"):
            j += 1
            continue

        if token.lower().startswith("until:"):
            if j + 1 >= len(fields) or mm is None:
                j += 1
                continue
            t_str = token.split(":", 1)[1].strip()
            try:
                val = float(fields[j + 1])
            except Exception:
                j += 2
                continue

            if t_str.startswith("08"):
                home = val
                away = None
                home2 = None
                k = j + 2
                while k + 1 < len(fields):
                    if str(fields[k]).lower().startswith("through:"):
                        break
                    if str(fields[k]).lower().startswith("until:"):
                        t2 = str(fields[k]).split(":", 1)[1].strip()
                        try:
                            v2 = float(fields[k + 1])
                        except Exception:
                            k += 2
                            continue
                        if t2.startswith("18"):
                            away = v2
                        if t2.startswith("24"):
                            home2 = v2
                            break
                    k += 2
                if away is not None and home2 is not None:
                    month_map[mm] = (float(home), float(away))
            j += 2
            continue

        j += 1

    if not month_map:
        raise RuntimeError(f"Could not parse (home, away) from schedule: {sched_name}")
    return month_map


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def enforce_home_ge_away(th: float, ta: float) -> Tuple[float, float]:
    """Clip to the bounds and enforce T_home >= T_away + gap."""
    if th < ta + MIN_HOME_AWAY_GAP:
        th = ta + MIN_HOME_AWAY_GAP
    th = clamp(th, BOUNDS["t_home"][0], BOUNDS["t_home"][1])
    ta = clamp(ta, BOUNDS["t_away"][0], BOUNDS["t_away"][1])
    if th < ta + MIN_HOME_AWAY_GAP:
        th = clamp(ta + MIN_HOME_AWAY_GAP, BOUNDS["t_home"][0], BOUNDS["t_home"][1])
    return th, ta


def write_monthly_setpoint_schedule(idf: IDF, sched_name: str, type_limits: str,
                                    month_to_home_away: Dict[int, Tuple[float, float]]) -> None:
    """Rebuild the Schedule:Compact for all 12 months, carrying forward when a
    month is missing."""
    ensure_schedule_type_limits(idf, type_limits, None, None, "Continuous")
    _delete_schedule_compact(idf, sched_name)

    sc = idf.newidfobject("SCHEDULE:COMPACT")
    sc.Name = sched_name
    sc.Schedule_Type_Limits_Name = type_limits

    provided = sorted(month_to_home_away.keys())
    if not provided:
        raise ValueError("month_to_home_away is empty")

    last_home, last_away = month_to_home_away[provided[0]]
    field_i = 1

    for mm in range(1, 13):
        if mm in month_to_home_away:
            last_home, last_away = month_to_home_away[mm]
        last_home, last_away = enforce_home_ge_away(last_home, last_away)

        setattr(sc, f"Field_{field_i}", f"Through: {mm:02d}/{MONTH_END_DAY[mm]:02d}"); field_i += 1
        setattr(sc, f"Field_{field_i}", "For: AllDays"); field_i += 1
        setattr(sc, f"Field_{field_i}", "Until: 08:00"); field_i += 1
        setattr(sc, f"Field_{field_i}", float(last_home)); field_i += 1
        setattr(sc, f"Field_{field_i}", "Until: 18:00"); field_i += 1
        setattr(sc, f"Field_{field_i}", float(last_away)); field_i += 1
        setattr(sc, f"Field_{field_i}", "Until: 24:00"); field_i += 1
        setattr(sc, f"Field_{field_i}", float(last_home)); field_i += 1


# ============================================================
# Evaluation
# ============================================================
def evaluate_schedule(month_to_home_away: Dict[int, Tuple[float, float]],
                      energyplus_exe: Path,
                      eval_dir: Path,
                      *,
                      tag: str) -> Tuple[float, float, Dict[int, float], Dict[int, float]]:
    """Return (CVRMSE, NMBE, sim_used_kwh, residual_by_month_kwh).

    residual = simulated - measured.
    """
    run_dir = eval_dir
    run_idf = run_dir / "in.idf"
    out_dir = run_dir / "out"
    run_dir.mkdir(parents=True, exist_ok=True)

    IDF.setiddname(str(IDD_PATH))
    idf = IDF(str(INPUT_IDF_PATH))

    ensure_schedule_type_limits(idf, "Any Number", None, None, "Continuous")

    write_monthly_setpoint_schedule(idf, RADIANT_SETPOINT_SCHED_NAME, "Any Number", month_to_home_away)
    write_monthly_setpoint_schedule(idf, AIR_HEATING_SETPOINT_SCHED_NAME, "Any Number", month_to_home_away)

    idf.saveas(str(run_idf))
    run_energyplus(energyplus_exe, run_idf, EPW_PATH, out_dir)

    sim_heat = read_sim_monthly_heating_kwh(out_dir)
    sim_kwh = heat_to_fuel(sim_heat, BOILER_EFF)
    cvr, nb, sim_used = compute_metrics(MEASURED_MONTHLY_KWH, sim_kwh)

    residual = {}
    for mm in MEASURED_MONTHS:
        if mm in sim_used:
            residual[mm] = float(sim_used[mm]) - float(MEASURED_MONTHLY_KWH[str(mm)])

    return cvr, nb, sim_used, residual


# ============================================================
# Residual-guided update rule
# ============================================================
def month_update_delta(residual_kwh: float, measured_kwh: float,
                       step_home: float, step_away: float) -> Tuple[float, float]:
    """Heuristic month update.

    residual > 0 : simulated heating too high -> reduce setpoints
    residual < 0 : simulated heating too low  -> raise setpoints
    """
    if measured_kwh <= 0:
        measured_kwh = 1.0

    if MONTH_WEIGHT_MODE == "relative":
        r = residual_kwh / measured_kwh
        mag = min(1.8, max(0.25, abs(r) * 2.2))
    else:
        mag = min(1.8, max(0.25, abs(residual_kwh) / 1000.0))

    if residual_kwh < 0:
        d_home = +1.45 * step_home * mag
        d_away = +1.90 * step_away * mag
        if residual_kwh < -400:
            d_home *= 1.20
            d_away *= 1.35
        if residual_kwh < -900:
            d_home *= 1.10
            d_away *= 1.20
    elif residual_kwh > 0:
        d_home = -0.90 * step_home * mag
        d_away = -1.25 * step_away * mag
        if residual_kwh > 400:
            d_home *= 1.10
            d_away *= 1.15
        if residual_kwh > 900:
            d_home *= 1.08
            d_away *= 1.12
    else:
        d_home = 0.0
        d_away = 0.0

    d_home = clamp(d_home, -CLAMP_MAX_PER_ITER, CLAMP_MAX_PER_ITER)
    d_away = clamp(d_away, -CLAMP_MAX_PER_ITER, CLAMP_MAX_PER_ITER)
    return d_home, d_away


def refine_months_greedily(
    base_map: Dict[int, Tuple[float, float]],
    residual: Dict[int, float],
    measured_kwh: Dict[str, float],
    energyplus_exe: Path,
    work_dir: Path,
    current_best_cvr: float,
    current_best_abs_nb: float,
    step_home: float,
    step_away: float,
) -> Tuple[Dict[int, Tuple[float, float]], float, float, Dict[int, float], bool]:
    """Adjust one month at a time, accepting only global improvements."""
    improved_any = False
    best_map = dict(base_map)
    best_cvr = current_best_cvr
    best_nb_abs = current_best_abs_nb
    best_resid = dict(residual)

    month_order = sorted(MEASURED_MONTHS,
                         key=lambda mm: abs(residual.get(mm, 0.0)), reverse=True)
    print("[HEUR] Month priority by |residual|:", month_order)

    for mm in month_order:
        th, ta = best_map[mm]
        r = residual.get(mm, 0.0)
        meas = float(measured_kwh[str(mm)])

        dth, dta = month_update_delta(r, meas, step_home, step_away)
        print(f"[HEUR][Month {mm:02d}] resid={r:+.1f} kWh  "
              f"old=({th:.2f}, {ta:.2f})  delta=({dth:+.2f}, {dta:+.2f})")

        trial_map = dict(best_map)
        th_new = clamp(th + dth, BOUNDS["t_home"][0], BOUNDS["t_home"][1])
        ta_new = clamp(ta + dta, BOUNDS["t_away"][0], BOUNDS["t_away"][1])
        th_new, ta_new = enforce_home_ge_away(th_new, ta_new)
        trial_map[mm] = (th_new, ta_new)

        run_dir = work_dir / f"month_refine_m{mm:02d}"
        try:
            cvr, nb, _sim_used, resid_new = evaluate_schedule(
                trial_map, energyplus_exe, run_dir, tag=f"month_refine_m{mm:02d}")
        except Exception:
            continue

        better = (
            (cvr < best_cvr + 0.30 and abs(nb) <= best_nb_abs + 0.20) or
            (cvr < best_cvr - 0.03)
        )

        if better:
            best_map = trial_map
            best_cvr = cvr
            best_nb_abs = abs(nb)
            best_resid = resid_new
            improved_any = True

    return best_map, best_cvr, best_nb_abs, best_resid, improved_any


def residual_rescue_pass(
    base_map: Dict[int, Tuple[float, float]],
    residual: Dict[int, float],
    measured_kwh: Dict[str, float],
    energyplus_exe: Path,
    work_dir: Path,
    current_best_cvr: float,
    current_best_nb: float,
) -> Tuple[Dict[int, Tuple[float, float]], float, float, Dict[int, float], bool]:
    """Targeted setback-first corrections for months with large residuals."""
    best_map = dict(base_map)
    best_cvr = current_best_cvr
    best_nb = current_best_nb
    best_resid = dict(residual)
    improved_any = False

    month_order = sorted(MEASURED_MONTHS,
                         key=lambda mm: abs(residual.get(mm, 0.0)), reverse=True)

    for mm in month_order:
        r = residual.get(mm, 0.0)
        th, ta = best_map[mm]
        candidates = []

        if r < -150:
            # simulated too low -> increase the heating effect
            candidates += [
                (th, clamp(ta + 1.0, BOUNDS["t_away"][0], BOUNDS["t_away"][1])),
                (th, clamp(ta + 2.0, BOUNDS["t_away"][0], BOUNDS["t_away"][1])),
                (th, clamp(ta + 3.0, BOUNDS["t_away"][0], BOUNDS["t_away"][1])),
                (clamp(th + 0.5, BOUNDS["t_home"][0], BOUNDS["t_home"][1]), ta),
            ]
        elif r > 150:
            # simulated too high -> decrease the heating effect
            candidates += [
                (th, clamp(ta - 1.0, BOUNDS["t_away"][0], BOUNDS["t_away"][1])),
                (th, clamp(ta - 2.0, BOUNDS["t_away"][0], BOUNDS["t_away"][1])),
                (clamp(th - 0.5, BOUNDS["t_home"][0], BOUNDS["t_home"][1]), ta),
            ]

        local_best = None
        local_best_score = None

        for idx, (th_try, ta_try) in enumerate(candidates, start=1):
            th_try, ta_try = enforce_home_ge_away(th_try, ta_try)
            trial_map = dict(best_map)
            trial_map[mm] = (th_try, ta_try)

            run_dir = work_dir / f"rescue_m{mm:02d}_{idx}"
            try:
                cvr, nb, _sim_used, resid_new = evaluate_schedule(
                    trial_map, energyplus_exe, run_dir, tag=f"rescue_m{mm:02d}_{idx}")
            except Exception:
                continue

            score = (cvr, abs(nb))
            if local_best_score is None or score < local_best_score:
                local_best = (trial_map, cvr, nb, resid_new)
                local_best_score = score

        if local_best is not None:
            trial_map, cvr, nb, resid_new = local_best
            better = (
                (cvr < best_cvr) or
                (math.isclose(cvr, best_cvr, rel_tol=1e-6) and abs(nb) < abs(best_nb))
            )
            if better:
                best_map = trial_map
                best_cvr = cvr
                best_nb = nb
                best_resid = resid_new
                improved_any = True
                print(f"[RESCUE] accepted month {mm:02d}: CVRMSE={cvr:.3f}% NMBE={nb:.3f}%")

    return best_map, best_cvr, best_nb, best_resid, improved_any


# ============================================================
# Local NSGA-II polish around the current best
# ============================================================
class LocalMonthlyTsetProblem(ElementwiseProblem):
    def __init__(self,
                 base_month_to_home_away: Dict[int, Tuple[float, float]],
                 measured_months: List[int],
                 energyplus_exe: Path,
                 work_dir: Path,
                 log_csv: Path):
        self.measured_months = measured_months
        self.energyplus_exe = energyplus_exe
        self.work_dir = work_dir
        self.base = base_month_to_home_away
        self.log_csv = log_csv
        self.eval_counter = 0

        n_var = 2 * len(measured_months)
        xl = np.zeros(n_var, dtype=float)
        xu = np.zeros(n_var, dtype=float)

        for i, mm in enumerate(measured_months):
            th0, ta0 = base_month_to_home_away.get(
                mm, base_month_to_home_away[min(base_month_to_home_away.keys())])
            xl[2 * i]     = clamp(th0 - NSGA_LOCAL_SPAN_HOME, *BOUNDS["t_home"])
            xu[2 * i]     = clamp(th0 + NSGA_LOCAL_SPAN_HOME, *BOUNDS["t_home"])
            xl[2 * i + 1] = clamp(ta0 - NSGA_LOCAL_SPAN_AWAY, *BOUNDS["t_away"])
            xu[2 * i + 1] = clamp(ta0 + NSGA_LOCAL_SPAN_AWAY, *BOUNDS["t_away"])

        super().__init__(n_var=n_var, n_obj=2, xl=xl, xu=xu)

        if not self.log_csv.exists():
            self.log_csv.parent.mkdir(parents=True, exist_ok=True)
            with self.log_csv.open("w", newline="", encoding="utf-8") as f:
                header = ["eval_id"]
                for mm in measured_months:
                    header += [f"t_home_m{mm:02d}", f"t_away_m{mm:02d}"]
                header += ["CVRMSE_%", "NMBE_%", "absNMBE_%", "runtime_s"]
                csv.writer(f).writerow(header)

    def _evaluate(self, x, out, *args, **kwargs):
        self.eval_counter += 1
        eval_id = self.eval_counter

        month_to_home_away = dict(self.base)
        for i, mm in enumerate(self.measured_months):
            th = float(x[2 * i])
            ta = float(x[2 * i + 1])
            th, ta = enforce_home_ge_away(th, ta)
            month_to_home_away[mm] = (th, ta)

        run_dir = self.work_dir / f"run_{eval_id:05d}"
        t0 = time.time()

        try:
            cvr, nb, _sim_used, _resid = evaluate_schedule(
                month_to_home_away, self.energyplus_exe, run_dir, tag=f"nsga_{eval_id:05d}")
            absnb = abs(nb)
        except Exception:
            cvr, nb, absnb = 1e6, 1e6, 1e6

        runtime = time.time() - t0
        out["F"] = np.array([cvr, absnb], dtype=float)

        with self.log_csv.open("a", newline="", encoding="utf-8") as f:
            row = [eval_id]
            for mm in self.measured_months:
                th, ta = month_to_home_away[mm]
                row += [th, ta]
            row += [cvr, nb, absnb, runtime]
            csv.writer(f).writerow(row)
            f.flush()

        if CLEANUP_EACH_RUN:
            failed = (cvr >= 1e5)
            if failed and KEEP_FAILED_RUNS:
                return
            shutil.rmtree(run_dir, ignore_errors=True)


def pick_best_from_log(csv_path: Path) -> Tuple[Dict[int, Tuple[float, float]], float, float]:
    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
    if df.empty:
        raise RuntimeError("NSGA polish log is empty or unreadable.")
    for col in ("CVRMSE_%", "absNMBE_%", "NMBE_%"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["CVRMSE_%", "absNMBE_%"])
    df = df.sort_values(["CVRMSE_%", "absNMBE_%"], ascending=[True, True]).reset_index(drop=True)
    best = df.iloc[0].to_dict()

    out_map: Dict[int, Tuple[float, float]] = {}
    for mm in MEASURED_MONTHS:
        thk = f"t_home_m{mm:02d}"
        tak = f"t_away_m{mm:02d}"
        if thk in best and tak in best:
            out_map[mm] = enforce_home_ge_away(float(best[thk]), float(best[tak]))

    return out_map, float(best["CVRMSE_%"]), float(best["NMBE_%"])


# ============================================================
# Logging
# ============================================================
def append_iter_log(log_csv: Path, iter_id: int,
                    month_to_home_away: Dict[int, Tuple[float, float]],
                    cvr: float, nb: float, residual: Dict[int, float],
                    runtime_s: float) -> None:
    first = not log_csv.exists()
    log_csv.parent.mkdir(parents=True, exist_ok=True)

    with log_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if first:
            header = ["iter_id", "CVRMSE_%", "NMBE_%", "runtime_s"]
            for mm in MEASURED_MONTHS:
                header += [f"t_home_m{mm:02d}", f"t_away_m{mm:02d}", f"resid_kwh_m{mm:02d}"]
            w.writerow(header)

        row = [iter_id, cvr, nb, runtime_s]
        for mm in MEASURED_MONTHS:
            th, ta = month_to_home_away[mm]
            row += [th, ta, residual.get(mm, float("nan"))]
        w.writerow(row)
        f.flush()


# ============================================================
# Main workflow
# ============================================================
def main() -> dict:
    if not IDD_PATH.exists():
        raise FileNotFoundError(f"IDD not found: {IDD_PATH.resolve()}")
    if not EPW_PATH.exists():
        raise FileNotFoundError(f"EPW not found: {EPW_PATH.resolve()}")
    if not INPUT_IDF_PATH.exists():
        raise FileNotFoundError(f"IDF not found: {INPUT_IDF_PATH.resolve()}")

    if CLEAN_WORKSPACE_EACH_START:
        reset_workspace(WORK_DIR)
    else:
        WORK_DIR.mkdir(parents=True, exist_ok=True)

    energyplus_exe = guess_energyplus_exe(IDD_PATH)

    echo_inputs()
    print(f"[INFO] EnergyPlus exe : {energyplus_exe}")
    print(f"[INFO] Work dir       : {WORK_DIR.resolve()}")

    # ---- Starting point: the schedule already in the input IDF
    IDF.setiddname(str(IDD_PATH))
    idf0 = IDF(str(INPUT_IDF_PATH))

    try:
        month_to_home_away = read_monthly_home_away_from_schedule(idf0, RADIANT_SETPOINT_SCHED_NAME)
        src = RADIANT_SETPOINT_SCHED_NAME
    except Exception:
        month_to_home_away = read_monthly_home_away_from_schedule(idf0, AIR_HEATING_SETPOINT_SCHED_NAME)
        src = AIR_HEATING_SETPOINT_SCHED_NAME

    provided = sorted(month_to_home_away.keys())
    last_home, last_away = month_to_home_away[provided[0]]
    for mm in range(1, 13):
        if mm in month_to_home_away:
            last_home, last_away = month_to_home_away[mm]
        last_home, last_away = enforce_home_ge_away(last_home, last_away)
        month_to_home_away[mm] = (last_home, last_away)

    print(f"[INFO] Loaded starting monthly setpoints from '{src}'")
    for mm in MEASURED_MONTHS:
        th, ta = month_to_home_away[mm]
        print(f"  Month {mm:02d}: T_home={th:.2f}, T_away={ta:.2f}")

    # ---- Baseline
    baseline_dir = WORK_DIR / "baseline"
    t0 = time.time()
    cvr_best, nb_best, sim_used, resid = evaluate_schedule(
        month_to_home_away, energyplus_exe, baseline_dir, tag="baseline")
    t_baseline = time.time() - t0

    print(f"\n=== Baseline (schedules from {INPUT_IDF_PATH.name}) ===")
    print(f"CVRMSE: {cvr_best:.3f} %")
    print(f"NMBE  : {nb_best:.3f} %")
    print(f"[INFO] Baseline runtime: {t_baseline:.1f} s")

    best_map = dict(month_to_home_away)
    best_score = (cvr_best, abs(nb_best))
    heur_log = WORK_DIR / "heuristic_log.csv"
    append_iter_log(heur_log, 0, best_map, cvr_best, nb_best, resid, t_baseline)

    # ---- Stage 3: residual-guided heuristic loop
    step_home = BASE_STEP_HOME
    step_away = BASE_STEP_AWAY

    for it in range(1, MAX_OUTER_ITERS + 1):
        t1 = time.time()

        proposal, cvr, absnb, resid_new, improved_any = refine_months_greedily(
            base_map=best_map,
            residual=resid,
            measured_kwh=MEASURED_MONTHLY_KWH,
            energyplus_exe=energyplus_exe,
            work_dir=WORK_DIR / f"heur_{it:02d}",
            current_best_cvr=cvr_best,
            current_best_abs_nb=abs(nb_best),
            step_home=step_home,
            step_away=step_away,
        )

        if improved_any:
            confirm_dir = WORK_DIR / f"heur_{it:02d}_confirm"
            try:
                cvr, nb, sim_used, resid_new = evaluate_schedule(
                    proposal, energyplus_exe, confirm_dir, tag=f"heur_{it:02d}_confirm")
            except Exception:
                cvr, nb = 1e6, 1e6
                sim_used, resid_new = {}, {}
        else:
            cvr, nb = cvr_best, nb_best
            sim_used, resid_new = {}, resid

        runtime = time.time() - t1
        append_iter_log(heur_log, it, proposal, cvr, nb, resid_new, runtime)

        improved = (cvr < best_score[0]) or (
            math.isclose(cvr, best_score[0], rel_tol=1e-6) and abs(nb) < best_score[1]
        )

        print(f"\n[HEUR] Iter {it:02d}: CVRMSE={cvr:.3f}%  NMBE={nb:.3f}%  "
              f"step_home={step_home:.2f} step_away={step_away:.2f}  improved={improved}")

        if improved:
            best_map = proposal
            resid = resid_new
            best_score = (cvr, abs(nb))
            cvr_best, nb_best = cvr, nb
            print("[HEUR] -> accepted as current best")
        else:
            print("[HEUR] -> rejected (keep best)")

        step_home *= STEP_DECAY
        step_away *= STEP_DECAY

        if cvr_best <= 15.0 and abs(nb_best) <= 5.0:
            print("[HEUR] Early stop: metrics already strong.")
            break

    # ---- Residual rescue pass
    print("\n[RESCUE] Starting targeted residual rescue pass...")
    rescue_map, rescue_cvr, rescue_nb, rescue_resid, rescue_improved = residual_rescue_pass(
        base_map=best_map,
        residual=resid,
        measured_kwh=MEASURED_MONTHLY_KWH,
        energyplus_exe=energyplus_exe,
        work_dir=WORK_DIR / "rescue_pass",
        current_best_cvr=cvr_best,
        current_best_nb=nb_best,
    )

    if rescue_improved:
        best_map = rescue_map
        cvr_best = rescue_cvr
        nb_best = rescue_nb
        resid = rescue_resid
        best_score = (cvr_best, abs(nb_best))
        print(f"[RESCUE] accepted improved solution: "
              f"CVRMSE={cvr_best:.3f}% NMBE={nb_best:.3f}%")
    else:
        print("[RESCUE] no improvement found")

    # ---- Stage 4: local NSGA-II polish
    if ENABLE_NSAA_POLISH:
        print("\n[NSGA] Starting local NSGA-II polish around the heuristic best...")
        nsga_dir = WORK_DIR / "nsga_polish"
        nsga_dir.mkdir(parents=True, exist_ok=True)
        nsga_log = nsga_dir / "eval_log.csv"

        problem = LocalMonthlyTsetProblem(
            base_month_to_home_away=best_map,
            measured_months=MEASURED_MONTHS,
            energyplus_exe=energyplus_exe,
            work_dir=nsga_dir,
            log_csv=nsga_log,
        )

        minimize(problem, NSGA2(pop_size=NSGA_POP), get_termination("n_gen", NSGA_GEN),
                 seed=NSGA_SEED, save_history=False, verbose=True)

        try:
            polish_map, cvr_p, nb_p = pick_best_from_log(nsga_log)
            merged = dict(best_map)
            for mm, (th, ta) in polish_map.items():
                merged[mm] = enforce_home_ge_away(th, ta)

            confirm_dir = WORK_DIR / "nsga_best_confirm"
            t2 = time.time()
            cvr_c, nb_c, sim_used_c, resid_c = evaluate_schedule(
                merged, energyplus_exe, confirm_dir, tag="nsga_best_confirm")
            tconf = time.time() - t2

            print("\n[NSGA] Best (confirmed) from the local polish:")
            print(f"CVRMSE={cvr_c:.3f}%  NMBE={nb_c:.3f}%  runtime={tconf:.1f}s")
            if (cvr_c < cvr_best) or (math.isclose(cvr_c, cvr_best, rel_tol=1e-6)
                                      and abs(nb_c) < abs(nb_best)):
                best_map = merged
                cvr_best, nb_best = cvr_c, nb_c
                resid = resid_c
                print("[NSGA] -> accepted polish improvement")
            else:
                print("[NSGA] -> no improvement over the heuristic best")
        except Exception as e:
            print(f"[NSGA] polish log parse or confirmation failed: {e}")

    # ---- Write the final IDF
    IDF.setiddname(str(IDD_PATH))
    idf_final = IDF(str(INPUT_IDF_PATH))

    ensure_schedule_type_limits(idf_final, "Any Number", None, None, "Continuous")
    write_monthly_setpoint_schedule(idf_final, RADIANT_SETPOINT_SCHED_NAME, "Any Number", best_map)
    write_monthly_setpoint_schedule(idf_final, AIR_HEATING_SETPOINT_SCHED_NAME, "Any Number", best_map)

    OUTPUT_IDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    idf_final.saveas(str(OUTPUT_IDF_PATH))

    print("\n=== FINAL (best found) ===")
    for mm in MEASURED_MONTHS:
        th, ta = best_map[mm]
        print(f"  Month {mm:02d}: T_home={th:.2f} degC, T_away={ta:.2f} degC  "
              f"(resid_kWh={resid.get(mm, float('nan')):+.1f})")

    print(f"CVRMSE = {cvr_best:.3f} %")
    print(f"NMBE   = {nb_best:.3f} %")
    print(f"FINAL_METRICS_JSON={{\"CVRMSE\": {cvr_best:.6f}, \"NMBE\": {nb_best:.6f}}}")
    print(f"[INFO] Heuristic log  : {heur_log.resolve()}")
    if ENABLE_NSAA_POLISH:
        print(f"[INFO] NSGA polish log: {(WORK_DIR / 'nsga_polish' / 'eval_log.csv').resolve()}")
    print(f"[OK] Calibrated IDF saved to: {OUTPUT_IDF_PATH.resolve()}")

    # ── Verification report and chart ────────────────────────────────────
    try:
        from cali_verification_report import run_verification
        _vdir = WORK_DIR / "best_verify"
        _vdir.mkdir(exist_ok=True)
        run_energyplus(energyplus_exe, OUTPUT_IDF_PATH, EPW_PATH, _vdir / "out")
        _heat = read_sim_monthly_heating_kwh(_vdir / "out")
        run_verification(
            stage_name="Stage 3/4 – Monthly setpoint refinement (final)",
            idf_path=str(OUTPUT_IDF_PATH),
            epw_path=str(EPW_PATH),
            measured=MEASURED_MONTHLY_KWH,
            simulated_heat=_heat,
            boiler_eff=BOILER_EFF,
            chart_out=OUTPUT_IDF_PATH.parent / "stage4_comparison.png")
    except Exception as _e:
        print(f"[WARN] Verification report failed: {_e}")
    # ─────────────────────────────────────────────────────────────────────

    # Remove baseline outputs to save disk; keep the logs and the final IDF
    try:
        shutil.rmtree(WORK_DIR / "baseline", ignore_errors=True)
    except Exception:
        pass

    return {
        "metrics": {
            "CVRMSE": float(cvr_best),
            "NMBE": float(nb_best),
        },
        "idf_path": str(OUTPUT_IDF_PATH.resolve()),
    }


if __name__ == "__main__":
    main()
