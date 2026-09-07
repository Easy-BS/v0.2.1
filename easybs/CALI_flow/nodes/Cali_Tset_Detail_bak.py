# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 18:20:07 2026

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

MEASURED_MONTHLY_KWH = _RUNTIME.get("measured_monthly_kwh", {
    "1": 5530.4167,
    "2": 5716.1806,
    "3": 5002.9167,
    "4": 3540.625,
    "10": 126.875,
    "11": 1469.7222,
    "12": 3137.6389,
})
IDD_PATH = Path(_RUNTIME.get("idd_path", r"C:/EnergyPlusV8-9-0/Energy+.idd"))
EPW_PATH = Path(_RUNTIME.get("epw_path", r"C:\EnergyPlusV8-9-0\WeatherData\KOR_INCH'ON_IWEC.epw"))
INPUT_IDF_PATH = Path(_RUNTIME.get("idf_path", "./Calibration/After1_Cali_RFH.idf"))
OUTPUT_IDF_PATH = Path("./Calibration/After2_Cali_RFH.idf")

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

# ============================================================
# Reasonable bounds (expanded, but still defensible)

#
# Typical residential:
# - Home heating setpoint: 18–26 °C (upper > 24 for old/poor insulation comfort compensation)
# - Away setback: 8–22 °C (avoid unrealistic freezing; allow mild setback if occupants keep warm)
#
# Enforce: T_home >= T_away (hard constraint)
# ============================================================
BOUNDS = {
    "t_home": (17.0, 27.0),
    "t_away": (8.0, 22.0),
}

# Hard constraint guard: minimum gap (optional)
MIN_HOME_AWAY_GAP = 0.0  # set to e.g., 0.5 if you want a strict separation

# ============================================================
# Heuristic tuning controls
# ============================================================
MAX_OUTER_ITERS = 18          # number of heuristic rounds
BASE_STEP_HOME = 0.7          # °C initial step for T_home
BASE_STEP_AWAY = 0.8          # °C initial step for T_away
STEP_DECAY = 0.96             # reduce step sizes each outer iter
MONTH_WEIGHT_MODE = "relative"  # "relative" or "absolute"
CLAMP_MAX_PER_ITER = 2.0      # max |delta| allowed per month per outer iter (°C)

# Optional NSGA-II local polish after heuristic improvement
ENABLE_NSAA_POLISH = False
NSGA_POP = 32
NSGA_GEN = 12
NSGA_SEED = 42
NSGA_LOCAL_SPAN_HOME = 2.5    # ± range around current best (°C)
NSGA_LOCAL_SPAN_AWAY = 4.0

# ============================================================
# Helpers: months
# ============================================================
MEASURED_MONTHS: List[int] = sorted(int(k) for k in MEASURED_MONTHLY_KWH.keys())

MONTH_END_DAY = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}


# ============================================================
# Workspace management
# ============================================================
def reset_workspace(work_dir: Path) -> None:
    if work_dir.exists():
        shutil.rmtree(work_dir)
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


def align_months(measured_kwh: Dict[str, float], simulated_kwh: Dict[int, float]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    months = sorted(int(k) for k in measured_kwh.keys())
    meas, sim, kept = [], [], []
    for mm in months:
        if mm in simulated_kwh:
            meas.append(float(measured_kwh[str(mm)]))
            sim.append(float(simulated_kwh[mm]))
            kept.append(mm)
    if len(meas) < 2:
        raise RuntimeError(f"Not enough overlapping months. Measured={months}, SimAvailable={sorted(simulated_kwh.keys())}")
    return np.array(meas), np.array(sim), kept


def compute_metrics(measured_kwh: Dict[str, float], simulated_kwh: Dict[int, float]) -> Tuple[float, float, Dict[int, float]]:
    meas, sim, months = align_months(measured_kwh, simulated_kwh)
    cvr = cvrmse_percent(meas, sim)
    nb = nmbe_percent(meas, sim)
    sim_used = {m: float(simulated_kwh[m]) for m in months}
    return cvr, nb, sim_used


# ============================================================
# EnergyPlus runner + parsing monthly meters
# ============================================================
def guess_energyplus_exe(idd_path: Path) -> Path:
    candidate = idd_path.parent / "energyplus.exe"
    return candidate if candidate.exists() else Path("energyplus")


def run_energyplus(energyplus_exe: Path, idf_path: Path, epw_path: Path, out_dir: Path, timeout_s: int = 3600) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [str(energyplus_exe), "-w", str(epw_path), "-d", str(out_dir), str(idf_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        msg = (proc.stdout[-4000:] if proc.stdout else "") + "\n" + (proc.stderr[-4000:] if proc.stderr else "")
        raise RuntimeError(f"EnergyPlus failed (code {proc.returncode}). Tail output:\n{msg}")


def read_monthly_meter_j_from_mtr(mtr_path: Path) -> Dict[str, Dict[int, float]]:
    """
    Parse EnergyPlus eplusout.mtr for Monthly meters.
    Returns: { meter_name: { month:int -> value_J:float } }
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
    data_lines = lines[end_idx + 1 :]

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
        name_with_units = parts[2]
        meter_name = name_with_units.split("[", 1)[0].strip()
        monthly_idx_to_name[idx] = meter_name

    if not monthly_idx_to_name:
        raise RuntimeError("No Monthly meters detected in eplusout.mtr dictionary (check Output:Meter Monthly).")

    meters_monthly: Dict[str, Dict[int, float]] = {n: {} for n in set(monthly_idx_to_name.values())}
    current_month: Optional[int] = None

    for line in data_lines:
        s = line.strip()
        if not s:
            continue
        parts = [p.strip() for p in s.split(",")]

        # Monthly header: "4, ..., <month>"
        if parts[0] == "4" and len(parts) >= 3:
            try:
                mm = int(parts[2])
                current_month = mm if 1 <= mm <= 12 else None
            except Exception:
                current_month = None
            continue

        # Data line: "<idx>, <value>, ..."
        try:
            idx = int(parts[0])
        except Exception:
            continue

        if idx in monthly_idx_to_name and current_month is not None and len(parts) >= 2:
            try:
                val_j = float(parts[1])
            except Exception:
                continue
            mname = monthly_idx_to_name[idx]
            meters_monthly.setdefault(mname, {})[current_month] = val_j

    meters_monthly = {k: v for k, v in meters_monthly.items() if v}
    if not meters_monthly:
        raise RuntimeError("Monthly meter indices found, but no monthly values parsed from data section.")
    return meters_monthly


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


def read_monthly_meter_j_from_meter_csv(meter_csv: Path) -> dict[str, dict[int, float]]:
    """
    Parse *Meter.csv like:
      Date/Time    DistrictHeating:Facility [J](Monthly)
      January      6966423054
    """
    df = pd.read_csv(meter_csv)
    month_col = df.columns[0]
    meters: dict[str, dict[int, float]] = {}
    for col in df.columns[1:]:
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
    """
    Returns {month:int -> heating_kwh} by summing available meters.
    Priority:
      1) *Meter.csv
      2) eplusout.mtr
    """
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
        key = next((k for k in meters.keys() if k.strip().lower() == meter_name.strip().lower()), None)
        if key is None:
            continue
        found_any = True
        for mm, val_j in meters[key].items():
            monthly_j[mm] = monthly_j.get(mm, 0.0) + float(val_j)

    if not found_any:
        raise RuntimeError(f"None of HEATING_METERS_PRIORITY found. Available: {list(meters.keys())}")

    return {mm: val_j / 3.6e6 for mm, val_j in monthly_j.items()}  # J -> kWh


# ============================================================
# Schedule parsing / writing (Schedule:Compact monthly setpoints)
# Pattern assumed (your snippet):
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
    to_del = [s for s in seq if getattr(s, "Name", "").strip().upper() == sched_name.upper()]
    for s in to_del:
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
    """
    Returns {month -> (T_home, T_away)} for months 1..12 when present.
    If schedule doesn't exist or parsing fails, raises.
    """
    sc = _find_schedule_compact(idf, sched_name)
    if sc is None:
        raise KeyError(f"Schedule:Compact not found: {sched_name}")

    # Read Field_1..Field_n
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
            # "Through: 01/31"
            m = re.search(r"(\d{1,2})/(\d{1,2})", token)
            if m:
                mm = int(m.group(1))
            j += 1
            continue

        if token.lower().startswith("for:"):
            j += 1
            continue

        # Expect sequence:
        # Until: 08:00, <home>
        # Until: 18:00, <away>
        # Until: 24:00, <home>
        if token.lower().startswith("until:"):
            # Next field is numeric
            if j + 1 >= len(fields) or mm is None:
                j += 1
                continue
            t_str = token.split(":", 1)[1].strip()
            try:
                val = float(fields[j + 1])
            except Exception:
                j += 2
                continue

            # Capture by time
            if t_str.startswith("08"):
                home = val
                # seek 18 and 24 next
                # simple scan ahead
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
    # enforce th >= ta + gap
    if th < ta + MIN_HOME_AWAY_GAP:
        th = ta + MIN_HOME_AWAY_GAP
    # also clamp again to avoid pushing out of bounds in later usage
    th = clamp(th, BOUNDS["t_home"][0], BOUNDS["t_home"][1])
    ta = clamp(ta, BOUNDS["t_away"][0], BOUNDS["t_away"][1])
    # re-enforce after clamping
    if th < ta + MIN_HOME_AWAY_GAP:
        th = clamp(ta + MIN_HOME_AWAY_GAP, BOUNDS["t_home"][0], BOUNDS["t_home"][1])
    return th, ta


def write_monthly_setpoint_schedule(
    idf: IDF,
    sched_name: str,
    type_limits: str,
    month_to_home_away: Dict[int, Tuple[float, float]],
) -> None:
    """
    Rebuild the Schedule:Compact for 12 months, carrying forward if a month missing.
    """
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
# Evaluation runner for a given month->(home, away)
# ============================================================
def evaluate_schedule(month_to_home_away: Dict[int, Tuple[float, float]],
                      energyplus_exe: Path,
                      eval_dir: Path,
                      *,
                      tag: str) -> Tuple[float, float, Dict[int, float], Dict[int, float]]:
    """
    Returns: (CVRMSE, NMBE, sim_used_kwh, residual_by_month_kwh)
    residual = sim - meas
    """
    run_dir = eval_dir
    run_idf = run_dir / "in.idf"
    out_dir = run_dir / "out"
    run_dir.mkdir(parents=True, exist_ok=True)

    IDF.setiddname(str(IDD_PATH))
    idf = IDF(str(INPUT_IDF_PATH))

    ensure_schedule_type_limits(idf, "Any Number", None, None, "Continuous")

    # Apply to both schedules (recommended to keep consistent)
    write_monthly_setpoint_schedule(idf, RADIANT_SETPOINT_SCHED_NAME, "Any Number", month_to_home_away)
    write_monthly_setpoint_schedule(idf, AIR_HEATING_SETPOINT_SCHED_NAME, "Any Number", month_to_home_away)

    idf.saveas(str(run_idf))
    run_energyplus(energyplus_exe, run_idf, EPW_PATH, out_dir)

    sim_kwh = read_sim_monthly_heating_kwh(out_dir)
    cvr, nb, sim_used = compute_metrics(MEASURED_MONTHLY_KWH, sim_kwh)

    residual = {}
    for mm in MEASURED_MONTHS:
        if mm in sim_used:
            residual[mm] = float(sim_used[mm]) - float(MEASURED_MONTHLY_KWH[str(mm)])

    return cvr, nb, sim_used, residual


# ============================================================
# Heuristic "inspired" update rule
# ============================================================
def month_update_delta(
    residual_kwh: float,
    measured_kwh: float,
    step_home: float,
    step_away: float
) -> Tuple[float, float]:
    """
    Heuristic month update.

    residual_kwh = sim - measured
      residual > 0 : simulated heating is too high  -> reduce setpoints
      residual < 0 : simulated heating is too low   -> increase setpoints
    """
    if measured_kwh <= 0:
        measured_kwh = 1.0

    if MONTH_WEIGHT_MODE == "relative":
        r = residual_kwh / measured_kwh
        mag = min(1.8, max(0.25, abs(r) * 2.2))
    else:
        mag = min(1.8, max(0.25, abs(residual_kwh) / 1000.0))
    
    if residual_kwh < 0:
        # simulated heating too low -> increase setpoints more aggressively
        d_home = +1.45 * step_home * mag
        d_away = +1.90 * step_away * mag
        if residual_kwh < -400:
            d_home *= 1.20
            d_away *= 1.35
        if residual_kwh < -900:
            d_home *= 1.10
            d_away *= 1.20
    
    elif residual_kwh > 0:
        # simulated heating too high -> reduce setpoints more aggressively
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
    """
    Greedy month-by-month refinement:
    - use current residuals
    - try adjusting one month at a time
    - accept only if global CVRMSE improves
    """
    improved_any = False
    best_map = dict(base_map)
    best_cvr = current_best_cvr
    best_nb_abs = current_best_abs_nb
    best_resid = dict(residual)

    month_order = sorted(
        MEASURED_MONTHS,
        key=lambda mm: abs(residual.get(mm, 0.0)),
        reverse=True
    )
    print("[HEUR] Month priority by |residual|:", month_order)

    for mm in month_order:
        th, ta = best_map[mm]
        r = residual.get(mm, 0.0)
        meas = float(measured_kwh[str(mm)])

        dth, dta = month_update_delta(r, meas, step_home, step_away)
        print(
            f"[HEUR][Month {mm:02d}] resid={r:+.1f} kWh  "
            f"old=({th:.2f}, {ta:.2f})  delta=({dth:+.2f}, {dta:+.2f})"
        )

        trial_map = dict(best_map)
        th_new = clamp(th + dth, BOUNDS["t_home"][0], BOUNDS["t_home"][1])
        ta_new = clamp(ta + dta, BOUNDS["t_away"][0], BOUNDS["t_away"][1])
        th_new, ta_new = enforce_home_ge_away(th_new, ta_new)
        trial_map[mm] = (th_new, ta_new)

        run_dir = work_dir / f"month_refine_m{mm:02d}"
        try:
            cvr, nb, _sim_used, resid_new = evaluate_schedule(
                trial_map,
                energyplus_exe,
                run_dir,
                tag=f"month_refine_m{mm:02d}"
            )
        except Exception:
            continue

        #better = (cvr < best_cvr) or (math.isclose(cvr, best_cvr, rel_tol=1e-6) and abs(nb) < best_nb_abs)
        '''better = (
            (cvr < best_cvr + 0.15 and abs(nb) < best_nb_abs) or
            (cvr < best_cvr - 0.05)
        )'''
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
# ============================================================
# NSGA-II polish (local, not blind)
# Only around the current best; constraints enforced by design.
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

        # vars: for each measured month: th, ta
        n_var = 2 * len(measured_months)
        xl = np.zeros(n_var, dtype=float)
        xu = np.zeros(n_var, dtype=float)

        for i, mm in enumerate(measured_months):
            th0, ta0 = base_month_to_home_away.get(mm, base_month_to_home_away[min(base_month_to_home_away.keys())])
            xl[2*i]   = clamp(th0 - NSGA_LOCAL_SPAN_HOME, BOUNDS["t_home"][0], BOUNDS["t_home"][1])
            xu[2*i]   = clamp(th0 + NSGA_LOCAL_SPAN_HOME, BOUNDS["t_home"][0], BOUNDS["t_home"][1])
            xl[2*i+1] = clamp(ta0 - NSGA_LOCAL_SPAN_AWAY, BOUNDS["t_away"][0], BOUNDS["t_away"][1])
            xu[2*i+1] = clamp(ta0 + NSGA_LOCAL_SPAN_AWAY, BOUNDS["t_away"][0], BOUNDS["t_away"][1])

        super().__init__(n_var=n_var, n_obj=2, xl=xl, xu=xu)

        if not self.log_csv.exists():
            self.log_csv.parent.mkdir(parents=True, exist_ok=True)
            with self.log_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                header = ["eval_id"]
                for mm in measured_months:
                    header += [f"t_home_m{mm:02d}", f"t_away_m{mm:02d}"]
                header += ["CVRMSE_%", "NMBE_%", "absNMBE_%", "runtime_s"]
                w.writerow(header)

    def _evaluate(self, x, out, *args, **kwargs):
        self.eval_counter += 1
        eval_id = self.eval_counter

        # build month map by starting from base then overriding measured months
        month_to_home_away = dict(self.base)
        for i, mm in enumerate(self.measured_months):
            th = float(x[2*i])
            ta = float(x[2*i+1])
            th, ta = enforce_home_ge_away(th, ta)
            month_to_home_away[mm] = (th, ta)

        run_dir = self.work_dir / f"run_{eval_id:05d}"
        t0 = time.time()

        try:
            cvr, nb, _sim_used, _resid = evaluate_schedule(month_to_home_away, self.energyplus_exe, run_dir, tag=f"nsga_{eval_id:05d}")
            absnb = abs(nb)
        except Exception:
            cvr, nb, absnb = 1e6, 1e6, 1e6

        runtime = time.time() - t0
        out["F"] = np.array([cvr, absnb], dtype=float)

        with self.log_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            row = [eval_id]
            for mm in self.measured_months:
                th, ta = month_to_home_away[mm]
                row += [th, ta]
            row += [cvr, nb, absnb, runtime]
            w.writerow(row)
            f.flush()

        if CLEANUP_EACH_RUN:
            # keep failed runs optional (here we don't detect failure cleanly; treat huge cvr as failure)
            failed = (cvr >= 1e5)
            if failed and KEEP_FAILED_RUNS:
                return
            shutil.rmtree(run_dir, ignore_errors=True)


def pick_best_from_log(csv_path: Path) -> Tuple[Dict[int, Tuple[float, float]], float, float]:
    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
    if df.empty:
        raise RuntimeError("NSGA polish log is empty/unreadable.")
    df["CVRMSE_%"] = pd.to_numeric(df["CVRMSE_%"], errors="coerce")
    df["absNMBE_%"] = pd.to_numeric(df["absNMBE_%"], errors="coerce")
    df["NMBE_%"] = pd.to_numeric(df["NMBE_%"], errors="coerce")
    df = df.dropna(subset=["CVRMSE_%", "absNMBE_%"])
    df = df.sort_values(["CVRMSE_%", "absNMBE_%"], ascending=[True, True]).reset_index(drop=True)
    best = df.iloc[0].to_dict()

    out_map: Dict[int, Tuple[float, float]] = {}
    for mm in MEASURED_MONTHS:
        thk = f"t_home_m{mm:02d}"
        tak = f"t_away_m{mm:02d}"
        if thk in best and tak in best:
            th = float(best[thk])
            ta = float(best[tak])
            th, ta = enforce_home_ge_away(th, ta)
            out_map[mm] = (th, ta)

    return out_map, float(best["CVRMSE_%"]), float(best["NMBE_%"])


# ============================================================
# Logging
# ============================================================
def append_iter_log(log_csv: Path,
                    iter_id: int,
                    month_to_home_away: Dict[int, Tuple[float, float]],
                    cvr: float,
                    nb: float,
                    residual: Dict[int, float],
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
def main() -> None:
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
        
    shutil.rmtree(WORK_DIR, ignore_errors=True); WORK_DIR.mkdir(parents=True, exist_ok=True)
    energyplus_exe = guess_energyplus_exe(IDD_PATH)
    print(f"[INFO] EnergyPlus exe: {energyplus_exe}")
    print(f"[INFO] Work dir: {WORK_DIR.resolve()}")
    print(f"[INFO] Measured months: {MEASURED_MONTHS}")

    # ---- Read current schedule values from the IDF as starting point
    IDF.setiddname(str(IDD_PATH))
    idf0 = IDF(str(INPUT_IDF_PATH))

    # Prefer radiant as the "truth"; if missing, fall back to air heating
    try:
        month_to_home_away = read_monthly_home_away_from_schedule(idf0, RADIANT_SETPOINT_SCHED_NAME)
        src = RADIANT_SETPOINT_SCHED_NAME
    except Exception:
        month_to_home_away = read_monthly_home_away_from_schedule(idf0, AIR_HEATING_SETPOINT_SCHED_NAME)
        src = AIR_HEATING_SETPOINT_SCHED_NAME

    # Ensure all 12 months exist (carry forward using the last known)
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

    # ---- Baseline evaluate
    baseline_dir = WORK_DIR / "baseline"
    t0 = time.time()
    cvr_best, nb_best, sim_used, resid = evaluate_schedule(month_to_home_away, energyplus_exe, baseline_dir, tag="baseline")
    t_baseline = time.time() - t0

    print("\n=== Baseline (from After1_Cali_RFH.idf schedules) ===")
    print(f"CVRMSE: {cvr_best:.3f} %")
    print(f"NMBE  : {nb_best:.3f} %")
    print(f"[INFO] Baseline runtime: {t_baseline:.1f} s")

    best_map = dict(month_to_home_away)
    best_score = (cvr_best, abs(nb_best))  # multi-criteria single pick
    heur_log = WORK_DIR / "heuristic_log.csv"
    append_iter_log(heur_log, 0, best_map, cvr_best, nb_best, resid, t_baseline)

    # ---- Heuristic loop
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
                    proposal,
                    energyplus_exe,
                    confirm_dir,
                    tag=f"heur_{it:02d}_confirm"
                )
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
    
        print(
            f"\n[HEUR] Iter {it:02d}: CVRMSE={cvr:.3f}%  "
            f"NMBE={nb:.3f}%  step_home={step_home:.2f} "
            f"step_away={step_away:.2f}  improved={improved}"
        )
    
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

    # ---- Optional NSGA-II polish (local)
    if ENABLE_NSAA_POLISH:
        print("\n[NSGA] Starting local NSGA-II polish around heuristic-best...")
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
        algorithm = NSGA2(pop_size=NSGA_POP)
        termination = get_termination("n_gen", NSGA_GEN)

        _ = minimize(
            problem,
            algorithm,
            termination,
            seed=NSGA_SEED,
            save_history=False,
            verbose=True,
        )

        # Pick best from polish results
        try:
            polish_map, cvr_p, nb_p = pick_best_from_log(nsga_log)
            # merge polish_map into best_map (only measured months are in polish_map)
            merged = dict(best_map)
            for mm, (th, ta) in polish_map.items():
                merged[mm] = enforce_home_ge_away(th, ta)

            # evaluate merged once to confirm
            confirm_dir = WORK_DIR / "nsga_best_confirm"
            t2 = time.time()
            cvr_c, nb_c, sim_used_c, resid_c = evaluate_schedule(merged, energyplus_exe, confirm_dir, tag="nsga_best_confirm")
            tconf = time.time() - t2

            print("\n[NSGA] Best (confirmed) from local polish:")
            print(f"CVRMSE={cvr_c:.3f}%  NMBE={nb_c:.3f}%  runtime={tconf:.1f}s")
            if (cvr_c < cvr_best) or (math.isclose(cvr_c, cvr_best, rel_tol=1e-6) and abs(nb_c) < abs(nb_best)):
                best_map = merged
                cvr_best, nb_best = cvr_c, nb_c
                resid = resid_c
                print("[NSGA] -> accepted polish improvement")
            else:
                print("[NSGA] -> no improvement vs heuristic-best")
        except Exception as e:
            print(f"[NSGA] polish log parse/confirm failed: {e}")

    # ---- Write final IDF (apply best_map to both schedules)
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
        print(f"  Month {mm:02d}: T_home={th:.2f} °C, T_away={ta:.2f} °C  (resid_kWh={resid.get(mm, float('nan')):+.1f})")
    
    print(f"CVRMSE = {cvr_best:.3f} %")
    print(f"NMBE   = {nb_best:.3f} %")
    
    print(f"[OK] Calibrated IDF saved to: {OUTPUT_IDF_PATH.resolve()}")
    
    return {
        "metrics": {
            "CVRMSE": float(cvr_best),
            "NMBE": float(nb_best)
        },
        "idf_path": str(OUTPUT_IDF_PATH.resolve())
    }
    print(f"FINAL_METRICS_JSON={{\"CVRMSE\": {cvr_best:.6f}, \"NMBE\": {nb_best:.6f}}}")
    print(f"[INFO] Heuristic log: {heur_log.resolve()}")
    if ENABLE_NSAA_POLISH:
        print(f"[INFO] NSGA polish log: {(WORK_DIR / 'nsga_polish' / 'eval_log.csv').resolve()}")
    
    print(f"[OK] Calibrated IDF saved to: {OUTPUT_IDF_PATH.resolve()}")

    # Optional: remove baseline outputs to save disk (keep logs + final)
    try:
        shutil.rmtree(WORK_DIR / "baseline")
    except Exception:
        pass


if __name__ == "__main__":
    main()
