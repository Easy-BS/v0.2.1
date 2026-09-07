# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 12:50:02 2026

@author: Xiguan Liang @SKKU
"""

# ./CALI_flow/nodes/Cali_Tset_Occ.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import shutil
import subprocess
import time
import re
import csv

import numpy as np
import pandas as pd
from eppy.modeleditor import IDF

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize

from cali_runtime_config import load_runtime_config
_RUNTIME = load_runtime_config()


IDD_PATH = Path(_RUNTIME.get("idd_path", r"C:/EnergyPlusV8-9-0/Energy+.idd"))
EPW_PATH = Path(_RUNTIME.get("epw_path", r"C:\EnergyPlusV8-9-0\WeatherData\KOR_INCH'ON_IWEC.epw"))
INPUT_IDF_PATH = Path(_RUNTIME.get("idf_path", "./Calibration/After_Cali_RFH.idf"))
OUTPUT_IDF_PATH = Path("./Calibration/After1_Cali_RFH.idf")

# Working directory for repeated simulations
WORK_DIR: Path = Path("./Calibration/_cali_runs_tset_occ")

# Cleanup policy
CLEANUP_EACH_RUN: bool = False #True          # delete run_00XXX after each evaluation
KEEP_FAILED_RUNS: bool = True #False         # if True, keep run folders when E+ fails (debugging)

# NSGA-II settings
POP_SIZE: int = 24 #24
N_GEN: int = 10
SEED: int = 42

# Heating meters (Monthly, J) to sum if present
HEATING_METERS_PRIORITY: List[str] = [
    "DistrictHeating:Facility",
    "Electricity:Heating",
    "Gas:Heating",
]

# Target schedules to overwrite (monthly-dependent)
# In many RFH models, the radiant schedule is the real driver; optionally also update HEATING SETPOINTS.
RADIANT_SETPOINT_SCHED_NAME = "RADIANT HEATING SETPOINTS"
AIR_HEATING_SETPOINT_SCHED_NAME = "HEATING SETPOINTS"  # will be updated too (recommended)

# Target People objects: we will standardize ALL PEOPLE objects to the same People/Area value
# and keep their existing schedules (Occ_Sch_Res_Quick etc.) unchanged in this step.


# ============================================================
# Bounds (quick-mode, residential 3-bedroom assumption)
# ============================================================
# Decision variables:
# 1) people_per_m2 (People/Area) applied to every PEOPLE object uniformly
# 2) For each measured month: T_home_month, T_away_month (°C), applied to setpoint schedules
#
# Reasonable initial ranges:
# - People/Area: 0.01–0.08 people/m²  (~1–6 people in 75–100 m², broad)
# - Heating setpoints: home 18–24°C; away 10–19°C (setback)
BOUNDS = {
    "people_per_m2": (0.01, 0.08),
    "t_home": (20.0, 25.0), 
    "t_away": (15.0, 20.0), }
#(18.0, 24.0)#(10.0, 19.0)

# Soft penalty preferences (not hard limits):
# Encourage “typical” residential values unless needed for fit
PREF = {
    "people_per_m2_hi_soft": 0.06,
    "people_per_m2_w": 8.0,         # penalty weight (percentage points)
    "t_home_hi_soft": 23.0,
    "t_home_w": 2.0,
    "t_away_lo_soft": 12.0,
    "t_away_w": 2.0,
}



HEATING_METERS_PRIORITY = [
    "DistrictHeating:Facility",
    "Electricity:Heating",
    "Gas:Heating",
]

def read_monthly_meter_j_from_mtr(mtr_path: Path) -> Dict[str, Dict[int, float]]:
    """
    Parse EnergyPlus eplusout.mtr for Monthly meters.
    Returns:
      { meter_name: { month:int -> value_J:float } }

    Works with patterns like:
      4,31, 1
      978,19335458796.15798,...
    and dictionary lines like:
      978,9,DistrictHeating:Facility [J] !Monthly ...
    """
    lines = mtr_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    # Split dictionary vs data
    end_idx = None
    for i, line in enumerate(lines):
        if "End of Data Dictionary" in line:
            end_idx = i
            break
    if end_idx is None:
        raise RuntimeError("Could not find 'End of Data Dictionary' in eplusout.mtr")

    dict_lines = lines[: end_idx + 1]
    data_lines = lines[end_idx + 1 :]

    # Map monthly meter index -> meter name (without [J])
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
        raise RuntimeError(
            "No Monthly meters detected in eplusout.mtr dictionary. "
            "Ensure your IDF has Output:Meter objects with Monthly frequency."
        )

    meters_monthly: Dict[str, Dict[int, float]] = {n: {} for n in set(monthly_idx_to_name.values())}
    current_month: Optional[int] = None

    for line in data_lines:
        s = line.strip()
        if not s:
            continue
        parts = [p.strip() for p in s.split(",")]

        # Monthly record header: "4, <cum_days>, <month>"
        if parts[0] == "4" and len(parts) >= 3:
            try:
                mm = int(parts[2])
                current_month = mm if 1 <= mm <= 12 else None
            except Exception:
                current_month = None
            continue

        # Meter data line: "<idx>, <value>, ..."
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
    for k, v in MONTH_NAME_TO_INT.items():
        if k.startswith(s) and len(s) >= 3:
            return v
    if s.isdigit():
        mm = int(s)
        return mm if 1 <= mm <= 12 else None
    return None


def read_monthly_meter_j_from_meter_csv(meter_csv: Path) -> dict[str, dict[int, float]]:
    """
    Parses '*Meter.csv' format:
      Date/Time    DistrictHeating:Facility [J](Monthly)
      January      6966423054
      ...
    """
    df = pd.read_csv(meter_csv)

    month_col = None
    for c in df.columns:
        if str(c).strip().lower() in ("date/time", "date", "time"):
            month_col = c
            break
    if month_col is None:
        month_col = df.columns[0]

    meters: dict[str, dict[int, float]] = {}

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


def _month_cell_to_int(x) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip().lower()
    month_map = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }
    if s in month_map:
        return month_map[s]
    if s.isdigit():
        mm = int(s)
        return mm if 1 <= mm <= 12 else None
    # allow Jan/Feb...
    for k, v in month_map.items():
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
    Returns {month:int -> heating_kwh:float} by summing available heating meters.

    Priority:
      1) *Meter.csv (if present)
      2) eplusout.mtr (robust fallback)
    """
    # 1) Prefer *Meter.csv
    meter_csvs = sorted(out_dir.glob("*Meter.csv"))
    if meter_csvs:
        meters = read_monthly_meter_j_from_meter_csv(meter_csvs[0])
    else:
        # 2) Fallback to eplusout.mtr
        mtr = out_dir / "eplusout.mtr"
        if not mtr.exists():
            raise RuntimeError(
                f"No *Meter.csv found in {out_dir} and no eplusout.mtr either. "
                "Check EnergyPlus outputs and Output:Meter (Monthly) in your IDF."
            )
        meters = read_monthly_meter_j_from_mtr(mtr)

    # Sum priority heating meters (J) -> kWh
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
        raise RuntimeError(
            f"None of HEATING_METERS_PRIORITY found: {HEATING_METERS_PRIORITY}. "
            f"Available meters: {list(meters.keys())}"
        )

    return {mm: val_j / 3.6e6 for mm, val_j in monthly_j.items()}  # J -> kWh



# ============================================================
# eppy: apply PEOPLE and monthly setpoint schedule
# ============================================================
def ensure_schedule_type_limits(idf: IDF, name: str, lower, upper, numeric_type: str) -> None:
    for stl in idf.idfobjects.get("SCHEDULETYPELIMITS", []):
        if getattr(stl, "Name", "").strip().upper() == name.upper():
            return
    stl = idf.newidfobject("SCHEDULETYPELIMITS")
    stl.Name = name
    if lower is not None:
        # fields exist in 8.9 for ScheduleTypeLimits
        stl.Lower_Limit_Value = float(lower)
    if upper is not None:
        stl.Upper_Limit_Value = float(upper)
    # Numeric_Type is optional in some IDDs; try safe set
    if hasattr(stl, "Numeric_Type"):
        stl.Numeric_Type = numeric_type


def delete_schedule_compact(idf: IDF, sched_name: str) -> None:
    seq = idf.idfobjects.get("SCHEDULE:COMPACT", [])
    to_del = [s for s in seq if getattr(s, "Name", "").strip().upper() == sched_name.upper()]
    for s in to_del:
        seq.remove(s)


def create_schedule_compact_monthly_setpoints(
    idf: IDF,
    sched_name: str,
    type_limits: str,
    month_to_home_away: Dict[int, Tuple[float, float]],
    *,
    home_periods: List[Tuple[str, str]] = None,
    away_period: Tuple[str, str] = None,
) -> None:
    """
    Build a Schedule:Compact that varies by month.
    We map:
      - Home (T_home): 00:00-08:00 and 18:00-24:00  (default)
      - Away (T_away): 08:00-18:00                  (default)
    You can change these time windows later without touching optimizer wiring.
    """
    if home_periods is None:
        home_periods = [("00:00", "08:00"), ("18:00", "24:00")]
    if away_period is None:
        away_period = ("08:00", "18:00")

    # Build "Through: mm/dd" segments in chronological order
    # For each month present in month_to_home_away, we define a segment.
    # For months absent, we keep the last defined values (OK because you only simulate measured months),
    # but we’ll still define a full-year schedule for safety.
    def month_end_day(mm: int) -> int:
        return {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}[mm]

    # Ensure schedule exists fresh
    delete_schedule_compact(idf, sched_name)

    sc = idf.newidfobject("SCHEDULE:COMPACT")
    sc.Name = sched_name
    sc.Schedule_Type_Limits_Name = type_limits

    # Extensible fields
    # Pattern per segment:
    #   Through: M/D
    #   For: AllDays
    #   Until: 08:00, T_home
    #   Until: 18:00, T_away
    #   Until: 24:00, T_home
    #
    # EnergyPlus Schedule:Compact fields are Field_1..Field_n
    field_i = 1

    # Fill missing months by carrying forward last specified values (robust)
    # Start with January if missing: use the first provided month
    provided_months = sorted(month_to_home_away.keys())
    if not provided_months:
        raise ValueError("month_to_home_away is empty.")

    last_home, last_away = month_to_home_away[provided_months[0]]

    for mm in range(1, 13):
        if mm in month_to_home_away:
            last_home, last_away = month_to_home_away[mm]

        sc.__setattr__(f"Field_{field_i}", f"Through: {mm:02d}/{month_end_day(mm):02d}"); field_i += 1
        sc.__setattr__(f"Field_{field_i}", "For: AllDays"); field_i += 1

        # 00-08 home -> represent as Until 08:00
        sc.__setattr__(f"Field_{field_i}", "Until: 08:00"); field_i += 1
        sc.__setattr__(f"Field_{field_i}", float(last_home)); field_i += 1

        # 08-18 away -> Until 18:00
        sc.__setattr__(f"Field_{field_i}", "Until: 18:00"); field_i += 1
        sc.__setattr__(f"Field_{field_i}", float(last_away)); field_i += 1

        # 18-24 home -> Until 24:00
        sc.__setattr__(f"Field_{field_i}", "Until: 24:00"); field_i += 1
        sc.__setattr__(f"Field_{field_i}", float(last_home)); field_i += 1


def apply_people_per_area_uniform(idf: IDF, people_per_m2: float) -> int:
    """
    Standardize all PEOPLE objects to:
      Number of People Calculation Method = People/Area
      People per Zone Floor Area = people_per_m2
    Returns number of PEOPLE objects updated.
    """
    count = 0
    for p in idf.idfobjects.get("PEOPLE", []):
        # IDD field names differ slightly across versions; handle both
        if hasattr(p, "Number_of_People_Calculation_Method"):
            p.Number_of_People_Calculation_Method = "People/Area"
        elif hasattr(p, "NumberofPeopleCalculationMethod"):
            p.NumberofPeopleCalculationMethod = "People/Area"

        # Now set People per Floor Area
        if hasattr(p, "People_per_Zone_Floor_Area"):
            p.People_per_Zone_Floor_Area = float(people_per_m2)
        elif hasattr(p, "PeopleperZoneFloorArea"):
            p.PeopleperZoneFloorArea = float(people_per_m2)
        elif hasattr(p, "People_per_Floor_Area"):
            p.People_per_Floor_Area = float(people_per_m2)
        elif hasattr(p, "PeopleperFloorArea"):
            p.PeopleperFloorArea = float(people_per_m2)

        # Clear absolute number fields (avoid conflicts)
        if hasattr(p, "Number_of_People"):
            p.Number_of_People = ""
        elif hasattr(p, "NumberofPeople"):
            p.NumberofPeople = ""

        count += 1
    return count


# ============================================================
# Soft penalty (optional but recommended)
# ============================================================
def soft_penalty(people_per_m2: float, t_home_by_m: Dict[int, float], t_away_by_m: Dict[int, float]) -> float:
    """
    Returns penalty in "percentage points" to add to objectives.
    This discourages extreme solutions but does not forbid them.
    """
    p = 0.0

    # Too high occupancy density for residential
    if people_per_m2 > PREF["people_per_m2_hi_soft"]:
        p += PREF["people_per_m2_w"] * ((people_per_m2 - PREF["people_per_m2_hi_soft"]) / 0.02) ** 2

    # Home setpoint too high
    for mm, th in t_home_by_m.items():
        if th > PREF["t_home_hi_soft"]:
            p += PREF["t_home_w"] * ((th - PREF["t_home_hi_soft"]) / 1.0) ** 2

    # Away setback too low (unrealistically cold)
    for mm, ta in t_away_by_m.items():
        if ta < PREF["t_away_lo_soft"]:
            p += PREF["t_away_w"] * ((PREF["t_away_lo_soft"] - ta) / 1.0) ** 2

    return p


# ============================================================
# Problem definition (pymoo)
# ============================================================
class TsetOccCalibrationProblem(ElementwiseProblem):
    def __init__(
        self,
        idd_path: Path,
        input_idf_path: Path,
        epw_path: Path,
        work_dir: Path,
        measured_monthly_kwh: Dict[str, float],
        energyplus_exe: Path,
        log_csv_path: Path,
        measured_months: List[int],
    ):
        self.measured = measured_monthly_kwh
        self.measured_months = measured_months

        # Variables:
        # x[0] = people_per_m2
        # then for each measured month i:
        #   x[1+2*i] = t_home_month
        #   x[1+2*i+1] = t_away_month
        n_var = 1 + 2 * len(measured_months)

        xl = np.zeros(n_var, dtype=float)
        xu = np.zeros(n_var, dtype=float)

        xl[0], xu[0] = BOUNDS["people_per_m2"]

        for i in range(len(measured_months)):
            xl[1 + 2*i],     xu[1 + 2*i]     = BOUNDS["t_home"]
            xl[1 + 2*i + 1], xu[1 + 2*i + 1] = BOUNDS["t_away"]

        super().__init__(n_var=n_var, n_obj=2, xl=xl, xu=xu)


        self.idd_path = idd_path
        self.input_idf_path = input_idf_path
        self.epw_path = epw_path
        self.work_dir = work_dir
        self.energyplus_exe = energyplus_exe
        self.log_csv_path = log_csv_path

        self.eval_counter = 0

        # Prepare log header
        if not self.log_csv_path.exists():
            self.log_csv_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                header = ["eval_id", "people_per_m2"]
                for mm in measured_months:
                    header += [f"t_home_m{mm:02d}", f"t_away_m{mm:02d}"]
                header += ["CVRMSE_%", "NMBE_%", "absNMBE_%", "penalty", "runtime_s"]
                w.writerow(header)

    def _evaluate(self, x, out, *args, **kwargs):
        self.eval_counter += 1
        eval_id = self.eval_counter

        people_per_m2 = float(x[0])

        # Map month -> (t_home, t_away)
        t_home_by_m: Dict[int, float] = {}
        t_away_by_m: Dict[int, float] = {}
        for i, mm in enumerate(self.measured_months):
            t_home_by_m[mm] = float(x[1 + 2*i])
            t_away_by_m[mm] = float(x[1 + 2*i + 1])
        
        
        # Penalty
        pen = soft_penalty(people_per_m2, t_home_by_m, t_away_by_m)

        # Prepare run folders
        run_dir = self.work_dir / f"run_{eval_id:05d}"
        run_idf = run_dir / "in.idf"
        out_dir = run_dir / "out"
        run_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        cvr = nb = absnb = 1e6

        try:
            IDF.setiddname(str(self.idd_path))
            idf = IDF(str(self.input_idf_path))

            # Ensure schedule type limits (safe)
            ensure_schedule_type_limits(idf, "Any Number", None, None, "Continuous")
            ensure_schedule_type_limits(idf, "Fraction", 0.0, 1.0, "Continuous")

            # 1) Apply occupancy density uniformly to all PEOPLE objects
            _n_people_objs = apply_people_per_area_uniform(idf, people_per_m2)

            # 2) Apply monthly setpoint schedules (radiant + air heating)
            month_to_home_away = {mm: (t_home_by_m[mm], t_away_by_m[mm]) for mm in self.measured_months}

            create_schedule_compact_monthly_setpoints(
                idf,
                RADIANT_SETPOINT_SCHED_NAME,
                "Any Number",
                month_to_home_away,
            )

            # Strongly recommended: keep air heating schedule consistent too
            create_schedule_compact_monthly_setpoints(
                idf,
                AIR_HEATING_SETPOINT_SCHED_NAME,
                "Any Number",
                month_to_home_away,
            )

            # Save + simulate
            idf.saveas(str(run_idf))
            run_energyplus(self.energyplus_exe, run_idf, self.epw_path, out_dir)

            sim_monthly_kwh = read_sim_monthly_heating_kwh(out_dir)
            cvr, nb, _ = compute_metrics(self.measured, sim_monthly_kwh)
            absnb = abs(nb)

        except Exception:
            cvr, nb, absnb = 1e6, 1e6, 1e6

        runtime = time.time() - t0

        # Objectives (soft-penalized)
        # Penalize both, but more on CVRMSE than bias
        cvr_p = cvr + pen
        absnb_p = absnb + 0.5 * pen
        out["F"] = np.array([cvr_p, absnb_p], dtype=float)
        
       
        # Log
        row = [eval_id, people_per_m2]
        for mm in self.measured_months:
            row += [t_home_by_m[mm], t_away_by_m[mm]]
        row += [cvr, nb, absnb, pen, runtime]

        with self.log_csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(row)
            f.flush()

        # IMPORTANT: clear run folder to avoid disk bloat
        # If you want to keep failed runs for debugging, guard with a flag.
        # Cleanup (requested)
        if CLEANUP_EACH_RUN:
            failed = False
            if failed and KEEP_FAILED_RUNS:
                return
            shutil.rmtree(run_dir, ignore_errors=True)
            
        


# ============================================================
# Selection + final IDF writing
# ============================================================
def select_best_from_log(log_csv: Path, measured_months: List[int]) -> Tuple[Dict[str, float], float, float]:
    """
    Single-point selection:
      minimize CVRMSE first, then minimize |NMBE|
    Uses raw metrics (not penalized) so you can interpret actual fit.
    """
    df = pd.read_csv(log_csv, engine="python", on_bad_lines="skip")
    if df.empty:
        raise RuntimeError("Log CSV is empty or unreadable.")

    for col in ["CVRMSE_%", "absNMBE_%", "NMBE_%"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["CVRMSE_%", "absNMBE_%"])
    df = df.sort_values(["CVRMSE_%", "absNMBE_%"], ascending=[True, True]).reset_index(drop=True)
    best = df.iloc[0].to_dict()

    params: Dict[str, float] = {
        "people_per_m2": float(best["people_per_m2"]),
    }
    for mm in measured_months:
        params[f"t_home_m{mm:02d}"] = float(best[f"t_home_m{mm:02d}"])
        params[f"t_away_m{mm:02d}"] = float(best[f"t_away_m{mm:02d}"])

    return params, float(best["CVRMSE_%"]), float(best["NMBE_%"])


def write_final_idf(
    idd_path: Path,
    input_idf_path: Path,
    output_idf_path: Path,
    measured_months: List[int],
    best_params: Dict[str, float],
) -> None:
    IDF.setiddname(str(idd_path))
    idf = IDF(str(input_idf_path))

    ensure_schedule_type_limits(idf, "Any Number", None, None, "Continuous")
    ensure_schedule_type_limits(idf, "Fraction", 0.0, 1.0, "Continuous")

    # People
    apply_people_per_area_uniform(idf, best_params["people_per_m2"])

    # Setpoints
    month_to_home_away = {
        mm: (best_params[f"t_home_m{mm:02d}"], best_params[f"t_away_m{mm:02d}"])
        for mm in measured_months
    }
    create_schedule_compact_monthly_setpoints(idf, RADIANT_SETPOINT_SCHED_NAME, "Any Number", month_to_home_away)
    create_schedule_compact_monthly_setpoints(idf, AIR_HEATING_SETPOINT_SCHED_NAME, "Any Number", month_to_home_away)

    output_idf_path.parent.mkdir(parents=True, exist_ok=True)
    idf.saveas(str(output_idf_path))


# ============================================================
# Baseline report (unchanged params)
# ============================================================
def baseline_report(idd_path: Path, input_idf_path: Path, epw_path: Path, energyplus_exe: Path, work_dir: Path) -> Tuple[float, float]:
    baseline_dir = work_dir / "baseline"
    baseline_idf = baseline_dir / "baseline.idf"
    out_dir = baseline_dir / "out"

    baseline_dir.mkdir(parents=True, exist_ok=True)

    IDF.setiddname(str(idd_path))
    idf = IDF(str(input_idf_path))
    idf.saveas(str(baseline_idf))

    run_energyplus(energyplus_exe, baseline_idf, epw_path, out_dir)
    sim_monthly_kwh = read_sim_monthly_heating_kwh(out_dir)
    cvr, nb, sim_used = compute_metrics(MEASURED_MONTHLY_KWH, sim_monthly_kwh)

    print("\n=== Baseline Monthly Comparison (kWh) ===")
    print("Month | Measured | Simulated | Residual (Sim-Meas)")
    for m in sorted(int(k) for k in MEASURED_MONTHLY_KWH.keys()):
        meas = MEASURED_MONTHLY_KWH[str(m)]
        sim = sim_used.get(m, float("nan"))
        print(f"{m:>5} | {meas:>8.2f} | {sim:>9.2f} | {sim - float(meas):>+12.2f}")

    print("\n=== Baseline Metrics (Monthly) ===")
    print(f"CVRMSE: {cvr:.3f} %")
    print(f"NMBE  : {nb:.3f} %\n")
    
    return cvr, nb


# ============================================================
# Main
# ============================================================
def main() -> None:
    if not IDD_PATH.exists():
        raise FileNotFoundError(f"IDD not found: {IDD_PATH.resolve()}")
    if not EPW_PATH.exists():
        raise FileNotFoundError(f"EPW not found: {EPW_PATH.resolve()}")
    if not INPUT_IDF_PATH.exists():
        raise FileNotFoundError(f"IDF not found: {INPUT_IDF_PATH.resolve()}")

    reset_workspace(WORK_DIR)
    
    shutil.rmtree(WORK_DIR, ignore_errors=True); WORK_DIR.mkdir(parents=True, exist_ok=True)
    energyplus_exe = guess_energyplus_exe(IDD_PATH)
    print(f"[INFO] EnergyPlus exe: {energyplus_exe}")
    print(f"[INFO] Work dir: {WORK_DIR.resolve()}")

    measured_months = sorted(int(k) for k in MEASURED_MONTHLY_KWH.keys())
    print(f"[INFO] Measured months: {measured_months}")

    # Baseline (for reference)
    baseline_report(IDD_PATH, INPUT_IDF_PATH, EPW_PATH, energyplus_exe, WORK_DIR)

    log_csv = WORK_DIR / "eval_log.csv"

    problem = TsetOccCalibrationProblem(
        idd_path=IDD_PATH,
        input_idf_path=INPUT_IDF_PATH,
        epw_path=EPW_PATH,
        work_dir=WORK_DIR,
        measured_monthly_kwh=MEASURED_MONTHLY_KWH,
        energyplus_exe=energyplus_exe,
        log_csv_path=log_csv,
        measured_months=measured_months,
    )

    algorithm = NSGA2(pop_size=POP_SIZE)
    termination = get_termination("n_gen", N_GEN)

    print("\n[INFO] Starting NSGA-II optimization (occupancy + monthly setpoints)...")
    _res = minimize(
        problem,
        algorithm,
        termination,
        seed=SEED,
        save_history=False,
        verbose=True,
    )

    best_params, best_cvr, best_nb = select_best_from_log(log_csv, measured_months)

    print("\n=== Selected Best (single-point pick, based on raw metrics) ===")
    print(f"people_per_m2 = {best_params['people_per_m2']:.4f}")
    for mm in measured_months:
        print(f"  Month {mm:02d}: T_home={best_params[f't_home_m{mm:02d}']:.2f} °C, "
              f"T_away={best_params[f't_away_m{mm:02d}']:.2f} °C")
    print(f"CVRMSE = {best_cvr:.3f} %")
    print(f"NMBE   = {best_nb:.3f} %")
    print(f"[INFO] Full evaluation log: {log_csv.resolve()}")

    write_final_idf(IDD_PATH, INPUT_IDF_PATH, OUTPUT_IDF_PATH, measured_months, best_params)
    print(f"[OK] Calibrated IDF saved to: {OUTPUT_IDF_PATH.resolve()}")

    # Optional: clear baseline outputs too (keep log + final IDF)
    try:
        shutil.rmtree(WORK_DIR / "baseline")
    except Exception:
        pass


if __name__ == "__main__":
    main()

