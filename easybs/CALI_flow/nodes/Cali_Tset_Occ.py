# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 17:11:44 2026

@author: Xiguan Liang @SKKU
"""

# ./CALI_flow/nodes/Cali_Tset_Occ.py


from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import shutil
import subprocess
import time
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
OUTPUT_IDF_PATH = Path("./Calibration/After1_Cali_RFH.idf")

# Report written by the amy_weather node, when present
WEATHER_REPORT = _RUNTIME.get("weather_report") or {}

# Working directory for repeated simulations
WORK_DIR: Path = Path("./Calibration/_cali_runs_tset_occ")

# Cleanup policy
CLEANUP_EACH_RUN: bool = False
KEEP_FAILED_RUNS: bool = True

# NSGA-II settings
POP_SIZE: int = 24
N_GEN: int = 10
SEED: int = 42

# Heating meters (Monthly, J) to sum if present
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
# Target schedules to overwrite (monthly-dependent)
RADIANT_SETPOINT_SCHED_NAME = "RADIANT HEATING SETPOINTS"
AIR_HEATING_SETPOINT_SCHED_NAME = "HEATING SETPOINTS"


# ============================================================
# Bounds
# ============================================================
# Decision variables:
#   1) people_per_m2, applied uniformly to every PEOPLE object
#   2) for each measured month: T_home and T_away in degrees C
#
# people_per_m2   persons/m2
# t_home         20 to 25 C, occupied period
# t_away         15 to 20 C, setback period
#
# The setpoint box is deliberately narrower here than in stage 3, where it is
# relaxed so that monthly residuals can drive the schedule.
BOUNDS = {
    "people_per_m2": (0.045, 0.095),
    "t_home": (20.0, 25.0),
    "t_away": (15.0, 20.0),
}


def echo_inputs() -> None:
    """Print the inputs actually received, so the log records what was
    calibrated against rather than what was assumed."""
    print("[INFO] Stage          : 2 - occupancy and monthly setpoints")
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
    for k, v in MONTH_NAME_TO_INT.items():
        if k.startswith(s) and len(s) >= 3:
            return v
    if s.isdigit():
        mm = int(s)
        return mm if 1 <= mm <= 12 else None
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
            raise RuntimeError(
                f"No *Meter.csv found in {out_dir} and no eplusout.mtr either. "
                "Check EnergyPlus outputs and Output:Meter (Monthly) in your IDF."
            )
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
        raise RuntimeError(
            f"None of HEATING_METERS_PRIORITY found: {HEATING_METERS_PRIORITY}. "
            f"Available meters: {list(meters.keys())}"
        )
    return {mm: val_j / 3.6e6 for mm, val_j in monthly_j.items()}


# ============================================================
# eppy: apply PEOPLE and monthly setpoint schedules
# ============================================================
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


def delete_schedule_compact(idf: IDF, sched_name: str) -> None:
    seq = idf.idfobjects.get("SCHEDULE:COMPACT", [])
    for s in [s for s in seq if getattr(s, "Name", "").strip().upper() == sched_name.upper()]:
        seq.remove(s)


def create_schedule_compact_monthly_setpoints(
    idf: IDF,
    sched_name: str,
    type_limits: str,
    month_to_home_away: Dict[int, Tuple[float, float]],
) -> None:
    """Build a Schedule:Compact that varies by month.

        Home (T_home): 00:00-08:00 and 18:00-24:00
        Away (T_away): 08:00-18:00

    Months absent from month_to_home_away carry forward the last defined
    values, so the schedule covers the full year.
    """
    def month_end_day(mm: int) -> int:
        return {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
                7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}[mm]

    delete_schedule_compact(idf, sched_name)

    sc = idf.newidfobject("SCHEDULE:COMPACT")
    sc.Name = sched_name
    sc.Schedule_Type_Limits_Name = type_limits

    provided_months = sorted(month_to_home_away.keys())
    if not provided_months:
        raise ValueError("month_to_home_away is empty.")

    last_home, last_away = month_to_home_away[provided_months[0]]
    field_i = 1

    for mm in range(1, 13):
        if mm in month_to_home_away:
            last_home, last_away = month_to_home_away[mm]

        setattr(sc, f"Field_{field_i}", f"Through: {mm:02d}/{month_end_day(mm):02d}"); field_i += 1
        setattr(sc, f"Field_{field_i}", "For: AllDays"); field_i += 1
        setattr(sc, f"Field_{field_i}", "Until: 08:00"); field_i += 1
        setattr(sc, f"Field_{field_i}", float(last_home)); field_i += 1
        setattr(sc, f"Field_{field_i}", "Until: 18:00"); field_i += 1
        setattr(sc, f"Field_{field_i}", float(last_away)); field_i += 1
        setattr(sc, f"Field_{field_i}", "Until: 24:00"); field_i += 1
        setattr(sc, f"Field_{field_i}", float(last_home)); field_i += 1


def apply_people_per_area_uniform(idf: IDF, people_per_m2: float) -> int:
    """Set every PEOPLE object to People/Area with the given density.

    Returns the number of PEOPLE objects updated.
    """
    count = 0
    for p in idf.idfobjects.get("PEOPLE", []):
        if hasattr(p, "Number_of_People_Calculation_Method"):
            p.Number_of_People_Calculation_Method = "People/Area"
        elif hasattr(p, "NumberofPeopleCalculationMethod"):
            p.NumberofPeopleCalculationMethod = "People/Area"

        if hasattr(p, "People_per_Zone_Floor_Area"):
            p.People_per_Zone_Floor_Area = float(people_per_m2)
        elif hasattr(p, "PeopleperZoneFloorArea"):
            p.PeopleperZoneFloorArea = float(people_per_m2)
        elif hasattr(p, "People_per_Floor_Area"):
            p.People_per_Floor_Area = float(people_per_m2)
        elif hasattr(p, "PeopleperFloorArea"):
            p.PeopleperFloorArea = float(people_per_m2)

        # Clear the absolute count so it cannot conflict with People/Area
        if hasattr(p, "Number_of_People"):
            p.Number_of_People = ""
        elif hasattr(p, "NumberofPeople"):
            p.NumberofPeople = ""

        count += 1
    return count


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

        # x[0] = people_per_m2
        # x[1 + 2i]     = t_home for measured_months[i]
        # x[1 + 2i + 1] = t_away for measured_months[i]
        n_var = 1 + 2 * len(measured_months)
        xl = np.zeros(n_var, dtype=float)
        xu = np.zeros(n_var, dtype=float)

        xl[0], xu[0] = BOUNDS["people_per_m2"]
        for i in range(len(measured_months)):
            xl[1 + 2 * i],     xu[1 + 2 * i]     = BOUNDS["t_home"]
            xl[1 + 2 * i + 1], xu[1 + 2 * i + 1] = BOUNDS["t_away"]

        super().__init__(n_var=n_var, n_obj=2, xl=xl, xu=xu)

        self.idd_path = idd_path
        self.input_idf_path = input_idf_path
        self.epw_path = epw_path
        self.work_dir = work_dir
        self.energyplus_exe = energyplus_exe
        self.log_csv_path = log_csv_path
        self.eval_counter = 0

        if not self.log_csv_path.exists():
            self.log_csv_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_csv_path.open("w", newline="", encoding="utf-8") as f:
                header = ["eval_id", "people_per_m2"]
                for mm in measured_months:
                    header += [f"t_home_m{mm:02d}", f"t_away_m{mm:02d}"]
                header += ["CVRMSE_%", "NMBE_%", "absNMBE_%", "runtime_s"]
                csv.writer(f).writerow(header)

    def _evaluate(self, x, out, *args, **kwargs):
        self.eval_counter += 1
        eval_id = self.eval_counter

        people_per_m2 = float(x[0])

        t_home_by_m: Dict[int, float] = {}
        t_away_by_m: Dict[int, float] = {}
        for i, mm in enumerate(self.measured_months):
            t_home_by_m[mm] = float(x[1 + 2 * i])
            t_away_by_m[mm] = float(x[1 + 2 * i + 1])

        run_dir = self.work_dir / f"run_{eval_id:05d}"
        run_idf = run_dir / "in.idf"
        out_dir = run_dir / "out"
        run_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        failed = False
        cvr = nb = absnb = 1e6

        try:
            IDF.setiddname(str(self.idd_path))
            idf = IDF(str(self.input_idf_path))

            ensure_schedule_type_limits(idf, "Any Number", None, None, "Continuous")
            ensure_schedule_type_limits(idf, "Fraction", 0.0, 1.0, "Continuous")

            apply_people_per_area_uniform(idf, people_per_m2)

            month_to_home_away = {mm: (t_home_by_m[mm], t_away_by_m[mm])
                                  for mm in self.measured_months}
            create_schedule_compact_monthly_setpoints(
                idf, RADIANT_SETPOINT_SCHED_NAME, "Any Number", month_to_home_away)
            create_schedule_compact_monthly_setpoints(
                idf, AIR_HEATING_SETPOINT_SCHED_NAME, "Any Number", month_to_home_away)

            idf.saveas(str(run_idf))
            run_energyplus(self.energyplus_exe, run_idf, self.epw_path, out_dir)

            sim_heat = read_sim_monthly_heating_kwh(out_dir)
            sim_monthly_kwh = heat_to_fuel(sim_heat, BOILER_EFF)
            cvr, nb, _ = compute_metrics(self.measured, sim_monthly_kwh)
            absnb = abs(nb)
        except Exception:
            failed = True
            cvr, nb, absnb = 1e6, 1e6, 1e6

        runtime = time.time() - t0

        # Objectives are the metrics themselves. The bounds are already
        # physical, so no plausibility penalty is applied and the reported
        # metrics are the optimized ones.
        out["F"] = np.array([cvr, absnb], dtype=float)

        row = [eval_id, people_per_m2]
        for mm in self.measured_months:
            row += [t_home_by_m[mm], t_away_by_m[mm]]
        row += [cvr, nb, absnb, runtime]

        with self.log_csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
            f.flush()

        if CLEANUP_EACH_RUN:
            if failed and KEEP_FAILED_RUNS:
                return
            shutil.rmtree(run_dir, ignore_errors=True)


# ============================================================
# Selection and final IDF writing
# ============================================================
def select_best_from_log(log_csv: Path,
                         measured_months: List[int]) -> Tuple[Dict[str, float], float, float]:
    """Lexicographic selection: minimize CVRMSE, then |NMBE|."""
    df = pd.read_csv(log_csv, engine="python", on_bad_lines="skip")
    if df.empty:
        raise RuntimeError("Log CSV is empty or unreadable.")

    for col in ["CVRMSE_%", "absNMBE_%", "NMBE_%"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["CVRMSE_%", "absNMBE_%"])
    if df.empty:
        raise RuntimeError("No valid evaluation rows remain after cleaning.")

    df = df.sort_values(["CVRMSE_%", "absNMBE_%"], ascending=[True, True]).reset_index(drop=True)
    best = df.iloc[0].to_dict()

    params: Dict[str, float] = {"people_per_m2": float(best["people_per_m2"])}
    for mm in measured_months:
        params[f"t_home_m{mm:02d}"] = float(best[f"t_home_m{mm:02d}"])
        params[f"t_away_m{mm:02d}"] = float(best[f"t_away_m{mm:02d}"])
    return params, float(best["CVRMSE_%"]), float(best["NMBE_%"])


def write_final_idf(idd_path: Path, input_idf_path: Path, output_idf_path: Path,
                    measured_months: List[int], best_params: Dict[str, float]) -> None:
    IDF.setiddname(str(idd_path))
    idf = IDF(str(input_idf_path))

    ensure_schedule_type_limits(idf, "Any Number", None, None, "Continuous")
    ensure_schedule_type_limits(idf, "Fraction", 0.0, 1.0, "Continuous")

    apply_people_per_area_uniform(idf, best_params["people_per_m2"])

    month_to_home_away = {
        mm: (best_params[f"t_home_m{mm:02d}"], best_params[f"t_away_m{mm:02d}"])
        for mm in measured_months
    }
    create_schedule_compact_monthly_setpoints(
        idf, RADIANT_SETPOINT_SCHED_NAME, "Any Number", month_to_home_away)
    create_schedule_compact_monthly_setpoints(
        idf, AIR_HEATING_SETPOINT_SCHED_NAME, "Any Number", month_to_home_away)

    output_idf_path.parent.mkdir(parents=True, exist_ok=True)
    idf.saveas(str(output_idf_path))


# ============================================================
# Baseline report
# ============================================================
def baseline_report(idd_path: Path, input_idf_path: Path, epw_path: Path,
                    energyplus_exe: Path, work_dir: Path) -> Tuple[float, float]:
    baseline_dir = work_dir / "baseline"
    baseline_idf = baseline_dir / "baseline.idf"
    out_dir = baseline_dir / "out"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    IDF.setiddname(str(idd_path))
    idf = IDF(str(input_idf_path))
    idf.saveas(str(baseline_idf))

    run_energyplus(energyplus_exe, baseline_idf, epw_path, out_dir)
    sim_heat = read_sim_monthly_heating_kwh(out_dir)
    sim_monthly_kwh = heat_to_fuel(sim_heat, BOILER_EFF)
    cvr, nb, sim_used = compute_metrics(MEASURED_MONTHLY_KWH, sim_monthly_kwh)

    print(f"\n=== Baseline Monthly Comparison (kWh fuel, boiler eff {BOILER_EFF:.2f}) ===")
    print("Month | Measured | Simulated | Residual (Sim-Meas)")
    for m in MEASURED_MONTHS:
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
    energyplus_exe = guess_energyplus_exe(IDD_PATH)

    echo_inputs()
    print(f"[INFO] EnergyPlus exe : {energyplus_exe}")
    print(f"[INFO] Work dir       : {WORK_DIR.resolve()}")

    measured_months = MEASURED_MONTHS

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

    print("\n[INFO] Starting NSGA-II optimization (occupancy + monthly setpoints)...")
    minimize(problem, NSGA2(pop_size=POP_SIZE), get_termination("n_gen", N_GEN),
             seed=SEED, save_history=False, verbose=True)

    best_params, best_cvr, best_nb = select_best_from_log(log_csv, measured_months)

    print("\n=== Selected Best (lexicographic: CVRMSE then |NMBE|) ===")
    print(f"people_per_m2 = {best_params['people_per_m2']:.4f}")
    for mm in measured_months:
        print(f"  Month {mm:02d}: T_home={best_params[f't_home_m{mm:02d}']:.2f} degC, "
              f"T_away={best_params[f't_away_m{mm:02d}']:.2f} degC")
    print(f"CVRMSE = {best_cvr:.3f} %")
    print(f"NMBE   = {best_nb:.3f} %")
    print(f"[INFO] Full evaluation log: {log_csv.resolve()}")

    write_final_idf(IDD_PATH, INPUT_IDF_PATH, OUTPUT_IDF_PATH, measured_months, best_params)
    print(f"[OK] Calibrated IDF saved to: {OUTPUT_IDF_PATH.resolve()}")

    # ── Verification report and chart ────────────────────────────────────
    try:
        from cali_verification_report import run_verification
        _vdir = WORK_DIR / "best_verify"
        _vdir.mkdir(exist_ok=True)
        run_energyplus(energyplus_exe, OUTPUT_IDF_PATH, EPW_PATH, _vdir / "out")
        _heat = read_sim_monthly_heating_kwh(_vdir / "out")
        run_verification(
            stage_name="Stage 2 – Occupancy and monthly setpoints",
            idf_path=str(OUTPUT_IDF_PATH),
            epw_path=str(EPW_PATH),
            measured=MEASURED_MONTHLY_KWH,
            simulated_heat=_heat,
            boiler_eff=BOILER_EFF,
            chart_out=OUTPUT_IDF_PATH.parent / "stage2_comparison.png")
    except Exception as _e:
        print(f"[WARN] Verification report failed: {_e}")
    # ─────────────────────────────────────────────────────────────────────

    try:
        shutil.rmtree(WORK_DIR / "baseline", ignore_errors=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
