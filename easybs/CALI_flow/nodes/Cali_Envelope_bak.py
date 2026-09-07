# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 14:00:46 2026

@author: Xiguan Liang @SKKU
"""


# ./CALI_flow/nodes/Cali_Envelope.py
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
INPUT_IDF_PATH = Path(_RUNTIME.get("idf_path", "./Calibration/Ready2_Cali_RFH.idf"))
OUTPUT_IDF_PATH = Path("./Calibration/After_Cali_RFH.idf")

# Working directory for repeated simulations
WORK_DIR: Path = Path("./Calibration/_cali_runs")

# NSGA-II settings (start small; EnergyPlus evaluations are expensive)
POP_SIZE: int = 16
N_GEN: int = 8
SEED: int = 42

# Cleanup policy
CLEANUP_EACH_RUN: bool = False #True          # delete run_00XXX after each evaluation
KEEP_FAILED_RUNS: bool = True #False         # if True, keep run folders when E+ fails (debugging)

# Choose which meters to use for "heating energy"
# We will sum what is available among these (Monthly, J) and convert to kWh.
HEATING_METERS_PRIORITY: List[str] = [
    "DistrictHeating:Facility",
    "Electricity:Heating",
    "Gas:Heating",
]


# ---------------------------
# Calibration targets in your IDF (from your snippets)
# ---------------------------
WALL_INSUL_MAT_NAME = "Ext_Insul"
ROOF_MAT_NAME = "DefaultMaterial"
FLOOR_INSUL_MAT_NAME = "INS - EXPANDED EXT POLYSTYRENE R12 2 IN"
WINDOW_SIMPLE_GLAZING_NAME = "SG_2p0"

# Bounds on multipliers (broad but physically reasonable for quick-mode calibration)
BOUNDS = { "m_wall_k": (0.5, 2.0), 
          "m_roof_k": (0.3, 3.0),
          "m_floor_k": (0.5, 2.0), 
          "m_window_u": (0.6, 2.5), 
          "infil_ach": (0.10, 2.0), }


# ============================================================
# Utilities: field-safe setter (handles E+8.9 naming variations)
# ============================================================
def reset_calibration_workspace(work_dir: Path) -> None:
    """
    Delete all previous calibration runs to ensure a clean optimization start.
    This removes old run_xxxxx folders and eval_log.csv.
    """
    if work_dir.exists():
        print(f"[INFO] Clearing previous calibration workspace: {work_dir.resolve()}")
        shutil.rmtree(work_dir)

    work_dir.mkdir(parents=True, exist_ok=True)

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def set_field(obj, candidates: List[str], value, *, required: bool = True) -> str:
    """
    Set a field on an eppy object, robust to fieldname variations in IDD.
    candidates: list of possible field names.
    """
    fieldnames = getattr(obj, "fieldnames", [])
    fmap = {_norm(fn): fn for fn in fieldnames if fn and fn.lower() != "key"}
    for cand in candidates:
        key = _norm(cand)
        if key in fmap:
            real = fmap[key]
            setattr(obj, real, value)
            return real
    if not required:
        return ""
    raise ValueError(
        f"Cannot find any of candidate fields {candidates} in object '{obj.key}'. "
        f"Available fields: {fieldnames}"
    )


#%% ============================================================
# Meter CSV helpers
# ============================================================
MONTH_NAME_TO_INT = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def _month_cell_to_int(x) -> int | None:
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


# ============================================================
# Metrics: ASHRAE Guideline 14 style (monthly)
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


# ============================================================
# EnergyPlus runner
# ============================================================
def guess_energyplus_exe(idd_path: Path) -> Path:
    candidate = idd_path.parent / "energyplus.exe"
    if candidate.exists():
        return candidate
    return Path("energyplus")


def run_energyplus(energyplus_exe: Path, idf_path: Path, epw_path: Path, out_dir: Path, timeout_s: int = 3600) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [str(energyplus_exe), "-w", str(epw_path), "-d", str(out_dir), str(idf_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        msg = (proc.stdout[-4000:] if proc.stdout else "") + "\n" + (proc.stderr[-4000:] if proc.stderr else "")
        raise RuntimeError(f"EnergyPlus failed (code {proc.returncode}). Tail output:\n{msg}")


# ============================================================
# Parse monthly meters (J) from eplusout.mtr (E+ 8.9 style)
# ============================================================
def read_monthly_meter_j_from_mtr(mtr_path: Path) -> Dict[str, Dict[int, float]]:
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
        raise RuntimeError(
            "No Monthly meters detected in eplusout.mtr dictionary. "
            "Check that your Output:Meter objects request Monthly frequency."
        )

    meters_monthly: Dict[str, Dict[int, float]] = {n: {} for n in monthly_idx_to_name.values()}
    current_month: Optional[int] = None

    for line in data_lines:
        s = line.strip()
        if not s:
            continue
        parts = [p.strip() for p in s.split(",")]

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
            mname = monthly_idx_to_name[idx]
            meters_monthly[mname][current_month] = val_j

    meters_monthly = {k: v for k, v in meters_monthly.items() if v}
    if not meters_monthly:
        raise RuntimeError("Monthly meter indices found, but no monthly values parsed from data section.")
    return meters_monthly


def read_sim_monthly_heating_kwh(out_dir: Path) -> Dict[int, float]:
    meter_csvs = sorted(out_dir.glob("*Meter.csv"))
    if meter_csvs:
        meters = read_monthly_meter_j_from_meter_csv(meter_csvs[0])
    else:
        mtr = out_dir / "eplusout.mtr"
        if mtr.exists():
            meters = read_monthly_meter_j_from_mtr(mtr)
        else:
            raise RuntimeError("No *Meter.csv and no eplusout.mtr found in output directory.")

    monthly_j: Dict[int, float] = {}
    found_any = False

    for meter_name in HEATING_METERS_PRIORITY:
        key = None
        for k in meters.keys():
            if k.strip().lower() == meter_name.strip().lower():
                key = k
                break
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

    return {mm: (val_j / 3.6e6) for mm, val_j in monthly_j.items()}


# ============================================================
# IDF editing utilities (eppy)
# ============================================================
def _get_material(idf: IDF, name: str):
    for m in idf.idfobjects.get("MATERIAL", []):
        if getattr(m, "Name", "").strip() == name:
            return m
    raise KeyError(f"Material not found: {name}")


def _get_simple_glazing(idf: IDF, name: str):
    for o in idf.idfobjects.get("WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM", []):
        if getattr(o, "Name", "").strip() == name:
            return o
    raise KeyError(f"WindowMaterial:SimpleGlazingSystem not found: {name}")


def read_base_params_from_idf(idf: IDF) -> Dict[str, float]:
    wall = _get_material(idf, WALL_INSUL_MAT_NAME)
    roof = _get_material(idf, ROOF_MAT_NAME)
    floor = _get_material(idf, FLOOR_INSUL_MAT_NAME)
    win = _get_simple_glazing(idf, WINDOW_SIMPLE_GLAZING_NAME)

    # Infiltration baseline: take first ZoneInfiltration object ACH
    infil_objs = idf.idfobjects.get("ZONEINFILTRATION:DESIGNFLOWRATE", [])
    if not infil_objs:
        raise RuntimeError("No ZONEINFILTRATION:DESIGNFLOWRATE objects found. Add infiltration first.")
    first = infil_objs[0]
    base_ach = getattr(first, "Air_Changes_per_Hour", None)
    if base_ach is None:
        base_ach = getattr(first, "AirChangesperHour", None)
    if base_ach is None or str(base_ach).strip() == "":
        base_ach = float("nan")
    else:
        base_ach = float(base_ach)

    return {
        "wall_k": float(getattr(wall, "Conductivity")),
        "roof_k": float(getattr(roof, "Conductivity")),
        "floor_k": float(getattr(floor, "Conductivity")),
        "window_u": float(getattr(win, "UFactor")),
        "infil_ach": base_ach,
    }


def apply_envelope_params(idf: IDF, m_wall_k: float, m_roof_k: float, m_floor_k: float, m_window_u: float) -> None:
    wall = _get_material(idf, WALL_INSUL_MAT_NAME)
    roof = _get_material(idf, ROOF_MAT_NAME)
    floor = _get_material(idf, FLOOR_INSUL_MAT_NAME)
    win = _get_simple_glazing(idf, WINDOW_SIMPLE_GLAZING_NAME)

    wall_base = float(getattr(wall, "Conductivity"))
    roof_base = float(getattr(roof, "Conductivity"))
    floor_base = float(getattr(floor, "Conductivity"))
    win_base = float(getattr(win, "UFactor"))

    wall.Conductivity = wall_base * float(m_wall_k)
    roof.Conductivity = roof_base * float(m_roof_k)
    floor.Conductivity = floor_base * float(m_floor_k)
    win.UFactor = win_base * float(m_window_u)


def apply_global_infiltration_ach(idf: IDF, infil_ach: float) -> None:
    """
    Standardize infiltration: set every ZONEINFILTRATION:DESIGNFLOWRATE object to the same ACH,
    keep schedule as-is (ON), keep coefficients as-is.
    """
    objs = idf.idfobjects.get("ZONEINFILTRATION:DESIGNFLOWRATE", [])
    if not objs:
        raise RuntimeError("No ZONEINFILTRATION:DESIGNFLOWRATE objects found. Add infiltration first.")

    for zinf in objs:
        # Ensure method is AirChanges/Hour (robust to IDD naming)
        set_field(
            zinf,
            ["Design_Flow_Rate_Calculation_Method", "DesignFlowRateCalculationMethod"],
            "AirChanges/Hour",
            required=False,
        )

        # Set ACH (field name varies by IDD)
        set_field(
            zinf,
            ["Air_Changes_per_Hour", "AirChangesperHour", "Air Changes per Hour"],
            float(infil_ach),
            required=True,
        )


# ============================================================
# Evaluation: simulate & compute metrics
# ============================================================
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
# Pymoo Problem (Envelope + Infiltration)
# ============================================================
class EnvelopeInfilCalibrationProblem(ElementwiseProblem):
    def __init__(
        self,
        idd_path: Path,
        input_idf_path: Path,
        epw_path: Path,
        work_dir: Path,
        measured_monthly_kwh: Dict[str, float],
        energyplus_exe: Path,
        base_idf_params: Dict[str, float],
        log_csv_path: Path,
    ):
        xl = np.array([
            BOUNDS["m_wall_k"][0],
            BOUNDS["m_roof_k"][0],
            BOUNDS["m_floor_k"][0],
            BOUNDS["m_window_u"][0],
            BOUNDS["infil_ach"][0],
        ], dtype=float)

        xu = np.array([
            BOUNDS["m_wall_k"][1],
            BOUNDS["m_roof_k"][1],
            BOUNDS["m_floor_k"][1],
            BOUNDS["m_window_u"][1],
            BOUNDS["infil_ach"][1],
        ], dtype=float)

        super().__init__(n_var=5, n_obj=2, xl=xl, xu=xu)

        self.idd_path = idd_path
        self.input_idf_path = input_idf_path
        self.epw_path = epw_path
        self.work_dir = work_dir
        self.measured = measured_monthly_kwh
        self.energyplus_exe = energyplus_exe
        self.base_idf_params = base_idf_params
        self.log_csv_path = log_csv_path

        self.eval_counter = 0

        if not self.log_csv_path.exists():
            self.log_csv_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "eval_id",
                    "m_wall_k", "m_roof_k", "m_floor_k", "m_window_u", "infil_ach",
                    "CVRMSE_%", "NMBE_%", "absNMBE_%",
                    "runtime_s",
                ])

    def _evaluate(self, x, out, *args, **kwargs):
        self.eval_counter += 1
        eval_id = self.eval_counter

        m_wall_k, m_roof_k, m_floor_k, m_window_u, infil_ach = map(float, x.tolist())

        run_dir = self.work_dir / f"run_{eval_id:05d}"
        run_idf = run_dir / "in.idf"
        out_dir = run_dir / "out"

        run_dir.mkdir(parents=True, exist_ok=True)

        IDF.setiddname(str(self.idd_path))
        idf = IDF(str(self.input_idf_path))

        # Apply envelope multipliers
        apply_envelope_params(idf, m_wall_k, m_roof_k, m_floor_k, m_window_u)

        # Apply standardized infiltration (global ACH)
        apply_global_infiltration_ach(idf, infil_ach)

        idf.saveas(str(run_idf))

        t0 = time.time()
        failed = False
        try:
            run_energyplus(self.energyplus_exe, run_idf, self.epw_path, out_dir)
            sim_monthly_kwh = read_sim_monthly_heating_kwh(out_dir)
            cvr, nb, _ = compute_metrics(self.measured, sim_monthly_kwh)
            absnb = abs(nb)
        except Exception:
            failed = True
            cvr, nb, absnb = 1e6, 1e6, 1e6
        runtime = time.time() - t0
        
        def soft_penalty(params: dict) -> float:
            """
            Returns a non-negative penalty (in 'percentage points') to add to objectives.
            You can tune weights later.
            """
            p = 0.0
        
            # ---- Infiltration plausibility (ACH) ----
            ach = params["infil_ach"]
            # No penalty up to 1.5 ACH, mild above, stronger above 3.0
            if ach > 1.5:
                p += 5.0 * ((ach - 1.5) / 1.5) ** 2
            if ach > 3.0:
                p += 10.0 * ((ach - 3.0) / 0.5) ** 2
        
            # ---- Window U plausibility (multiplier) ----
            # If base U=2.0, then m=3.2 => U=6.4 (already very poor).
            mw = params["m_window_u"]
            if mw > 2.2:
                p += 3.0 * ((mw - 2.2) / 1.0) ** 2
        
            # ---- Insulation conductivity multipliers ----
            # Penalize very high k multipliers (i.e., "insulation" acting like solid material).
            for key, thresh, w in [
                ("m_wall_k", 4.0, 2.0),
                ("m_roof_k", 5.0, 2.5),
                ("m_floor_k", 4.0, 2.0),
            ]:
                x = params[key]
                if x > thresh:
                    p += w * ((x - thresh) / (thresh)) ** 2
        
            return p


        #out["F"] = np.array([cvr, absnb], dtype=float)
        #-------------------------------------------------
        params = {
            "m_wall_k": m_wall_k,
            "m_roof_k": m_roof_k,
            "m_floor_k": m_floor_k,
            "m_window_u": m_window_u,
            "infil_ach": infil_ach,
        }
        
        pen = soft_penalty(params)
        
        # Soft-penalized objectives
        cvr_p = cvr + pen
        absnb_p = absnb + 0.5 * pen  # often penalize bias less than spread
        
        out["F"] = np.array([cvr_p, absnb_p], dtype=float)
        #-------------------------------------------------

        with self.log_csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([eval_id, m_wall_k, m_roof_k, m_floor_k, m_window_u, infil_ach, cvr, nb, absnb, runtime])

        # Cleanup (requested)
        if CLEANUP_EACH_RUN:
            if failed and KEEP_FAILED_RUNS:
                return
            shutil.rmtree(run_dir, ignore_errors=True)


# ============================================================
# Main workflow
# ============================================================
def baseline_report(idd_path: Path, input_idf_path: Path, epw_path: Path, energyplus_exe: Path) -> Tuple[float, float]:
    baseline_dir = WORK_DIR / "baseline"
    baseline_idf = baseline_dir / "baseline.idf"
    out_dir = baseline_dir / "out"

    baseline_dir.mkdir(parents=True, exist_ok=True)

    IDF.setiddname(str(idd_path))
    idf = IDF(str(input_idf_path))
    idf.saveas(str(baseline_idf))

    run_energyplus(energyplus_exe, baseline_idf, epw_path, out_dir)

    sim_monthly_kwh = read_sim_monthly_heating_kwh(out_dir)
    cvr, nb, sim_used = compute_metrics(MEASURED_MONTHLY_KWH, sim_monthly_kwh)

    months = sorted(int(k) for k in MEASURED_MONTHLY_KWH.keys())
    print("\n=== Baseline Monthly Comparison (kWh) ===")
    print("Month | Measured | Simulated | Residual (Sim-Meas)")
    for m in months:
        meas = MEASURED_MONTHLY_KWH.get(str(m))
        sim = sim_used.get(m, float("nan"))
        if meas is None:
            continue
        print(f"{m:>5} | {meas:>8.2f} | {sim:>9.2f} | {sim - float(meas):>+12.2f}")

    print("\n=== Baseline Metrics (Monthly) ===")
    print(f"CVRMSE: {cvr:.3f} %")
    print(f"NMBE  : {nb:.3f} %\n")
    return cvr, nb


def select_best_from_log(log_csv: Path) -> Tuple[Dict[str, float], float, float]:
    """
    Robust reader:
    - Skips malformed lines (common if run was interrupted mid-write)
    - Forces numeric coercion
    - Drops incomplete rows
    """
    if not log_csv.exists():
        raise FileNotFoundError(f"Log CSV not found: {log_csv.resolve()}")

    # Pandas >= 1.3: on_bad_lines available (engine='python' is most tolerant)
    try:
        df = pd.read_csv(log_csv, engine="python", on_bad_lines="skip")
    except TypeError:
        # Older pandas fallback
        df = pd.read_csv(log_csv, engine="python", error_bad_lines=False, warn_bad_lines=True)

    if df.empty:
        raise RuntimeError(f"Log CSV is empty or all lines were malformed: {log_csv.resolve()}")

    required_cols = ["CVRMSE_%", "NMBE_%", "absNMBE_%"]
    for c in required_cols:
        if c not in df.columns:
            raise RuntimeError(f"Missing column '{c}' in log. Found columns: {list(df.columns)}")

    # Coerce numerics; drop any broken rows
    for c in ["m_wall_k", "m_roof_k", "m_floor_k", "m_window_u", "infil_ach", "CVRMSE_%", "NMBE_%", "absNMBE_%"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["m_wall_k", "m_roof_k", "m_floor_k", "m_window_u", "infil_ach", "CVRMSE_%", "absNMBE_%"])

    if df.empty:
        raise RuntimeError("No valid evaluation rows remain after cleaning malformed/NaN rows.")

    df = df.sort_values(["CVRMSE_%", "absNMBE_%"], ascending=[True, True]).reset_index(drop=True)
    best = df.iloc[0].to_dict()

    params = {
        "m_wall_k": float(best["m_wall_k"]),
        "m_roof_k": float(best["m_roof_k"]),
        "m_floor_k": float(best["m_floor_k"]),
        "m_window_u": float(best["m_window_u"]),
        "infil_ach": float(best["infil_ach"]),
    }
    return params, float(best["CVRMSE_%"]), float(best["NMBE_%"])


def write_final_idf_with_best_params(
    idd_path: Path,
    input_idf_path: Path,
    output_idf_path: Path,
    best_params: Dict[str, float],
) -> None:
    IDF.setiddname(str(idd_path))
    idf = IDF(str(input_idf_path))

    apply_envelope_params(
        idf,
        best_params["m_wall_k"],
        best_params["m_roof_k"],
        best_params["m_floor_k"],
        best_params["m_window_u"],
    )
    apply_global_infiltration_ach(idf, best_params["infil_ach"])

    output_idf_path.parent.mkdir(parents=True, exist_ok=True)
    idf.saveas(str(output_idf_path))


def main() -> None:
    if not IDD_PATH.exists():
        raise FileNotFoundError(f"IDD not found: {IDD_PATH.resolve()}")
    if not EPW_PATH.exists():
        raise FileNotFoundError(f"EPW not found: {EPW_PATH.resolve()}")
    if not INPUT_IDF_PATH.exists():
        raise FileNotFoundError(f"IDF not found: {INPUT_IDF_PATH.resolve()}")
    
    reset_calibration_workspace(WORK_DIR)
    
    shutil.rmtree(WORK_DIR, ignore_errors=True); WORK_DIR.mkdir(parents=True, exist_ok=True)
    energyplus_exe = guess_energyplus_exe(IDD_PATH)
    print(f"[INFO] EnergyPlus exe: {energyplus_exe}")

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    baseline_report(IDD_PATH, INPUT_IDF_PATH, EPW_PATH, energyplus_exe)

    IDF.setiddname(str(IDD_PATH))
    base_idf = IDF(str(INPUT_IDF_PATH))
    base_params = read_base_params_from_idf(base_idf)
    print("[INFO] Base parameters (from IDF):")
    for k, v in base_params.items():
        print(f"  - {k}: {v}")

    log_csv = WORK_DIR / "eval_log.csv"
    problem = EnvelopeInfilCalibrationProblem(
        idd_path=IDD_PATH,
        input_idf_path=INPUT_IDF_PATH,
        epw_path=EPW_PATH,
        work_dir=WORK_DIR,
        measured_monthly_kwh=MEASURED_MONTHLY_KWH,
        energyplus_exe=energyplus_exe,
        base_idf_params=base_params,
        log_csv_path=log_csv,
    )

    algorithm = NSGA2(pop_size=POP_SIZE)
    termination = get_termination("n_gen", N_GEN)

    print("\n[INFO] Starting NSGA-II optimization (Envelope + Infiltration)...")
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=SEED,
        save_history=False,
        verbose=True,
    )

    best_params, best_cvr, best_nb = select_best_from_log(log_csv)
    print("\n=== Selected Best (single-point pick) ===")
    print("Params:")
    for k, v in best_params.items():
        print(f"  {k} = {v:.4f}")
    print(f"CVRMSE = {best_cvr:.3f} %")
    print(f"NMBE   = {best_nb:.3f} %")
    print(f"[INFO] Full evaluation log saved to: {log_csv.resolve()}")

    write_final_idf_with_best_params(IDD_PATH, INPUT_IDF_PATH, OUTPUT_IDF_PATH, best_params)
    print(f"[OK] Calibrated IDF saved to: {OUTPUT_IDF_PATH.resolve()}")


if __name__ == "__main__":
    main()

