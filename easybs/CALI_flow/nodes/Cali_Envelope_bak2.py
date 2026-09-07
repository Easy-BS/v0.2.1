# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 16:43:46 2026

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


def _require(key: str, what: str):
    """Fetch a runtime value, or fail loudly.

    No silent defaults for experiment inputs. A missing measured dataset,
    weather file or model must stop the run rather than be replaced by a
    plausible-looking substitute, which would otherwise produce a calibrated
    model and a reported CVRMSE against data nobody supplied.
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

# The IDD keeps a default: it is a machine installation path, not experiment
# data, and its existence is verified in main().
IDD_PATH = Path(_RUNTIME.get("idd_path", r"C:/EnergyPlusV8-9-0/Energy+.idd"))
EPW_PATH = Path(_require("epw_path", "weather file for the measured year"))
INPUT_IDF_PATH = Path(_require("idf_path", "model to calibrate"))
PREPARED_IDF_PATH = Path("./Calibration/Prepared_Cali_RFH.idf")
OUTPUT_IDF_PATH = Path("./Calibration/After_Cali_RFH.idf")

WORK_DIR: Path = Path("./Calibration/_cali_runs")

POP_SIZE: int = 16
N_GEN: int = 8
SEED: int = 42

CLEANUP_EACH_RUN: bool = False
KEEP_FAILED_RUNS: bool = True

HEATING_METERS_PRIORITY: List[str] = [
    "DistrictHeating:Facility",
    "Electricity:Heating",
    "Gas:Heating",
]


# ============================================================
# Layer build-ups
# ============================================================
#
# Floor, ondol on grade, bottom to top as specified for the case building:
#     polyethylene film 0.05 mm      omitted, see note below
#     expanded polystyrene 30 mm     <- calibrated
#     reinforced concrete 250 mm
#     polyethylene film 0.05 mm      omitted
#     foamed concrete 60 mm
#     hydronic pipe, 16 mm OD at 250 mm spacing   <- source plane
#     cement mortar 40 mm            entirely above the pipe
#     sheet flooring 4 mm
#
# The two polyethylene films are omitted from the thermal model. At 0.05 mm
# their combined resistance is below 0.001 m2K/W, four orders of magnitude
# smaller than the insulation layer, and layers that thin degrade the
# EnergyPlus conduction transfer function calculation. They are vapour
# barriers, not thermal layers.
#
# Roof: a conventional Korean protected-membrane flat roof.

# name -> (roughness, thickness_m, conductivity, density, specific_heat)
MATERIAL_SPECS: Dict[str, tuple] = {
    # --- calibrated insulations (conductivity is a decision variable) ---
    "Wall_Insulation":  ("MediumRough", 0.080, 0.035,   30.0, 1400.0),
    "Roof_Insulation":  ("Rough",       0.050, 0.034,   30.0, 1400.0),
    "Floor_EPS":        ("Rough",       0.030, 0.034,   25.0, 1400.0),

    # --- fixed structural and finishing layers ---
    "RC_Slab_250":      ("MediumRough", 0.250, 1.600, 2300.0,  880.0),
    "RC_Slab_150":      ("MediumRough", 0.150, 1.600, 2300.0,  880.0),
    "Foam_Concrete_60": ("MediumRough", 0.060, 0.160,  500.0, 1000.0),
    "Foam_Concrete_30": ("MediumRough", 0.030, 0.160,  500.0, 1000.0),
    "Cement_Mortar_40": ("Smooth",      0.040, 1.400, 2000.0, 1000.0),
    "Cement_Mortar_30": ("Smooth",      0.030, 1.400, 2000.0, 1000.0),
    "Floor_Finish_4":   ("Smooth",      0.004, 0.190, 1200.0, 1200.0),
    "Ceiling_Gypsum":   ("Smooth",      0.0125, 0.180, 800.0, 1090.0),
}

# Construction:InternalSource for the heated floor. The pipe plane sits on
# top of the foamed concrete, so the source follows layer 3.
RFH_FLOOR_CONSTRUCTION = "Slab Floor with Radiant"   # name kept: surfaces reference it
RFH_FLOOR_LAYERS = ["Floor_EPS", "RC_Slab_250", "Foam_Concrete_60",
                    "Cement_Mortar_40", "Floor_Finish_4"]
RFH_SOURCE_AFTER_LAYER = 3
RFH_TEMPCALC_AFTER_LAYER = 4
RFH_TUBE_SPACING_M = 0.25
RFH_TUBE_INSIDE_DIAMETER_M = 0.012      # 16 mm OD, 2 mm wall

# Unheated ground floor: same stack without the pipe plane or foamed concrete
PLAIN_FLOOR_CONSTRUCTION = "Project Floor"
PLAIN_FLOOR_LAYERS = ["Floor_EPS", "RC_Slab_250",
                      "Cement_Mortar_40", "Floor_Finish_4"]

ROOF_CONSTRUCTION = "Project Flat Roof"
ROOF_LAYERS = ["Cement_Mortar_30", "Foam_Concrete_60", "Roof_Insulation",
               "RC_Slab_150", "Ceiling_Gypsum"]

# Calibration targets
WALL_INSUL_MAT_NAME = "Wall_Insulation"
ROOF_INSUL_MAT_NAME = "Roof_Insulation"
FLOOR_INSUL_MAT_NAME = "Floor_EPS"
WINDOW_SIMPLE_GLAZING_NAME = "SG_2p0"

# ------------------------------------------------------------------
# Absolute bounds, one range per material rather than a blanket multiplier
# ------------------------------------------------------------------
# k_wall_ins   expanded polystyrene or glass wool board, 0.031 to 0.043
#              nominal; widened slightly for ageing and workmanship
# k_roof_ins   same material class, same reasoning
# k_floor_ins  expanded polystyrene under slab, as above
# u_window     1.4 low-emissivity double glazing to 4.0 poor double glazing.
#              Single glazing (about 5.8) is excluded: the building has
#              double glazing on record.
# infil_ach    0.20 to 1.50 air changes per hour. The upper limit
#              corresponds to roughly 30 ACH at 50 Pa under the n50/20 rule,
#              which is at the leaky end of the reported range for Korean
#              detached housing of this vintage.
BOUNDS = {
    "k_wall_ins":  (0.028, 0.050),
    "k_roof_ins":  (0.028, 0.050),
    "k_floor_ins": (0.028, 0.050),
    "u_window":    (1.40,  4.00),
    "infil_ach":   (0.20,  1.50),
}

VAR_ORDER = ["k_wall_ins", "k_roof_ins", "k_floor_ins", "u_window", "infil_ach"]


# ============================================================
# Utilities
# ============================================================
def reset_calibration_workspace(work_dir: Path) -> None:
    if work_dir.exists():
        print(f"[INFO] Clearing previous calibration workspace: {work_dir.resolve()}")
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def set_field(obj, candidates: List[str], value, *, required: bool = True) -> str:
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


def _del_by_name(idf: IDF, key: str, name: str) -> None:
    nl = name.strip().lower()
    for o in list(idf.idfobjects.get(key, [])):
        if (getattr(o, "Name", "") or "").strip().lower() == nl:
            idf.removeidfobject(o)


# ============================================================
# Construction rebuild
# ============================================================
def ensure_material(idf: IDF, name: str) -> None:
    """Create or overwrite a MATERIAL from MATERIAL_SPECS."""
    roughness, thickness, k, rho, cp = MATERIAL_SPECS[name]
    _del_by_name(idf, "MATERIAL", name)
    m = idf.newidfobject("MATERIAL")
    m.Name = name
    m.Roughness = roughness
    m.Thickness = thickness
    m.Conductivity = k
    m.Density = rho
    m.Specific_Heat = cp
    m.Thermal_Absorptance = 0.9
    m.Solar_Absorptance = 0.7
    m.Visible_Absorptance = 0.7


def ensure_construction(idf: IDF, name: str, layers: List[str]) -> None:
    """Create or overwrite a plain CONSTRUCTION. Layers run outside to inside."""
    _del_by_name(idf, "CONSTRUCTION", name)
    c = idf.newidfobject("CONSTRUCTION")
    c.Name = name
    c.Outside_Layer = layers[0]
    for i, lyr in enumerate(layers[1:], start=2):
        setattr(c, f"Layer_{i}", lyr)


def ensure_internal_source_construction(idf: IDF, name: str, layers: List[str],
                                        source_after: int, tempcalc_after: int,
                                        tube_spacing: float) -> None:
    """Create or overwrite a CONSTRUCTION:INTERNALSOURCE."""
    _del_by_name(idf, "CONSTRUCTION:INTERNALSOURCE", name)
    c = idf.newidfobject("CONSTRUCTION:INTERNALSOURCE")
    c.Name = name
    c.Source_Present_After_Layer_Number = source_after
    c.Temperature_Calculation_Requested_After_Layer_Number = tempcalc_after
    # One-dimensional conduction transfer functions. Two-dimensional would be
    # more accurate at 250 mm tube spacing, but costs an order of magnitude
    # more runtime across several hundred calibration evaluations.
    c.Dimensions_for_the_CTF_Calculation = 1
    c.Tube_Spacing = tube_spacing
    c.Outside_Layer = layers[0]
    for i, lyr in enumerate(layers[1:], start=2):
        setattr(c, f"Layer_{i}", lyr)


def rebuild_envelope_constructions(idf: IDF) -> None:
    """Replace the placeholder floor and roof with documented build-ups.

    Construction names are preserved so that existing
    BuildingSurface:Detailed references remain valid.
    """
    for name in MATERIAL_SPECS:
        ensure_material(idf, name)

    ensure_internal_source_construction(
        idf, RFH_FLOOR_CONSTRUCTION, RFH_FLOOR_LAYERS,
        RFH_SOURCE_AFTER_LAYER, RFH_TEMPCALC_AFTER_LAYER, RFH_TUBE_SPACING_M)

    ensure_construction(idf, PLAIN_FLOOR_CONSTRUCTION, PLAIN_FLOOR_LAYERS)
    ensure_construction(idf, ROOF_CONSTRUCTION, ROOF_LAYERS)

    # Exterior wall keeps its layer order but uses the renamed insulation
    wall = next((c for c in idf.idfobjects.get("CONSTRUCTION", [])
                 if (c.Name or "").strip().lower() == "exterior_wall_construction"), None)
    if wall is not None:
        for fld in ("Outside_Layer", "Layer_2", "Layer_3", "Layer_4"):
            if (getattr(wall, fld, "") or "").strip() == "Ext_Insul":
                setattr(wall, fld, WALL_INSUL_MAT_NAME)

    # Radiant device: match the specified tube diameter
    for r in idf.idfobjects.get("ZONEHVAC:LOWTEMPERATURERADIANT:VARIABLEFLOW", []):
        set_field(r, ["Hydronic_Tubing_Inside_Diameter"],
                  RFH_TUBE_INSIDE_DIAMETER_M, required=False)

    print("[INFO] Rebuilt constructions:")
    print(f"  {RFH_FLOOR_CONSTRUCTION}: " + " | ".join(RFH_FLOOR_LAYERS)
          + f"  (source after layer {RFH_SOURCE_AFTER_LAYER}, "
            f"tube spacing {RFH_TUBE_SPACING_M} m)")
    print(f"  {PLAIN_FLOOR_CONSTRUCTION}: " + " | ".join(PLAIN_FLOOR_LAYERS))
    print(f"  {ROOF_CONSTRUCTION}: " + " | ".join(ROOF_LAYERS))


def prepare_idf(idd_path: Path, input_idf_path: Path, prepared_path: Path) -> Path:
    """Write a copy of the input model with the rebuilt constructions."""
    IDF.setiddname(str(idd_path))
    idf = IDF(str(input_idf_path))
    rebuild_envelope_constructions(idf)
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    idf.saveas(str(prepared_path))
    print(f"[OK] Prepared IDF written to: {prepared_path.resolve()}")
    return prepared_path


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
# Metrics
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
# Monthly meters from eplusout.mtr
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
            meters_monthly[monthly_idx_to_name[idx]][current_month] = val_j

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
    return {mm: (val_j / 3.6e6) for mm, val_j in monthly_j.items()}


# ============================================================
# IDF editing
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
    infil_objs = idf.idfobjects.get("ZONEINFILTRATION:DESIGNFLOWRATE", [])
    if not infil_objs:
        raise RuntimeError("No ZONEINFILTRATION:DESIGNFLOWRATE objects found.")
    base_ach = getattr(infil_objs[0], "Air_Changes_per_Hour", None)
    try:
        base_ach = float(base_ach)
    except Exception:
        base_ach = float("nan")

    return {
        "k_wall_ins": float(_get_material(idf, WALL_INSUL_MAT_NAME).Conductivity),
        "k_roof_ins": float(_get_material(idf, ROOF_INSUL_MAT_NAME).Conductivity),
        "k_floor_ins": float(_get_material(idf, FLOOR_INSUL_MAT_NAME).Conductivity),
        "u_window": float(_get_simple_glazing(idf, WINDOW_SIMPLE_GLAZING_NAME).UFactor),
        "infil_ach": base_ach,
    }


def apply_envelope_params(idf: IDF, k_wall_ins: float, k_roof_ins: float,
                          k_floor_ins: float, u_window: float) -> None:
    """Set absolute material properties. No multipliers, no hidden baselines."""
    _get_material(idf, WALL_INSUL_MAT_NAME).Conductivity = float(k_wall_ins)
    _get_material(idf, ROOF_INSUL_MAT_NAME).Conductivity = float(k_roof_ins)
    _get_material(idf, FLOOR_INSUL_MAT_NAME).Conductivity = float(k_floor_ins)
    _get_simple_glazing(idf, WINDOW_SIMPLE_GLAZING_NAME).UFactor = float(u_window)


def apply_global_infiltration_ach(idf: IDF, infil_ach: float) -> None:
    objs = idf.idfobjects.get("ZONEINFILTRATION:DESIGNFLOWRATE", [])
    if not objs:
        raise RuntimeError("No ZONEINFILTRATION:DESIGNFLOWRATE objects found.")
    for zinf in objs:
        set_field(zinf,
                  ["Design_Flow_Rate_Calculation_Method", "DesignFlowRateCalculationMethod"],
                  "AirChanges/Hour", required=False)
        set_field(zinf,
                  ["Air_Changes_per_Hour", "AirChangesperHour", "Air Changes per Hour"],
                  float(infil_ach), required=True)


# ============================================================
# Evaluation
# ============================================================
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
    return (cvrmse_percent(meas, sim), nmbe_percent(meas, sim),
            {m: float(simulated_kwh[m]) for m in months})


class EnvelopeInfilCalibrationProblem(ElementwiseProblem):
    def __init__(self, idd_path: Path, input_idf_path: Path, epw_path: Path,
                 work_dir: Path, measured_monthly_kwh: Dict[str, float],
                 energyplus_exe: Path, base_idf_params: Dict[str, float],
                 log_csv_path: Path):

        xl = np.array([BOUNDS[v][0] for v in VAR_ORDER], dtype=float)
        xu = np.array([BOUNDS[v][1] for v in VAR_ORDER], dtype=float)
        super().__init__(n_var=len(VAR_ORDER), n_obj=2, xl=xl, xu=xu)

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
                csv.writer(f).writerow(
                    ["eval_id"] + VAR_ORDER
                    + ["CVRMSE_%", "NMBE_%", "absNMBE_%", "runtime_s"])

    def _evaluate(self, x, out, *args, **kwargs):
        self.eval_counter += 1
        eval_id = self.eval_counter

        vals = dict(zip(VAR_ORDER, map(float, x.tolist())))

        run_dir = self.work_dir / f"run_{eval_id:05d}"
        run_idf = run_dir / "in.idf"
        out_dir = run_dir / "out"
        run_dir.mkdir(parents=True, exist_ok=True)

        IDF.setiddname(str(self.idd_path))
        idf = IDF(str(self.input_idf_path))

        apply_envelope_params(idf, vals["k_wall_ins"], vals["k_roof_ins"],
                              vals["k_floor_ins"], vals["u_window"])
        apply_global_infiltration_ach(idf, vals["infil_ach"])
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

        # Objectives are the metrics themselves. Every point the optimizer can
        # reach lies within a documented material range, so no plausibility
        # penalty is required and the reported metrics are the optimized ones.
        out["F"] = np.array([cvr, absnb], dtype=float)

        with self.log_csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [eval_id] + [vals[v] for v in VAR_ORDER] + [cvr, nb, absnb, runtime])

        if CLEANUP_EACH_RUN:
            if failed and KEEP_FAILED_RUNS:
                return
            shutil.rmtree(run_dir, ignore_errors=True)


# ============================================================
# Main workflow
# ============================================================
def baseline_report(idd_path: Path, input_idf_path: Path, epw_path: Path,
                    energyplus_exe: Path) -> Tuple[float, float]:
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

    print("\n=== Baseline Monthly Comparison (kWh) ===")
    print("Month | Measured | Simulated | Residual (Sim-Meas)")
    for m in sorted(int(k) for k in MEASURED_MONTHLY_KWH.keys()):
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
    if not log_csv.exists():
        raise FileNotFoundError(f"Log CSV not found: {log_csv.resolve()}")

    try:
        df = pd.read_csv(log_csv, engine="python", on_bad_lines="skip")
    except TypeError:
        df = pd.read_csv(log_csv, engine="python", error_bad_lines=False, warn_bad_lines=True)

    if df.empty:
        raise RuntimeError(f"Log CSV is empty or all lines were malformed: {log_csv.resolve()}")

    missing = [c for c in ("CVRMSE_%", "NMBE_%", "absNMBE_%") if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns {missing} in log. Found: {list(df.columns)}")

    for c in VAR_ORDER + ["CVRMSE_%", "NMBE_%", "absNMBE_%"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=VAR_ORDER + ["CVRMSE_%", "absNMBE_%"])

    if df.empty:
        raise RuntimeError("No valid evaluation rows remain after cleaning malformed/NaN rows.")

    df = df.sort_values(["CVRMSE_%", "absNMBE_%"], ascending=[True, True]).reset_index(drop=True)
    best = df.iloc[0].to_dict()
    params = {v: float(best[v]) for v in VAR_ORDER}
    return params, float(best["CVRMSE_%"]), float(best["NMBE_%"])


def write_final_idf_with_best_params(idd_path: Path, input_idf_path: Path,
                                     output_idf_path: Path,
                                     best_params: Dict[str, float]) -> None:
    IDF.setiddname(str(idd_path))
    idf = IDF(str(input_idf_path))
    apply_envelope_params(idf, best_params["k_wall_ins"], best_params["k_roof_ins"],
                          best_params["k_floor_ins"], best_params["u_window"])
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
    energyplus_exe = guess_energyplus_exe(IDD_PATH)

    # Echo the inputs actually received, so the log records what was
    # calibrated against rather than what was assumed.
    print(f"[INFO] EnergyPlus exe : {energyplus_exe}")
    print(f"[INFO] Input model    : {INPUT_IDF_PATH}")
    print(f"[INFO] Weather file   : {EPW_PATH.name}")
    print(f"[INFO] Measured months: "
          f"{sorted(int(m) for m in MEASURED_MONTHLY_KWH)}  "
          f"(total {sum(MEASURED_MONTHLY_KWH.values()):,.0f} kWh)")

    # Rebuild the floor and roof once, then calibrate on the prepared model
    prepared = prepare_idf(IDD_PATH, INPUT_IDF_PATH, PREPARED_IDF_PATH)

    baseline_report(IDD_PATH, prepared, EPW_PATH, energyplus_exe)

    IDF.setiddname(str(IDD_PATH))
    base_params = read_base_params_from_idf(IDF(str(prepared)))
    print("[INFO] Base parameters (from prepared IDF):")
    for k, v in base_params.items():
        lo, hi = BOUNDS[k]
        print(f"  - {k}: {v}   bounds [{lo}, {hi}]")

    log_csv = WORK_DIR / "eval_log.csv"
    problem = EnvelopeInfilCalibrationProblem(
        idd_path=IDD_PATH,
        input_idf_path=prepared,
        epw_path=EPW_PATH,
        work_dir=WORK_DIR,
        measured_monthly_kwh=MEASURED_MONTHLY_KWH,
        energyplus_exe=energyplus_exe,
        base_idf_params=base_params,
        log_csv_path=log_csv,
    )

    print("\n[INFO] Starting NSGA-II optimization (Envelope + Infiltration)...")
    minimize(problem, NSGA2(pop_size=POP_SIZE), get_termination("n_gen", N_GEN),
             seed=SEED, save_history=False, verbose=True)

    best_params, best_cvr, best_nb = select_best_from_log(log_csv)
    print("\n=== Selected Best (lexicographic: CVRMSE then |NMBE|) ===")
    for k, v in best_params.items():
        lo, hi = BOUNDS[k]
        pos = 100.0 * (v - lo) / (hi - lo)
        flag = "  <-- at bound" if pos <= 2.0 or pos >= 98.0 else ""
        print(f"  {k} = {v:.4f}   [{lo}, {hi}]   {pos:5.1f}% of range{flag}")
    print(f"CVRMSE = {best_cvr:.3f} %")
    print(f"NMBE   = {best_nb:.3f} %")
    print(f"[INFO] Full evaluation log saved to: {log_csv.resolve()}")

    write_final_idf_with_best_params(IDD_PATH, prepared, OUTPUT_IDF_PATH, best_params)
    print(f"[OK] Calibrated IDF saved to: {OUTPUT_IDF_PATH.resolve()}")


if __name__ == "__main__":
    main()