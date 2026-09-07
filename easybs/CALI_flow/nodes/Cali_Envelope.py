# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 20:39:13 2026

@author: Xiguan Liang @SKKU
"""

# ./CALI_flow/nodes/Cali_Envelope.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
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


# Measured monthly heating fuel in kWh, keyed by month number as a string.
MEASURED_MONTHLY_KWH: Dict[str, float] = {
    str(int(k)): float(v)
    for k, v in _require("measured_monthly_kwh",
                         "measured monthly heating fuel in kWh").items()
}
if len(MEASURED_MONTHLY_KWH) < 2:
    raise RuntimeError(
        "At least two measured months are required for CVRMSE and NMBE; got "
        f"{sorted(int(m) for m in MEASURED_MONTHLY_KWH)}."
    )
MEASURED_MONTHS: List[int] = sorted(int(m) for m in MEASURED_MONTHLY_KWH)

IDD_PATH = Path(_RUNTIME.get("idd_path", r"C:/EnergyPlusV8-9-0/Energy+.idd"))
EPW_PATH = Path(_require("epw_path", "weather file for the measured year"))
INPUT_IDF_PATH = Path(_require("idf_path", "model to calibrate"))
PREPARED_IDF_PATH = Path("./Calibration/Prepared_Cali_RFH.idf")
OUTPUT_IDF_PATH = Path("./Calibration/After_Cali_RFH.idf")
STAGE1_RESULT_JSON = Path("./Calibration/stage1_result.json")

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
# What the user told us about the building
# ============================================================
# Extracted from the plain-language request, for example
#     "my residential building (masonry, built 1969)"
# Fallbacks widen the prior rather than narrowing it.
STRUCTURE = str(_RUNTIME.get("structure", "masonry")).strip().lower()
CONSTRUCTION_YEAR = int(_RUNTIME.get("construction_year", 0) or 0)
WINDOWS_REPLACED = _RUNTIME.get("windows_replaced")   # True, False or None


# ============================================================
# Prior library
# ============================================================
# One entry per structure class and construction era. Values are assembly
# U-values in W/(m2K), and air changes per hour, expressed as the range an
# experienced modeller would consider credible for that class of building
# before seeing any measurement.
#
# Korean insulation requirements for residential buildings were introduced
# in 1979 and tightened repeatedly afterwards, so the era boundaries follow
# the regulatory history rather than round decades.
#
# For the present case building, a 1969 masonry dwelling, independent
# measurements give wall 1.81, roof 1.605, floor 1.705, window 2.82 and
# 0.357 ACH. They are not inputs. They are quoted here only so a reader can
# confirm that the prior brackets them.

PRIOR_LIBRARY: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]] = {
    # Uninsulated construction, before the 1979 requirement
    ("masonry", "pre1980"): {
        "u_wall":    (1.40, 2.10),
        "u_roof":    (1.20, 2.10),
        "u_floor":   (1.20, 2.10),
        "infil_ach": (0.25, 0.60),
    },
    ("concrete", "pre1980"): {
        "u_wall":    (1.30, 2.00),
        "u_roof":    (1.10, 2.00),
        "u_floor":   (1.10, 2.00),
        "infil_ach": (0.25, 0.60),
    },
    # First generations of insulation requirements
    ("masonry", "1980to2000"): {
        "u_wall":    (0.55, 1.20),
        "u_roof":    (0.40, 1.00),
        "u_floor":   (0.50, 1.20),
        "infil_ach": (0.20, 0.50),
    },
    ("concrete", "1980to2000"): {
        "u_wall":    (0.45, 1.00),
        "u_roof":    (0.30, 0.80),
        "u_floor":   (0.45, 1.00),
        "infil_ach": (0.20, 0.50),
    },
    ("concrete", "post2000"): {
        "u_wall":    (0.20, 0.55),
        "u_roof":    (0.15, 0.40),
        "u_floor":   (0.20, 0.55),
        "infil_ach": (0.10, 0.35),
    },
}

# Glazing depends on whether the original windows survive, which a user can
# usually answer and which moves the range by roughly a factor of two.
WINDOW_PRIOR: Dict[Tuple[str, Optional[bool]], Tuple[float, float]] = {
    ("pre1980",    True):  (2.20, 3.60),   # replaced, double glazing
    ("pre1980",    False): (4.50, 6.00),   # original single glazing
    ("pre1980",    None):  (2.20, 6.00),   # unknown, prior stays wide
    ("1980to2000", True):  (1.80, 3.20),
    ("1980to2000", False): (2.80, 4.50),
    ("1980to2000", None):  (1.80, 4.50),
    ("post2000",   True):  (1.10, 2.40),
    ("post2000",   False): (1.40, 3.00),
    ("post2000",   None):  (1.10, 3.00),
}

# Seasonal efficiency of a domestic gas boiler. The nameplate figure is
# measured under favourable conditions; seasonal performance in service is
# normally a little lower.
BOILER_EFF_PRIOR = (0.75, 0.88)


def _era(year: int) -> str:
    if year and year < 1980:
        return "pre1980"
    if year and year < 2000:
        return "1980to2000"
    if year:
        return "post2000"
    return "pre1980"          # widest, least favourable assumption


def _structure_class(s: str) -> str:
    if any(t in s for t in ("masonry", "brick", "조적", "벽돌")):
        return "masonry"
    if any(t in s for t in ("concrete", "rc", "철근", "콘크리트")):
        return "concrete"
    return "masonry"


def select_prior_bounds() -> Dict[str, Tuple[float, float]]:
    """Choose the bound for each parameter from what the user stated."""
    era = _era(CONSTRUCTION_YEAR)
    cls = _structure_class(STRUCTURE)

    key = (cls, era)
    if key not in PRIOR_LIBRARY:
        key = ("masonry", era) if ("masonry", era) in PRIOR_LIBRARY else ("masonry", "pre1980")
        print(f"[WARN] No prior for {cls}/{era}; falling back to {key}.")

    b = dict(PRIOR_LIBRARY[key])
    b["u_window"] = WINDOW_PRIOR[(era, WINDOWS_REPLACED)]
    b["boiler_eff"] = BOILER_EFF_PRIOR

    print(f"[INFO] Prior selected for structure='{cls}', era='{era}', "
          f"windows_replaced={WINDOWS_REPLACED}")
    return b


BOUNDS = select_prior_bounds()
VAR_ORDER = ["u_wall", "u_roof", "u_floor", "u_window", "infil_ach", "boiler_eff"]


# ============================================================
# Constructions
# ============================================================
# Real layers keep handbook properties and thermal mass. One massless layer
# per assembly carries the uncertain resistance and is solved from the
# target U, so the decision variable stays a U-value while the model keeps a
# physically meaningful layer structure.
#
# For a pre-1980 masonry dwelling, from outside inwards:
#     wall   fired brick 200 mm, variable layer, cement plaster 15 mm
#     roof   protective mortar 30 mm, variable layer, RC slab 150 mm,
#            ceiling plaster 12.5 mm
#     floor  variable layer on the ground side, RC slab 250 mm, pipe plane,
#            cement mortar 40 mm, sheet flooring 4 mm
#
# The entire mortar layer sits above the pipe plane, as specified for the
# case building. The two polyethylene films in that build-up are omitted:
# at 0.05 mm their combined resistance is below 0.001 m2K/W, and layers that
# thin degrade the conduction transfer function calculation. They are vapour
# barriers, not thermal layers.

# name -> (roughness, thickness_m, conductivity, density, specific_heat)
MATERIAL_SPECS: Dict[str, tuple] = {
    "Ext_Brick_200":    ("Rough",       0.200, 0.770, 1700.0,  840.0),
    "Int_Plaster_15":   ("Smooth",      0.015, 0.600, 1600.0, 1000.0),
    "RC_Slab_250":      ("MediumRough", 0.250, 1.600, 2300.0,  880.0),
    "RC_Slab_150":      ("MediumRough", 0.150, 1.600, 2300.0,  880.0),
    "Cement_Mortar_40": ("Smooth",      0.040, 1.400, 2000.0, 1000.0),
    "Cement_Mortar_30": ("Smooth",      0.030, 1.400, 2000.0, 1000.0),
    "Floor_Finish_4":   ("Smooth",      0.004, 0.190, 1200.0, 1200.0),
    "Ceiling_Gypsum":   ("Smooth",      0.0125, 0.180, 800.0, 1090.0),
}

VARIABLE_LAYERS = ["Wall_Var_R", "Roof_Var_R", "Floor_Var_R"]
R_VAR_MIN = 0.001            # EnergyPlus lower limit for Material:NoMass

# Surface film resistances, ISO 6946
R_SI_WALL, R_SE = 0.13, 0.04
R_SI_ROOF = 0.10
R_SI_FLOOR = 0.17            # downward heat flow

WALL_CONSTRUCTION = "Exterior_Wall_Construction"
WALL_LAYERS = ["Ext_Brick_200", "Wall_Var_R", "Int_Plaster_15"]
WALL_FILMS = R_SI_WALL + R_SE

ROOF_CONSTRUCTION = "Project Flat Roof"
ROOF_LAYERS = ["Cement_Mortar_30", "Roof_Var_R", "RC_Slab_150", "Ceiling_Gypsum"]
ROOF_FILMS = R_SI_ROOF + R_SE

RFH_FLOOR_CONSTRUCTION = "Slab Floor with Radiant"   # name kept: surfaces reference it
RFH_FLOOR_LAYERS = ["Floor_Var_R", "RC_Slab_250",
                    "Cement_Mortar_40", "Floor_Finish_4"]
RFH_SOURCE_AFTER_LAYER = 2       # pipe plane on top of the slab
RFH_TEMPCALC_AFTER_LAYER = 3     # top of the mortar
RFH_TUBE_SPACING_M = 0.25
RFH_TUBE_INSIDE_DIAMETER_M = 0.012      # 16 mm OD, 2 mm wall
FLOOR_FILMS = R_SI_FLOOR         # ground on the other side, no external film

PLAIN_FLOOR_CONSTRUCTION = "Project Floor"
PLAIN_FLOOR_LAYERS = ["Floor_Var_R", "RC_Slab_250",
                      "Cement_Mortar_40", "Floor_Finish_4"]

WINDOW_SIMPLE_GLAZING_NAME = "SG_2p0"


def _layer_resistance(name: str) -> float:
    if name in VARIABLE_LAYERS:
        return 0.0
    _, thickness, k, _, _ = MATERIAL_SPECS[name]
    return thickness / k


def fixed_resistance(layers: List[str], films: float) -> float:
    """Resistance of everything except the variable layer."""
    return films + sum(_layer_resistance(n) for n in layers)


def r_var_for_target_u(target_u: float, layers: List[str], films: float,
                       label: str) -> float:
    """Resistance the variable layer must carry to reach the target U."""
    r_fixed = fixed_resistance(layers, films)
    r_var = 1.0 / float(target_u) - r_fixed
    if r_var < R_VAR_MIN:
        print(f"[WARN] {label}: U={target_u:.3f} exceeds the maximum "
              f"{1.0 / r_fixed:.3f} achievable with these layers; clipped.")
        r_var = R_VAR_MIN
    return r_var


def report_construction_limits() -> None:
    print("[INFO] Assembly limits with the fixed layers only:")
    for label, layers, films, key in [
        ("wall ", WALL_LAYERS, WALL_FILMS, "u_wall"),
        ("roof ", ROOF_LAYERS, ROOF_FILMS, "u_roof"),
        ("floor", RFH_FLOOR_LAYERS, FLOOR_FILMS, "u_floor"),
    ]:
        rf = fixed_resistance(layers, films)
        lo, hi = BOUNDS[key]
        ok = "ok" if hi <= 1.0 / rf else "PRIOR UPPER BOUND IS UNREACHABLE"
        print(f"  {label}: R_fixed={rf:.3f}  U_max={1.0 / rf:.2f}  "
              f"prior=({lo:.2f}, {hi:.2f})  {ok}")


# ============================================================
# Utilities
# ============================================================
def reset_calibration_workspace(work_dir: Path) -> None:
    if work_dir.exists():
        print(f"[INFO] Clearing previous calibration workspace: {work_dir.resolve()}")
        shutil.rmtree(work_dir, ignore_errors=True)
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


def ensure_nomass(idf: IDF, name: str, resistance: float) -> None:
    _del_by_name(idf, "MATERIAL:NOMASS", name)
    m = idf.newidfobject("MATERIAL:NOMASS")
    m.Name = name
    m.Roughness = "Smooth"
    m.Thermal_Resistance = max(float(resistance), R_VAR_MIN)
    m.Thermal_Absorptance = 0.9
    m.Solar_Absorptance = 0.7
    m.Visible_Absorptance = 0.7


def _set_layers(obj, layers: List[str]) -> None:
    obj.Outside_Layer = layers[0]
    for i, lyr in enumerate(layers[1:], start=2):
        setattr(obj, f"Layer_{i}", lyr)


def ensure_construction(idf: IDF, name: str, layers: List[str]) -> None:
    _del_by_name(idf, "CONSTRUCTION", name)
    c = idf.newidfobject("CONSTRUCTION")
    c.Name = name
    _set_layers(c, layers)


def ensure_internal_source_construction(idf: IDF, name: str, layers: List[str],
                                        source_after: int, tempcalc_after: int,
                                        tube_spacing: float) -> None:
    _del_by_name(idf, "CONSTRUCTION:INTERNALSOURCE", name)
    c = idf.newidfobject("CONSTRUCTION:INTERNALSOURCE")
    c.Name = name
    c.Source_Present_After_Layer_Number = source_after
    c.Temperature_Calculation_Requested_After_Layer_Number = tempcalc_after
    # One-dimensional conduction transfer functions. Two dimensions would be
    # more accurate at 250 mm tube spacing but cost an order of magnitude
    # more runtime across several hundred evaluations.
    c.Dimensions_for_the_CTF_Calculation = 1
    c.Tube_Spacing = tube_spacing
    _set_layers(c, layers)


def rebuild_envelope_constructions(idf: IDF) -> None:
    """Replace the generated placeholders with build-ups appropriate to the
    stated structure and era. Construction names are preserved so existing
    BuildingSurface:Detailed references stay valid."""
    for name in MATERIAL_SPECS:
        ensure_material(idf, name)

    # Start each variable layer at the midpoint of its prior
    for key, name, layers, films in [
        ("u_wall", "Wall_Var_R", WALL_LAYERS, WALL_FILMS),
        ("u_roof", "Roof_Var_R", ROOF_LAYERS, ROOF_FILMS),
        ("u_floor", "Floor_Var_R", RFH_FLOOR_LAYERS, FLOOR_FILMS),
    ]:
        lo, hi = BOUNDS[key]
        ensure_nomass(idf, name, r_var_for_target_u(0.5 * (lo + hi), layers, films, name))

    ensure_construction(idf, WALL_CONSTRUCTION, WALL_LAYERS)
    ensure_construction(idf, ROOF_CONSTRUCTION, ROOF_LAYERS)
    ensure_construction(idf, PLAIN_FLOOR_CONSTRUCTION, PLAIN_FLOOR_LAYERS)
    ensure_internal_source_construction(
        idf, RFH_FLOOR_CONSTRUCTION, RFH_FLOOR_LAYERS,
        RFH_SOURCE_AFTER_LAYER, RFH_TEMPCALC_AFTER_LAYER, RFH_TUBE_SPACING_M)

    for r in idf.idfobjects.get("ZONEHVAC:LOWTEMPERATURERADIANT:VARIABLEFLOW", []):
        set_field(r, ["Hydronic_Tubing_Inside_Diameter"],
                  RFH_TUBE_INSIDE_DIAMETER_M, required=False)

    print("[INFO] Rebuilt constructions:")
    print(f"  {WALL_CONSTRUCTION}: " + " | ".join(WALL_LAYERS))
    print(f"  {ROOF_CONSTRUCTION}: " + " | ".join(ROOF_LAYERS))
    print(f"  {PLAIN_FLOOR_CONSTRUCTION}: " + " | ".join(PLAIN_FLOOR_LAYERS))
    print(f"  {RFH_FLOOR_CONSTRUCTION}: " + " | ".join(RFH_FLOOR_LAYERS)
          + f"  (source after layer {RFH_SOURCE_AFTER_LAYER}, "
            f"tube spacing {RFH_TUBE_SPACING_M} m)")


def prepare_idf(idd_path: Path, input_idf_path: Path, prepared_path: Path) -> Path:
    IDF.setiddname(str(idd_path))
    idf = IDF(str(input_idf_path))
    rebuild_envelope_constructions(idf)
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    idf.saveas(str(prepared_path))
    print(f"[OK] Prepared IDF written to: {prepared_path.resolve()}")
    return prepared_path


#%% ============================================================
# Meter parsing
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


def read_monthly_meter_j_from_meter_csv(meter_csv: Path) -> Dict[str, Dict[int, float]]:
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

    meters_monthly: Dict[str, Dict[int, float]] = {n: {} for n in set(monthly_idx_to_name.values())}
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
            meters_monthly.setdefault(monthly_idx_to_name[idx], {})[current_month] = val_j

    meters_monthly = {k: v for k, v in meters_monthly.items() if v}
    if not meters_monthly:
        raise RuntimeError("Monthly meter indices found, but no monthly values parsed from data section.")
    return meters_monthly


def read_sim_monthly_heat_kwh(out_dir: Path) -> Dict[int, float]:
    """Delivered heat by month in kWh. This is heat at the emitter, not fuel."""
    meter_csvs = sorted(out_dir.glob("*Meter.csv"))
    if meter_csvs:
        meters = read_monthly_meter_j_from_meter_csv(meter_csvs[0])
    else:
        mtr = out_dir / "eplusout.mtr"
        if not mtr.exists():
            raise RuntimeError("No *Meter.csv and no eplusout.mtr found in output directory.")
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
    return {mm: (val_j / 3.6e6) for mm, val_j in monthly_j.items()}


def heat_to_fuel(heat_kwh: Dict[int, float], efficiency: float) -> Dict[int, float]:
    """Convert delivered heat into metered fuel.

    The measured record is fuel purchased; EnergyPlus reports heat delivered
    to the zones. Comparing the two directly understates the model by the
    boiler loss.
    """
    eta = max(float(efficiency), 1e-6)
    return {mm: v / eta for mm, v in heat_kwh.items()}


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


def align_months(measured_kwh: Dict[str, float],
                 simulated_kwh: Dict[int, float]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    meas, sim, kept = [], [], []
    for mm in MEASURED_MONTHS:
        if mm in simulated_kwh:
            meas.append(float(measured_kwh[str(mm)]))
            sim.append(float(simulated_kwh[mm]))
            kept.append(mm)
    if len(meas) < 2:
        raise RuntimeError(
            f"Not enough overlapping months. Measured={MEASURED_MONTHS}, "
            f"SimAvailable={sorted(simulated_kwh.keys())}")
    return np.array(meas), np.array(sim), kept


def compute_metrics(measured_kwh: Dict[str, float],
                    simulated_fuel_kwh: Dict[int, float]) -> Tuple[float, float, Dict[int, float]]:
    meas, sim, months = align_months(measured_kwh, simulated_fuel_kwh)
    return (cvrmse_percent(meas, sim), nmbe_percent(meas, sim),
            {m: float(simulated_fuel_kwh[m]) for m in months})


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
# IDF editing
# ============================================================
def _get_nomass(idf: IDF, name: str):
    for m in idf.idfobjects.get("MATERIAL:NOMASS", []):
        if getattr(m, "Name", "").strip() == name:
            return m
    raise KeyError(f"Material:NoMass not found: {name}")


def _get_simple_glazing(idf: IDF, name: str):
    for o in idf.idfobjects.get("WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM", []):
        if getattr(o, "Name", "").strip() == name:
            return o
    raise KeyError(f"WindowMaterial:SimpleGlazingSystem not found: {name}")


def apply_envelope_params(idf: IDF, u_wall: float, u_roof: float,
                          u_floor: float, u_window: float) -> None:
    """Set assembly U-values by solving each variable layer's resistance."""
    _get_nomass(idf, "Wall_Var_R").Thermal_Resistance = r_var_for_target_u(
        u_wall, WALL_LAYERS, WALL_FILMS, "wall")
    _get_nomass(idf, "Roof_Var_R").Thermal_Resistance = r_var_for_target_u(
        u_roof, ROOF_LAYERS, ROOF_FILMS, "roof")
    _get_nomass(idf, "Floor_Var_R").Thermal_Resistance = r_var_for_target_u(
        u_floor, RFH_FLOOR_LAYERS, FLOOR_FILMS, "floor")
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
# Problem
# ============================================================
class EnvelopeInfilCalibrationProblem(ElementwiseProblem):
    def __init__(self, idd_path: Path, input_idf_path: Path, epw_path: Path,
                 work_dir: Path, measured_monthly_kwh: Dict[str, float],
                 energyplus_exe: Path, log_csv_path: Path):

        xl = np.array([BOUNDS[v][0] for v in VAR_ORDER], dtype=float)
        xu = np.array([BOUNDS[v][1] for v in VAR_ORDER], dtype=float)
        super().__init__(n_var=len(VAR_ORDER), n_obj=2, xl=xl, xu=xu)

        self.idd_path = idd_path
        self.input_idf_path = input_idf_path
        self.epw_path = epw_path
        self.work_dir = work_dir
        self.measured = measured_monthly_kwh
        self.energyplus_exe = energyplus_exe
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
        apply_envelope_params(idf, vals["u_wall"], vals["u_roof"],
                              vals["u_floor"], vals["u_window"])
        apply_global_infiltration_ach(idf, vals["infil_ach"])
        idf.saveas(str(run_idf))

        t0 = time.time()
        failed = False
        try:
            run_energyplus(self.energyplus_exe, run_idf, self.epw_path, out_dir)
            sim_heat = read_sim_monthly_heat_kwh(out_dir)
            sim_fuel = heat_to_fuel(sim_heat, vals["boiler_eff"])
            cvr, nb, _ = compute_metrics(self.measured, sim_fuel)
            absnb = abs(nb)
        except Exception:
            failed = True
            cvr, nb, absnb = 1e6, 1e6, 1e6
        runtime = time.time() - t0

        # Objectives are the metrics themselves. Every reachable point lies
        # inside a prior derived from the stated structure and era, so no
        # plausibility penalty is needed and the reported metrics are the
        # optimized ones.
        out["F"] = np.array([cvr, absnb], dtype=float)

        with self.log_csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [eval_id] + [vals[v] for v in VAR_ORDER] + [cvr, nb, absnb, runtime])

        if CLEANUP_EACH_RUN:
            if failed and KEEP_FAILED_RUNS:
                return
            shutil.rmtree(run_dir, ignore_errors=True)


# ============================================================
# Reporting and selection
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
    sim_heat = read_sim_monthly_heat_kwh(out_dir)
    eta = 0.5 * (BOUNDS["boiler_eff"][0] + BOUNDS["boiler_eff"][1])
    sim_fuel = heat_to_fuel(sim_heat, eta)
    cvr, nb, sim_used = compute_metrics(MEASURED_MONTHLY_KWH, sim_fuel)

    print(f"\n=== Baseline Monthly Comparison (kWh fuel, boiler eff {eta:.2f}) ===")
    print("Month | Measured | Simulated | Residual (Sim-Meas)")
    for m in MEASURED_MONTHS:
        meas = MEASURED_MONTHLY_KWH[str(m)]
        sim = sim_used.get(m, float("nan"))
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
    return {v: float(best[v]) for v in VAR_ORDER}, float(best["CVRMSE_%"]), float(best["NMBE_%"])


def write_final_idf_with_best_params(idd_path: Path, input_idf_path: Path,
                                     output_idf_path: Path,
                                     best_params: Dict[str, float]) -> None:
    IDF.setiddname(str(idd_path))
    idf = IDF(str(input_idf_path))
    apply_envelope_params(idf, best_params["u_wall"], best_params["u_roof"],
                          best_params["u_floor"], best_params["u_window"])
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

    print("[INFO] Stage          : 1 - envelope, infiltration, boiler efficiency")
    print(f"[INFO] EnergyPlus exe : {energyplus_exe}")
    print(f"[INFO] Input model    : {INPUT_IDF_PATH}")
    print(f"[INFO] Weather file   : {EPW_PATH.name}")
    print(f"[INFO] Building       : {STRUCTURE}, built {CONSTRUCTION_YEAR or 'unknown'}")
    print(f"[INFO] Measured months: {MEASURED_MONTHS}  "
          f"(total {sum(MEASURED_MONTHLY_KWH.values()):,.0f} kWh fuel)")
    print("[INFO] Prior bounds:")
    for v in VAR_ORDER:
        lo, hi = BOUNDS[v]
        print(f"  - {v:11s} [{lo:6.3f}, {hi:6.3f}]")
    report_construction_limits()

    prepared = prepare_idf(IDD_PATH, INPUT_IDF_PATH, PREPARED_IDF_PATH)
    baseline_report(IDD_PATH, prepared, EPW_PATH, energyplus_exe)

    log_csv = WORK_DIR / "eval_log.csv"
    problem = EnvelopeInfilCalibrationProblem(
        idd_path=IDD_PATH,
        input_idf_path=prepared,
        epw_path=EPW_PATH,
        work_dir=WORK_DIR,
        measured_monthly_kwh=MEASURED_MONTHLY_KWH,
        energyplus_exe=energyplus_exe,
        log_csv_path=log_csv,
    )

    print("\n[INFO] Starting NSGA-II optimization (envelope, infiltration, boiler)...")
    minimize(problem, NSGA2(pop_size=POP_SIZE), get_termination("n_gen", N_GEN),
             seed=SEED, save_history=False, verbose=True)

    best_params, best_cvr, best_nb = select_best_from_log(log_csv)
    print("\n=== Selected Best (lexicographic: CVRMSE then |NMBE|) ===")
    for k, v in best_params.items():
        lo, hi = BOUNDS[k]
        pos = 100.0 * (v - lo) / (hi - lo)
        flag = "  <-- at bound" if pos <= 2.0 or pos >= 98.0 else ""
        print(f"  {k:11s} = {v:8.4f}   [{lo:6.3f}, {hi:6.3f}]   {pos:5.1f}% of range{flag}")
    print(f"CVRMSE = {best_cvr:.3f} %")
    print(f"NMBE   = {best_nb:.3f} %")
    print(f"[INFO] Full evaluation log saved to: {log_csv.resolve()}")

    write_final_idf_with_best_params(IDD_PATH, prepared, OUTPUT_IDF_PATH, best_params)
    print(f"[OK] Calibrated IDF saved to: {OUTPUT_IDF_PATH.resolve()}")

    # ── Verification report and chart ────────────────────────────────────
    try:
        from cali_verification_report import run_verification
        _vdir = WORK_DIR / "best_verify"
        _vdir.mkdir(exist_ok=True)
        run_energyplus(energyplus_exe, OUTPUT_IDF_PATH, EPW_PATH, _vdir / "out")
        _heat = read_sim_monthly_heat_kwh(_vdir / "out")
        run_verification(
            stage_name="Stage 1 – Envelope calibration",
            idf_path=str(OUTPUT_IDF_PATH),
            epw_path=str(EPW_PATH),
            measured=MEASURED_MONTHLY_KWH,
            simulated_heat=_heat,
            boiler_eff=best_params["boiler_eff"],
            chart_out=OUTPUT_IDF_PATH.parent / "stage1_comparison.png")
    except Exception as _e:
        print(f"[WARN] Verification report failed: {_e}")
    # ─────────────────────────────────────────────────────────────────────

    # Stages 2 to 4 also compare against metered fuel, so they need the
    # boiler efficiency selected here.
    STAGE1_RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    STAGE1_RESULT_JSON.write_text(json.dumps({
        "params": best_params,
        "bounds": {k: list(v) for k, v in BOUNDS.items()},
        "CVRMSE": best_cvr,
        "NMBE": best_nb,
        "structure": STRUCTURE,
        "construction_year": CONSTRUCTION_YEAR,
    }, indent=2), encoding="utf-8")
    print(f"[OK] Stage 1 result written to: {STAGE1_RESULT_JSON.resolve()}")


if __name__ == "__main__":
    main()
