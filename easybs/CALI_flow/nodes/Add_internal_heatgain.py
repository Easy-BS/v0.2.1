# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 09:05:10 2026

@author: Xiguan Liang @SKKU
"""


# ./CALI_flow/nodes/Add_internal_heatgain.py

# Purpose (Step 1: Occupancy/Internal Gains):
# - Add uniform occupant + activity schedules and internal heat gains to ALL zones
# - Reserve interfaces for future per-room customization (helper modules)
#
# Output: ./Calibration/Before_Occ_Cali_RFH.idf

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import re

from eppy.modeleditor import IDF


from cali_runtime_config import load_runtime_config
_RUNTIME = load_runtime_config()

MEASURED_MONTHLY_KWH = _RUNTIME.get("measured_monthly_kwh", {...})
BUILDING_TYPE = _RUNTIME.get("building_type", "residential")
IDD_PATH = Path(_RUNTIME.get("idd_path", r"C:/EnergyPlusV8-9-0/Energy+.idd"))
INPUT_IDF_PATH = Path(_RUNTIME.get("idf_path", "./Calibration/Before_Cali_RFH.idf"))
OUTPUT_IDF_PATH = Path("./Calibration/Before_All_Cali_RFH.idf")


# ---------------------------
# QUICK MODE DEFAULTS (uniform for all zones)
# ---------------------------
# Occupancy model:
# - Use People per floor area (people/m2) so it scales with zone area.
# - If later want per-room schedules, use the reserved interfaces below.
PEOPLE_PER_FLOOR_AREA: float = 0.03        # ~1 person per  m2 (tune later)
FRACTION_RADIANT_PEOPLE: float = 0.30      # NREL-like
CO2_GEN_RATE_M3_PER_S_W: float = 3.82e-8   # NREL-like (ok as placeholder)

# Activity / comfort:
ACTIVITY_W_PER_PERSON: float = 120.0       # seated/light activity
WORK_EFFICIENCY: float = 0.0               # typical (all metabolic becomes heat)
CLOTHING_CLO: float = 0.9                  # winter-ish default; can be seasonal later
AIR_VELOCITY_M_PER_S: float = 0.10         # comfort only

# Internal gains:
LIGHTING_W_PER_M2: float = 5.0             # residential placeholder
EQUIPMENT_W_PER_M2: float = 7.0            # residential placeholder

# Schedule shapes (very simple, uniform)
# These are FRACTION schedules (0..1) except activity/clothing/air velocity/work eff (Any Number).
# can refine later (weekday/weekend, seasonal, room-based).
OCC_DAY = [
    ("00:00", 0.90),
    ("07:00", 0.50),
    ("09:00", 0.15),
    ("18:00", 0.80),
    ("23:00", 0.95),
    ("24:00", 0.95),
]
LIGHT_DAY = [
    ("00:00", 0.15),
    ("06:00", 0.30),
    ("08:00", 0.20),
    ("18:00", 0.70),
    ("23:00", 0.25),
    ("24:00", 0.15),
]
EQUIP_DAY = [
    ("00:00", 0.35),
    ("07:00", 0.45),
    ("09:00", 0.30),
    ("18:00", 0.60),
    ("23:00", 0.40),
    ("24:00", 0.35),
]


# ---------------------------
# RESERVED INTERFACES (for future flexible per-room customization)
# ---------------------------
# In the future, can move these into helper modules, e.g.:
#   from helpers.schedules import build_residential_schedules
#   from helpers.internal_gains import apply_zone_overrides
#
# For now, quick mode applies the same schedules and gains to all zones.

def get_zone_profile(building_type: str, zone_name: str) -> Dict[str, float]:
    """
    Reserved interface:
    Return per-zone overrides (people density, LPD, EPD, etc.) based on building type and room type.

    Current quick-mode behavior: uniform settings for all zones.
    """
    return {
        "people_per_m2": PEOPLE_PER_FLOOR_AREA,
        "lighting_w_per_m2": LIGHTING_W_PER_M2,
        "equip_w_per_m2": EQUIPMENT_W_PER_M2,
    }


def get_schedule_profile(building_type: str, zone_name: str) -> Dict[str, List]:
    """
    Reserved interface:
    Return per-zone schedule shapes (occupancy, lights, equipment, activity).

    Current quick-mode behavior: uniform schedules for all zones.
    """
    return {
        "occ_day": OCC_DAY,
        "light_day": LIGHT_DAY,
        "equip_day": EQUIP_DAY,
    }


# ---------------------------
# Robust eppy field assignment utilities
# ---------------------------
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def set_field(obj, candidates: List[str], value, *, required: bool = True) -> str:
    """
    Set obj.<field> using the first matching field name in candidates.
    Matching is case/underscore-insensitive.

    Returns the actual field name used.
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


def ensure_schedule_type_limits(idf: IDF, name: str, lower: Optional[float], upper: Optional[float], numeric_type: str) -> None:
    """
    Ensure ScheduleTypeLimits exists. Many files already have these, but LLM-generated IDFs often don't.
    - For Fraction schedules: lower=0, upper=1, numeric_type='Continuous'
    - For Any Number: lower/upper blank, numeric_type='Continuous'
    """
    for stl in idf.idfobjects.get("SCHEDULETYPELIMITS", []):
        if getattr(stl, "Name", "").strip().upper() == name.upper():
            return

    stl = idf.newidfobject("SCHEDULETYPELIMITS")
    set_field(stl, ["Name"], name)

    # Some IDDs use different field names, so keep this flexible.
    if lower is not None:
        set_field(stl, ["Lower_Limit_Value", "LowerLimitValue"], float(lower), required=False)
    if upper is not None:
        set_field(stl, ["Upper_Limit_Value", "UpperLimitValue"], float(upper), required=False)

    set_field(stl, ["Numeric_Type", "NumericType"], numeric_type, required=False)


def ensure_schedule_compact(idf: IDF, name: str, type_limits: str, day_profile: List) -> None:
    """
    Create (or overwrite) a Schedule:Compact with a single AllDays profile.
    day_profile example: [("00:00",0.9),("07:00",0.5),...,("24:00",0.95)]
    """
    # Delete existing same-name schedule to avoid duplicates/ambiguity
    existing = [s for s in idf.idfobjects.get("SCHEDULE:COMPACT", []) if getattr(s, "Name", "").strip().upper() == name.upper()]
    for s in existing:
        idf.idfobjects["SCHEDULE:COMPACT"].remove(s)

    sc = idf.newidfobject("SCHEDULE:COMPACT")
    set_field(sc, ["Name"], name)
    set_field(sc, ["Schedule_Type_Limits_Name", "ScheduleTypeLimitsName"], type_limits, required=False)

    # Extensible fields Field_1..Field_n
    # Format:
    #   Through: 12/31
    #   For: AllDays
    #   Until: HH:MM, value
    setattr(sc, "Field_1", "Through: 12/31")
    setattr(sc, "Field_2", "For: AllDays")

    # Build Until blocks
    idx = 3
    for t, v in day_profile:
        setattr(sc, f"Field_{idx}", f"Until: {t}")
        setattr(sc, f"Field_{idx+1}", float(v))
        idx += 2


def list_zone_names(idf: IDF) -> List[str]:
    return [z.Name for z in idf.idfobjects.get("ZONE", []) if getattr(z, "Name", "").strip()]


def has_any_object(idf: IDF, key: str) -> bool:
    return len(idf.idfobjects.get(key.upper(), [])) > 0


def remove_objects_by_key(idf: IDF, key: str) -> None:
    if key.upper() in idf.idfobjects:
        idf.idfobjects[key.upper()][:] = []  # NOTE: this slice is safe for some keys, but can break for others


# ---------------------------
# Main: Apply People/Lights/Equipment to ALL zones uniformly
# ---------------------------
def add_people_lights_equipment_uniform(idf: IDF, building_type: str) -> None:
    zones = list_zone_names(idf)
    if not zones:
        raise RuntimeError("No Zone objects found; cannot add People/Lights/Equipment.")

    # 1) Ensure schedule type limits
    ensure_schedule_type_limits(idf, name="Fraction", lower=0.0, upper=1.0, numeric_type="Continuous")
    ensure_schedule_type_limits(idf, name="Any Number", lower=None, upper=None, numeric_type="Continuous")

    # 2) Create schedules (uniform for all zones)
    # Fraction schedules
    ensure_schedule_compact(idf, "Occ_Sch_Res_Quick", "Fraction", OCC_DAY)
    ensure_schedule_compact(idf, "Light_Sch_Res_Quick", "Fraction", LIGHT_DAY)
    ensure_schedule_compact(idf, "Equip_Sch_Res_Quick", "Fraction", EQUIP_DAY)

    # Any Number schedules (constant values)
    ensure_schedule_compact(idf, "Activity_Sch_Res_Quick", "Any Number", [("24:00", ACTIVITY_W_PER_PERSON)])
    ensure_schedule_compact(idf, "WorkEff_Sch_Res_Quick", "Any Number", [("24:00", WORK_EFFICIENCY)])
    ensure_schedule_compact(idf, "Clothing_Sch_Res_Quick", "Any Number", [("24:00", CLOTHING_CLO)])
    ensure_schedule_compact(idf, "AirVelo_Sch_Res_Quick", "Any Number", [("24:00", AIR_VELOCITY_M_PER_S)])

    # 3) Add People/Lights/ElectricEquipment per zone
    # If the target IDF "lacked occupancy parameters", we add them.
    # If IDF later contains these objects, can switch to overwrite/merge logic.

    for zn in zones:
        prof = get_zone_profile(building_type, zn)

        # --- People ---
        p = idf.newidfobject("PEOPLE")
        # Common E+ fields (8.9). Use robust mapping by candidates.
        set_field(p, ["Name"], f"{safe_name(zn)}_People")
        set_field(p, ["Zone_or_ZoneList_or_Space_or_SpaceList_Name", "Zone_or_ZoneList_Name", "Zone_Name"], zn, required=False)

        set_field(p, ["Number_of_People_Schedule_Name", "NumberofPeopleScheduleName"], "Occ_Sch_Res_Quick", required=False)

        # Calculation method: People / People/Area / Area/Person
        set_field(p, ["Number_of_People_Calculation_Method", "NumberofPeopleCalculationMethod"], "People/Area", required=False)
        # Use People per Floor Area
        set_field(p, ["People_per_Zone_Floor_Area","People_per_Floor_Area", "PeopleperZoneFloorArea","PeopleperFloorArea"], float(prof["people_per_m2"]), required=False)

        set_field(p, ["Fraction_Radiant", "FractionRadiant"], float(FRACTION_RADIANT_PEOPLE), required=False)
        # Sensible Heat Fraction left blank -> E+ default autosplit (acceptable for quick mode)

        set_field(p, ["Activity_Level_Schedule_Name", "ActivityLevelScheduleName"], "Activity_Sch_Res_Quick", required=False)
        set_field(p, ["Carbon_Dioxide_Generation_Rate", "CarbonDioxideGenerationRate"], float(CO2_GEN_RATE_M3_PER_S_W), required=False)

        # Comfort-related fields (optional but matches NREL style)
        set_field(p, ["Mean_Radiant_Temperature_Calculation_Type", "MeanRadiantTemperatureCalculationType"], "ZoneAveraged", required=False)
        set_field(p, ["Work_Efficiency_Schedule_Name", "WorkEfficiencyScheduleName"], "WorkEff_Sch_Res_Quick", required=False)
        set_field(p, ["Clothing_Insulation_Calculation_Method", "ClothingInsulationCalculationMethod"], "ClothingInsulationSchedule", required=False)
        set_field(p, ["Clothing_Insulation_Schedule_Name", "ClothingInsulationScheduleName"], "Clothing_Sch_Res_Quick", required=False)
        set_field(p, ["Air_Velocity_Schedule_Name", "AirVelocityScheduleName"], "AirVelo_Sch_Res_Quick", required=False)
        set_field(p, ["Thermal_Comfort_Model_1_Type", "ThermalComfortModel1Type"], "FANGER", required=False)

        # --- Lights ---
        l = idf.newidfobject("LIGHTS")
        set_field(l, ["Name"], f"{safe_name(zn)}_Lights")
        set_field(l, ["Zone_or_ZoneList_or_Space_or_SpaceList_Name", "Zone_or_ZoneList_Name", "Zone_Name"], zn, required=False)
        set_field(l, ["Schedule_Name", "ScheduleName"], "Light_Sch_Res_Quick", required=False)

        # Lighting level method: use Watts/Area
        set_field(l, ["Design_Level_Calculation_Method", "DesignLevelCalculationMethod"], "Watts/Area", required=False)
        set_field(l, ["Watts_per_Zone_Floor_Area", "WattsperZoneFloorArea"], float(prof["lighting_w_per_m2"]), required=False)

        # Fractions (reasonable defaults)
        set_field(l, ["Fraction_Radiant", "FractionRadiant"], 0.6, required=False)
        set_field(l, ["Fraction_Visible", "FractionVisible"], 0.2, required=False)
        set_field(l, ["Fraction_Replaceable", "FractionReplaceable"], 1.0, required=False)

        # --- ElectricEquipment ---
        e = idf.newidfobject("ELECTRICEQUIPMENT")
        set_field(e, ["Name"], f"{safe_name(zn)}_Equip")
        set_field(e, ["Zone_or_ZoneList_or_Space_or_SpaceList_Name", "Zone_or_ZoneList_Name", "Zone_Name"], zn, required=False)
        set_field(e, ["Schedule_Name", "ScheduleName"], "Equip_Sch_Res_Quick", required=False)

        set_field(e, ["Design_Level_Calculation_Method", "DesignLevelCalculationMethod"], "Watts/Area", required=False)
        set_field(e, ["Watts_per_Zone_Floor_Area", "WattsperZoneFloorArea"], float(prof["equip_w_per_m2"]), required=False)

        # Fractions (typical)
        set_field(e, ["Fraction_Latent", "FractionLatent"], 0.0, required=False)
        set_field(e, ["Fraction_Radiant", "FractionRadiant"], 0.2, required=False)
        set_field(e, ["Fraction_Lost", "FractionLost"], 0.0, required=False)


def safe_name(s: str) -> str:
    s2 = re.sub(r"[^A-Za-z0-9_]+", "_", s).strip("_")
    return s2 if s2 else "Zone"


def main() -> None:
    if not IDD_PATH.exists():
        raise FileNotFoundError(f"IDD not found: {IDD_PATH.resolve()}")
    if not INPUT_IDF_PATH.exists():
        raise FileNotFoundError(f"Input IDF not found: {INPUT_IDF_PATH.resolve()}")

    IDF.setiddname(str(IDD_PATH))
    idf = IDF(str(INPUT_IDF_PATH))

    add_people_lights_equipment_uniform(idf, BUILDING_TYPE)

    OUTPUT_IDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    idf.saveas(str(OUTPUT_IDF_PATH))
    print(f"[OK] Saved with occupancy/internal gains: {OUTPUT_IDF_PATH.resolve()}")


if __name__ == "__main__":
    main()
