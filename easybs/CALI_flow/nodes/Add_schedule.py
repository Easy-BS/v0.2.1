# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 13:08:21 2026

@author: Xiguan Liang @SKKU
"""
# ./CALI_flow/nodes/Add_schedule.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import re

from eppy.modeleditor import IDF


from cali_runtime_config import load_runtime_config
_RUNTIME = load_runtime_config()

BUILDING_TYPE = _RUNTIME.get("building_type", "residential")
IDD_PATH = Path(_RUNTIME.get("idd_path", r"C:/EnergyPlusV8-9-0/Energy+.idd"))
INPUT_IDF_PATH = Path(_RUNTIME.get("idf_path", "./Calibration/Before_All_Cali_RFH.idf"))
OUTPUT_IDF_PATH = Path("./Calibration/Ready1_Cali_RFH.idf")


# ---------------------------
# Schedule Profiles
# ---------------------------
# Format: List[Tuple[str, float]] meaning Until: HH:MM, value
# Must end at ("24:00", value)

def profiles_residential_evening_home() -> Dict[str, List[Tuple[str, float]]]:
    """
    Residential: away during work hours; home evenings/night.
    - Occupancy, lights, equipment: low daytime, high evening/night.
    - Heating setpoint: setback daytime; comfort evening/night.
    - Radiant heating setpoint: same pattern (can offset if desired).
    """
    occ = [
        ("06:00", 0.95),  # sleeping at home
        ("08:00", 0.80),  # leave for work
        ("18:00", 0.15),  # return home
        ("23:30", 0.95),  # late evening
        ("24:00", 0.95),
    ]

    light = [
        ("06:00", 0.05),
        ("08:00", 0.85),
        ("18:00", 0.10),
        ("23:30", 0.95),
        ("24:00", 0.85),
    ]

    equip = [
        ("06:00", 0.10),
        ("08:00", 0.55),
        ("18:00", 0.15),
        ("23:30", 0.65),
        ("24:00", 0.60),
    ]

    # Heating setpoint schedule (°C)
    # Daytime setback, comfort in evening/night
    heating_sp = [
        ("06:00", 22.0),  # morning comfort
        ("08:00", 20.0),  # away (setback)
        ("18:00", 16.0),  # evening comfort
        ("24:00", 22.0),
    ]

    # Radiant heating setpoint schedule (°C)
    # Often similar to heating setpoint. can set slightly higher if needed for slab response.
    radiant_sp = [
        ("06:00", 22.0),
        ("08:00", 20.0),
        ("18:00", 16.0),
        ("24:00", 22.0),
    ]

    return {
        "Occ_Sch_Res_Quick": occ,
        "Light_Sch_Res_Quick": light,
        "Equip_Sch_Res_Quick": equip,
        "HEATING SETPOINTS": heating_sp,
        "RADIANT HEATING SETPOINTS": radiant_sp,
    }


def profiles_public_daytime() -> Dict[str, List[Tuple[str, float]]]:
    """
    Public building: occupied during daytime (e.g., office/school), mostly unoccupied at night.
    - Occupancy, lights, equipment: high daytime, low night.
    - Heating: comfort in daytime, setback at night.
    """
    occ = [
        ("06:00", 0.05),
        ("09:00", 0.15),
        ("18:00", 0.95),
        ("22:00", 0.35),
        ("24:00", 0.15),
    ]

    light = [
        ("06:00", 0.05),
        ("09:00", 0.35),
        ("18:00", 0.90),
        ("22:00", 0.55),
        ("24:00", 0.05),
    ]

    equip = [
        ("06:00", 0.05),
        ("09:00", 0.35),
        ("18:00", 0.90),
        ("22:00", 0.55),
        ("24:00", 0.05),
    ]

    heating_sp = [
        ("06:00", 15.0),  # night setback until morning warmup
        ("09:00", 17.0),  # occupied comfort
        ("18:00", 21.0),  # after-hours mild
        ("22:00", 17.0),  # night setback
        ("24:00", 15.0),
    ]

    radiant_sp = [
        ("06:00", 15.0),
        ("09:00", 17.0),
        ("18:00", 21.0),
        ("22:00", 17.0),
        ("24:00", 15.0),
    ]

    return {
        "Occ_Sch_Res_Quick": occ,
        "Light_Sch_Res_Quick": light,
        "Equip_Sch_Res_Quick": equip,
        "HEATING SETPOINTS": heating_sp,
        "RADIANT HEATING SETPOINTS": radiant_sp,
    }


# ---------------------------
# eppy utilities
# ---------------------------
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


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


def find_schedule_compact(idf: IDF, sched_name: str):
    for s in idf.idfobjects.get("SCHEDULE:COMPACT", []):
        if getattr(s, "Name", "").strip().upper() == sched_name.upper():
            return s
    return None


def delete_schedule_compact(idf: IDF, sched_name: str) -> int:
    """
    Delete all Schedule:Compact objects with matching name.
    Returns how many were removed.
    """
    seq = idf.idfobjects.get("SCHEDULE:COMPACT", [])
    to_del = [s for s in seq if getattr(s, "Name", "").strip().upper() == sched_name.upper()]
    for s in to_del:
        seq.remove(s)
    return len(to_del)


def create_schedule_compact(idf: IDF, name: str, type_limits: str, day_profile: List[Tuple[str, float]]):
    """
    Create Schedule:Compact with:
      Through: 12/31
      For: AllDays
      Until: ...
    """
    sc = idf.newidfobject("SCHEDULE:COMPACT")
    set_field(sc, ["Name"], name)
    set_field(sc, ["Schedule_Type_Limits_Name", "ScheduleTypeLimitsName"], type_limits, required=False)

    # Extensible fields Field_1..Field_n
    sc.Field_1 = "Through: 12/31"
    sc.Field_2 = "For: AllDays"

    idx = 3
    for t, v in day_profile:
        setattr(sc, f"Field_{idx}", f"Until: {t}")
        setattr(sc, f"Field_{idx+1}", float(v))
        idx += 2
    return sc


def ensure_schedule_type_limits(idf: IDF, name: str, lower, upper, numeric_type: str) -> None:
    for stl in idf.idfobjects.get("SCHEDULETYPELIMITS", []):
        if getattr(stl, "Name", "").strip().upper() == name.upper():
            return
    stl = idf.newidfobject("SCHEDULETYPELIMITS")
    set_field(stl, ["Name"], name)
    if lower is not None:
        set_field(stl, ["Lower_Limit_Value", "LowerLimitValue"], float(lower), required=False)
    if upper is not None:
        set_field(stl, ["Upper_Limit_Value", "UpperLimitValue"], float(upper), required=False)
    set_field(stl, ["Numeric_Type", "NumericType"], numeric_type, required=False)


def replace_schedule_compact(idf: IDF, name: str, type_limits: str, day_profile: List[Tuple[str, float]]):
    """
    Direct replacement: delete existing schedule(s) with that name, then recreate.
    """
    delete_schedule_compact(idf, name)
    return create_schedule_compact(idf, name, type_limits, day_profile)


# ---------------------------
# Main
# ---------------------------
def build_profiles(building_type: str) -> Dict[str, List[Tuple[str, float]]]:
    bt = building_type.strip().lower()
    if bt == "residential":
        return profiles_residential_evening_home()
    if bt == "public":
        return profiles_public_daytime()
    raise ValueError(f"Unsupported BUILDING_TYPE: {building_type}. Use 'residential' or 'public'.")


def main() -> None:
    if not IDD_PATH.exists():
        raise FileNotFoundError(f"IDD not found: {IDD_PATH.resolve()}")
    if not INPUT_IDF_PATH.exists():
        raise FileNotFoundError(f"Input IDF not found: {INPUT_IDF_PATH.resolve()}")

    IDF.setiddname(str(IDD_PATH))
    idf = IDF(str(INPUT_IDF_PATH))

    profiles = build_profiles(BUILDING_TYPE)

    # Ensure common schedule type limits exist
    ensure_schedule_type_limits(idf, "Fraction", 0.0, 1.0, "Continuous")
    ensure_schedule_type_limits(idf, "Any Number", None, None, "Continuous")

    # Apply replacements
    # - Fraction schedules: Occ/Light/Equip
    replace_schedule_compact(idf, "Occ_Sch_Res_Quick", "Fraction", profiles["Occ_Sch_Res_Quick"])
    replace_schedule_compact(idf, "Light_Sch_Res_Quick", "Fraction", profiles["Light_Sch_Res_Quick"])
    replace_schedule_compact(idf, "Equip_Sch_Res_Quick", "Fraction", profiles["Equip_Sch_Res_Quick"])

    # - Setpoint schedules (temperature): Any Number
    replace_schedule_compact(idf, "HEATING SETPOINTS", "Any Number", profiles["HEATING SETPOINTS"])
    replace_schedule_compact(idf, "RADIANT HEATING SETPOINTS", "Any Number", profiles["RADIANT HEATING SETPOINTS"])

    OUTPUT_IDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    idf.saveas(str(OUTPUT_IDF_PATH))
    print(f"[OK] Saved updated schedules to: {OUTPUT_IDF_PATH.resolve()}")
    print(f"Applied BUILDING_TYPE = {BUILDING_TYPE}")


if __name__ == "__main__":
    main()


