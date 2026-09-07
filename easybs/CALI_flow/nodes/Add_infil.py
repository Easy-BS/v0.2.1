# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 09:05:10 2026

@author: Xiguan Liang @SKKU
"""
# ./CALI_flow/nodes/Add_infil.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import re

from eppy.modeleditor import IDF


# ---------------------------
# User inputs (edit as needed)
# ---------------------------
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

BUILDING_TYPE = _RUNTIME.get("building_type", "residential")

IDD_PATH = Path(_RUNTIME.get("idd_path", r"C:/EnergyPlusV8-9-0/Energy+.idd"))

INPUT_IDF_PATH = Path(_RUNTIME.get("idf_path", "./generated_idfs/geom_multizone_modified_RFH.idf"))
OUTPUT_IDF_PATH = Path("./Calibration/Before_Cali_RFH.idf")

DEFAULT_INFILTRATION_ACH = float(_RUNTIME.get("default_infiltration_ach", 0.50))

print("[DEBUG] MEASURED_MONTHLY_KWH =", MEASURED_MONTHLY_KWH)
print("[DEBUG] months used for RunPeriod =", sorted(int(m) for m in MEASURED_MONTHLY_KWH.keys()))

# ---------------------------
# Robust eppy field assignment helpers
# ---------------------------
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def set_field(obj, candidates: List[str], value, *, required: bool = True) -> str:
    """
    Set obj.<field> using the first matching field name in candidates.
    Matching is done by normalized field name (case/underscore-insensitive).

    Returns the actual field name used.
    """
    fieldnames = getattr(obj, "fieldnames", [])
    # fieldnames typically include "key" at index 0
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


def set_first_n_fields(obj, values: List, *, start_index_after_key: int = 1) -> None:
    """
    Fallback setter when IDD field names are unexpected:
    assign by the first N real fields in obj.fieldnames (excluding 'key').

    NOTE: eppy does NOT support numeric indexing (m[0]) for fields.
    We assign by *field name strings* obtained from obj.fieldnames.
    """
    fieldnames = [fn for fn in getattr(obj, "fieldnames", []) if fn and fn.lower() != "key"]
    need = start_index_after_key - 1 + len(values)
    if len(fieldnames) < need:
        raise ValueError(
            f"Not enough fields to assign {len(values)} values for '{obj.key}'. "
            f"Available fields: {fieldnames}"
        )
    start = start_index_after_key - 1  # convert to 0-based in filtered list
    for i, v in enumerate(values):
        setattr(obj, fieldnames[start + i], v)


# ---------------------------
# General helpers
# ---------------------------
def safe_name(s: str) -> str:
    s2 = re.sub(r"[^A-Za-z0-9_]+", "_", s).strip("_")
    return s2 if s2 else "Zone"


def month_set_to_runperiods(months: List[int]) -> List[Tuple[int, int, int, int, str]]:
    mset = sorted(set(months))
    if not mset:
        raise ValueError("No months provided.")

    blocks: List[List[int]] = []
    cur = [mset[0]]
    for m in mset[1:]:
        if m == cur[-1] + 1:
            cur.append(m)
        else:
            blocks.append(cur)
            cur = [m]
    blocks.append(cur)

    if len(blocks) > 2:
        raise ValueError(
            f"Quick mode supports at most 2 month blocks; got {len(blocks)} blocks: {blocks}. "
            "Consider simulating the full year and ignoring unmeasured months."
        )

    def end_day(mm: int) -> int:
        return {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}[mm]

    out = []
    for i, b in enumerate(blocks, start=1):
        bm, em = b[0], b[-1]
        out.append((bm, 1, em, end_day(em), f"Cali_RunPeriod_{i}"))
    return out


def _clear_idf_objects(idf: IDF, key_upper: str) -> int:
    """
    Safely delete all objects under a key in eppy (Idf_MSequence-safe).
    """
    key_upper = key_upper.upper()
    if key_upper not in idf.idfobjects:
        return 0
    seq = idf.idfobjects[key_upper]
    n = len(seq)
    for _ in range(n):
        seq.pop()
    return n


# ---------------------------
# IDF edits
# ---------------------------
def remove_all_outputs(idf: IDF) -> None:
    """
    Remove all OUTPUT:* and OUTPUTCONTROL:* objects.
    """
    for k in list(idf.idfobjects.keys()):
        ku = k.upper()
        if ku.startswith("OUTPUT:") or ku.startswith("OUTPUTCONTROL:"):
            _clear_idf_objects(idf, ku)


def remove_all_runperiods(idf: IDF) -> None:
    _clear_idf_objects(idf, "RUNPERIOD")


def ensure_schedule_on(idf: IDF) -> None:
    """
    Ensure Schedule:Compact named 'ON' exists (always 1.0).
    Uses extensible fields Field_1..Field_4.
    """
    for s in idf.idfobjects.get("SCHEDULE:COMPACT", []):
        if getattr(s, "Name", "").strip().upper() == "ON":
            return

    sc = idf.newidfobject("SCHEDULE:COMPACT")
    # These fields are stable in E+8.9 IDD
    set_field(sc, ["Name"], "ON")
    set_field(sc, ["Schedule_Type_Limits_Name", "ScheduleTypeLimitsName"], "Fraction")

    # Extensible pairs (string tokens) in Schedule:Compact are usually Field_1..Field_n
    # Keep it minimal and robust.
    setattr(sc, "Field_1", "Through: 12/31")
    setattr(sc, "Field_2", "For: AllDays")
    setattr(sc, "Field_3", "Until: 24:00")
    setattr(sc, "Field_4", 1.0)


def add_minimal_monthly_meters(idf: IDF) -> None:
    """
    Add ONLY monthly meters:
      - Electricity:Heating
      - Gas:Heating
      - DistrictHeating:Facility

    Field names for OUTPUT:METER vary by IDD; we set via robust detection and fallback.
    """
    for meter_name in ("DistrictHeating:Facility",):
    #for meter_name in ("Electricity:Heating", "Gas:Heating", "DistrictHeating:Facility"): 
        m = idf.newidfobject("OUTPUT:METER")

        # Preferred candidates across IDDs
        try:
            set_field(m, ["Key_Name", "Meter_Name", "Name"], meter_name, required=False)
            set_field(m, ["Reporting_Frequency", "ReportingFrequency"], "Monthly", required=False)

            # If either didn't get set (because fieldnames are unexpected), fallback to first two fields.
            # Detect unset by checking whether any candidate matched.
            # (If required=False and no match, set_field returns "")
            # We'll simply verify both are present in the object after setting.
            # If not, fallback by order.
            # NOTE: Accessing by getattr with unknown field name is unreliable, so we use fallback immediately if needed.
            fieldnames = [fn for fn in m.fieldnames if fn and fn.lower() != "key"]
            if len(fieldnames) >= 2:
                # Check if the first two fields are still blank-ish
                v0 = getattr(m, fieldnames[0], "")
                v1 = getattr(m, fieldnames[1], "")
                if (v0 is None or str(v0).strip() == "") and (v1 is None or str(v1).strip() == ""):
                    set_first_n_fields(m, [meter_name, "Monthly"])
            else:
                # Very unusual IDD; force by order helper (will raise clearly)
                set_first_n_fields(m, [meter_name, "Monthly"])
        except Exception:
            # Hard fallback: assign first two real fields (excluding key) as (meter_name, Monthly)
            set_first_n_fields(m, [meter_name, "Monthly"])


def add_runperiod(idf: IDF, bm: int, bd: int, em: int, ed: int, name: str) -> None:
    rp = idf.newidfobject("RUNPERIOD")

    # Try canonical E+8.9 names; fallback to ordering if something is off
    try:
        set_field(rp, ["Name"], name)
        set_field(rp, ["Begin_Month", "BeginMonth"], bm)
        set_field(rp, ["Begin_Day_of_Month", "BeginDayOfMonth"], bd)
        set_field(rp, ["End_Month", "EndMonth"], em)
        set_field(rp, ["End_Day_of_Month", "EndDayOfMonth"], ed)

        set_field(rp, ["Day_of_Week_for_Start_Day", "DayOfWeekForStartDay"], "UseWeatherFile")
        set_field(rp, ["Use_Weather_File_Holidays_and_Special_Days", "UseWeatherFileHolidaysandSpecialDays"], "Yes")
        set_field(rp, ["Use_Weather_File_Daylight_Saving_Period", "UseWeatherFileDaylightSavingPeriod"], "Yes")
        set_field(rp, ["Apply_Weekend_Holiday_Rule", "ApplyWeekendHolidayRule"], "No")
        set_field(rp, ["Use_Weather_File_Rain_Indicators", "UseWeatherFileRainIndicators"], "Yes")
        set_field(rp, ["Use_Weather_File_Snow_Indicators", "UseWeatherFileSnowIndicators"], "Yes")
    except Exception:
        # Fallback by the first 11 fields after key (common RunPeriod layout)
        set_first_n_fields(
            rp,
            [
                name, bm, bd, em, ed,
                "UseWeatherFile",
                "Yes",
                "Yes",
                "No",
                "Yes",
                "Yes",
            ],
        )


def any_zone_infiltration(idf: IDF) -> bool:
    for k, seq in idf.idfobjects.items():
        if k.upper().startswith("ZONEINFILTRATION:") and len(seq) > 0:
            return True
    return False


def list_zone_names(idf: IDF) -> List[str]:
    return [z.Name for z in idf.idfobjects.get("ZONE", []) if getattr(z, "Name", "").strip()]


def add_infiltration_all_zones(idf: IDF, ach: float) -> None:
    zones = list_zone_names(idf)
    if not zones:
        raise RuntimeError("No Zone objects found; cannot add infiltration.")

    for zn in zones:
        inf = idf.newidfobject("ZONEINFILTRATION:DESIGNFLOWRATE")

        # name
        try:
            set_field(inf, ["Name"], f"{safe_name(zn)}_Infiltration")
        except Exception:
            # if name field is unexpected, first field after key is typically Name
            set_first_n_fields(inf, [f"{safe_name(zn)}_Infiltration"])

        # zone reference (fieldname varies)
        try:
            set_field(
                inf,
                [
                    "Zone_or_ZoneList_or_Space_or_SpaceList_Name",
                    "Zone_or_ZoneList_Name",
                    "Zone_Name",
                    "ZoneName",
                ],
                zn,
                required=False,
            )
            # If nothing matched, fall back to the second field after key (often the zone name)
            fieldnames = [fn for fn in inf.fieldnames if fn and fn.lower() != "key"]
            if len(fieldnames) >= 2 and (getattr(inf, fieldnames[1], "") is None or str(getattr(inf, fieldnames[1], "")).strip() == ""):
                setattr(inf, fieldnames[1], zn)
        except Exception:
            # fallback: set first two fields after key (Name already set), then zone
            fieldnames = [fn for fn in inf.fieldnames if fn and fn.lower() != "key"]
            if len(fieldnames) >= 2:
                setattr(inf, fieldnames[1], zn)
            else:
                raise

        # schedule / method / ach / coefficients
        # Use candidate fields; if some aren't found, fallback by order at the end.
        try:
            set_field(inf, ["Schedule_Name", "ScheduleName"], "ON", required=False)
            set_field(
                inf,
                ["Design_Flow_Rate_Calculation_Method", "DesignFlowRateCalculationMethod"],
                "AirChanges/Hour",
                required=False,
            )
            set_field(inf, ["Air_Changes_per_Hour", "AirChangesperHour"], float(ach), required=False)

            set_field(inf, ["Constant_Term_Coefficient", "ConstantTermCoefficient"], 1.0, required=False)
            set_field(inf, ["Temperature_Term_Coefficient", "TemperatureTermCoefficient"], 0.0, required=False)
            set_field(inf, ["Velocity_Term_Coefficient", "VelocityTermCoefficient"], 0.0, required=False)
            set_field(inf, ["Velocity_Squared_Term_Coefficient", "VelocitySquaredTermCoefficient"], 0.0, required=False)

            # If key fields are still blank because IDD naming is odd, do a final positional fallback:
            # For DesignFlowRate, common order after key is:
            # Name, Zone..., Schedule, Method, DesignFlow(m3/s), Flow/Area, Flow/ExtArea, ACH, A, B, C, D
            fieldnames = [fn for fn in inf.fieldnames if fn and fn.lower() != "key"]
            # Only apply fallback if the "Schedule" and "Method" appear unset
            if len(fieldnames) >= 12:
                sched_val = getattr(inf, fieldnames[2], "")
                meth_val = getattr(inf, fieldnames[3], "")
                if (sched_val is None or str(sched_val).strip() == "") and (meth_val is None or str(meth_val).strip() == ""):
                    # keep Name + Zone already set, set the rest by order
                    setattr(inf, fieldnames[2], "ON")
                    setattr(inf, fieldnames[3], "AirChanges/Hour")
                    # fields[4..6] empty
                    setattr(inf, fieldnames[7], float(ach))
                    setattr(inf, fieldnames[8], 1.0)
                    setattr(inf, fieldnames[9], 0.0)
                    setattr(inf, fieldnames[10], 0.0)
                    setattr(inf, fieldnames[11], 0.0)
        except Exception:
            # As last resort, assign by known common order (after key):
            # Name, Zone, Schedule, Method, (blank x3), ACH, A, B, C, D
            # We won't touch Name/Zone again; we set fields from schedule onward if possible.
            fieldnames = [fn for fn in inf.fieldnames if fn and fn.lower() != "key"]
            if len(fieldnames) >= 12:
                setattr(inf, fieldnames[2], "ON")
                setattr(inf, fieldnames[3], "AirChanges/Hour")
                setattr(inf, fieldnames[7], float(ach))
                setattr(inf, fieldnames[8], 1.0)
                setattr(inf, fieldnames[9], 0.0)
                setattr(inf, fieldnames[10], 0.0)
                setattr(inf, fieldnames[11], 0.0)
            else:
                raise


def preprocess_idf_with_eppy(
    idd_path: Path,
    input_idf: Path,
    output_idf: Path,
    measured_monthly_kwh: Dict[str, float],
    default_infiltration_ach: float = 0.50,
) -> None:
    if not idd_path.exists():
        raise FileNotFoundError(f"IDD not found: {idd_path.resolve()}")
    if not input_idf.exists():
        raise FileNotFoundError(f"Input IDF not found: {input_idf.resolve()}")

    IDF.setiddname(str(idd_path))
    idf = IDF(str(input_idf))

    # Outputs: remove all; add minimal meters only
    remove_all_outputs(idf)
    add_minimal_monthly_meters(idf)

    # RunPeriod: remove and re-add 1–2 blocks based on measured months
    remove_all_runperiods(idf)
    months = [int(m) for m in measured_monthly_kwh.keys()]
    for bm, bd, em, ed, name in month_set_to_runperiods(months):
        add_runperiod(idf, bm, bd, em, ed, name)

    # Infiltration: add if missing (and ensure ON schedule exists)
    ensure_schedule_on(idf)
    if not any_zone_infiltration(idf):
        add_infiltration_all_zones(idf, default_infiltration_ach)

    # Save
    output_idf.parent.mkdir(parents=True, exist_ok=True)
    idf.saveas(str(output_idf))
    print(f"[OK] Wrote preprocessed IDF to: {output_idf.resolve()}")


if __name__ == "__main__":
    print("=== Preprocess IDF for Calibration (Quick Mode, eppy, E+8.9) ===")
    print(f"IDD       : {IDD_PATH.resolve()}")
    print(f"Input IDF : {INPUT_IDF_PATH.resolve()}")
    print(f"Output IDF: {OUTPUT_IDF_PATH.resolve()}")
    print(f"Measured months: {sorted(int(k) for k in MEASURED_MONTHLY_KWH.keys())}")

    preprocess_idf_with_eppy(
        idd_path=IDD_PATH,
        input_idf=INPUT_IDF_PATH,
        output_idf=OUTPUT_IDF_PATH,
        measured_monthly_kwh=MEASURED_MONTHLY_KWH,
        default_infiltration_ach=DEFAULT_INFILTRATION_ACH,
    )

