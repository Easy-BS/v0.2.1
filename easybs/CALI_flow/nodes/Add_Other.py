# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 15:37:22 2026

@author: Xiguan Liang @SKKU
"""

# ./CALI_flow/nodes/Add_Other.py

from __future__ import annotations

from pathlib import Path
from typing import List
import re

from eppy.modeleditor import IDF


from cali_runtime_config import load_runtime_config
_RUNTIME = load_runtime_config()

IDD_PATH = Path(_RUNTIME.get("idd_path", r"C:/EnergyPlusV8-9-0/Energy+.idd"))
EPW_PATH = Path(_RUNTIME.get("epw_path", r"C:\EnergyPlusV8-9-0\WeatherData\KOR_INCH'ON_IWEC.epw"))
INPUT_IDF_PATH = Path(_RUNTIME.get("idf_path", "./Calibration/Ready1_Cali_RFH.idf"))
OUTPUT_IDF_PATH = Path("./Calibration/Ready2_Cali_RFH.idf")


# ---------------------------
# Helpers
# ---------------------------
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _read_epw_monthly_mean_drybulb(epw_path: Path) -> List[float]:
    """
    EPW columns (typical):
      1 Year, 2 Month, 3 Day, 4 Hour, 5 Minute, 6 Flags, 7 DryBulb(°C), ...
    DryBulb is usually column 7 => index 6 (0-based).
    """
    sums = [0.0] * 12
    cnts = [0] * 12

    with epw_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Skip header (first 8 lines in standard EPW)
    for line in lines[8:]:
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 7:
            continue
        try:
            month = int(parts[1])
            tdb = float(parts[6])
        except Exception:
            continue
        if 1 <= month <= 12:
            sums[month - 1] += tdb
            cnts[month - 1] += 1

    if any(c == 0 for c in cnts):
        raise RuntimeError(f"EPW parsing failed: some months have zero records. cnts={cnts}")

    return [sums[i] / cnts[i] for i in range(12)]


def generate_groundtemps_buildingsurface(monthly_air: List[float], damp: float = 0.30, lag_months: int = 2) -> List[float]:
    """
    Simple climate-adaptive ground temperature profile:
    - Start from annual mean air temperature
    - Add a damped seasonal anomaly (deep ground has smaller swing)
    - Apply a lag (deep ground peaks later than air)

    ground[m] = Tmean + damp * (air[(m - lag) mod 12] - Tmean)

    Typical:
      damp ~ 0.2–0.4
      lag  ~ 1–2 months
    """
    if len(monthly_air) != 12:
        raise ValueError("monthly_air must have length 12.")

    tmean = sum(monthly_air) / 12.0
    anomalies = [t - tmean for t in monthly_air]

    ground = []
    for m in range(12):
        src = (m - lag_months) % 12
        ground.append(tmean + damp * anomalies[src])

    return ground


def upsert_site_groundtemperature_buildingsurface(idf: IDF, ground_monthly: List[float]) -> None:
    """
    Insert if missing, else overwrite Site:GroundTemperature:BuildingSurface.
    Fields are 12 monthly temperatures (Jan..Dec).
    """
    key = "SITE:GROUNDTEMPERATURE:BUILDINGSURFACE"
    objs = idf.idfobjects.get(key, [])

    if objs:
        obj = objs[0]
    else:
        obj = idf.newidfobject(key)

    # Robust field mapping: try to match common IDD month fieldnames
    # e.g., "January_Ground_Temperature", "February_Ground_Temperature", ...
    month_names = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]

    fmap = {_norm(fn): fn for fn in obj.fieldnames if fn and fn.lower() != "key"}

    for i, mn in enumerate(month_names):
        candidates = [
            f"{mn}_Ground_Temperature",
            f"{mn}_GroundTemperature",
            mn,  # last-resort (some IDDs use just month names)
        ]
        target_field = None
        for cand in candidates:
            k = _norm(cand)
            if k in fmap:
                target_field = fmap[k]
                break

        if target_field is None:
            # Fallback: assign by position using fieldnames order (after key).
            # In most IDDs, the first 12 non-key fields are Jan..Dec.
            nonkey_fields = [fn for fn in obj.fieldnames if fn and fn.lower() != "key"]
            if len(nonkey_fields) >= 12:
                target_field = nonkey_fields[i]
            else:
                raise RuntimeError(f"Cannot locate monthly fields in {key}. fieldnames={obj.fieldnames}")

        #setattr(obj, target_field, float(ground_monthly[i]))
        val = max(15.0, float(ground_monthly[i]))
        setattr(obj, target_field, val)


def main() -> None:
    if not IDD_PATH.exists():
        raise FileNotFoundError(f"IDD not found: {IDD_PATH.resolve()}")
    if not EPW_PATH.exists():
        raise FileNotFoundError(f"EPW not found: {EPW_PATH.resolve()}")
    if not INPUT_IDF_PATH.exists():
        raise FileNotFoundError(f"Input IDF not found: {INPUT_IDF_PATH.resolve()}")

    # 1) Parse EPW -> monthly mean dry-bulb
    monthly_air = _read_epw_monthly_mean_drybulb(EPW_PATH)

    # 2) Generate ground temps (damped + lagged)
    ground = generate_groundtemps_buildingsurface(monthly_air, damp=0.30, lag_months=2)

    # 3) Insert/replace in IDF
    IDF.setiddname(str(IDD_PATH))
    idf = IDF(str(INPUT_IDF_PATH))

    upsert_site_groundtemperature_buildingsurface(idf, ground)

    OUTPUT_IDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    idf.saveas(str(OUTPUT_IDF_PATH))

    # quick console summary
    print("[OK] Inserted/updated Site:GroundTemperature:BuildingSurface")
    print("Monthly mean air (°C):   ", [round(x, 2) for x in monthly_air])
    print("Derived ground temps (°C):", [round(x, 2) for x in ground])
    print("Saved:", OUTPUT_IDF_PATH.resolve())


if __name__ == "__main__":
    main()
