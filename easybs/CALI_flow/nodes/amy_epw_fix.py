# -*- coding: utf-8 -*-
# ./CALI_flow/nodes/amy_epw_fix.py
#
# Repair two defects in AMY EPW files produced by diyepw.
#
#   1. Atmospheric pressure. diyepw does not reliably carry the ISD-Lite
#      pressure column through, and writes denormal values such as 1e-311.
#      EnergyPlus rejects anything outside 31,000 to 120,000 Pa with a
#      Severe error. Bad values are replaced from the TMY template, matched
#      on month/day/hour, and any that remain are set from station
#      elevation using the barometric formula.
#
#   2. Leap-year flag. diyepw copies the HOLIDAYS/DAYLIGHT SAVINGS header
#      line from the TMY template, where the leap-year field reads "No". In
#      a leap year the file then contains 29 February data that EnergyPlus
#      silently discards, shortening the simulated year by one day.
#
# Optionally also sets the DATA PERIODS start day of week to the real
# weekday of 1 January for that year.
#
# EPW data columns, zero-indexed: 0 year, 1 month, 2 day, 3 hour, 4 minute,
# 5 data source flags, 6 dry bulb, 7 dew point, 8 relative humidity,
# 9 atmospheric station pressure.

from __future__ import annotations

import datetime
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

HEADER_LINES = 8
IDX_MONTH, IDX_DAY, IDX_HOUR, IDX_PRESSURE = 1, 2, 3, 9
P_MIN, P_MAX = 31000.0, 120000.0
EPW_MISSING_PRESSURE = 999999.0

WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]


def log(msg: str):
    print(f"[amy-epw-fix] {msg}", file=sys.stderr)


def _is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def pressure_from_elevation(elevation_m: float) -> float:
    """Standard atmosphere station pressure in Pa."""
    return 101325.0 * (1.0 - 2.25577e-5 * float(elevation_m)) ** 5.25588


def _read(path: Path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    return lines[:HEADER_LINES], [ln for ln in lines[HEADER_LINES:] if ln.strip()]


def _valid_pressure(tok: str) -> Optional[float]:
    try:
        v = float(tok)
    except Exception:
        return None
    if v == EPW_MISSING_PRESSURE:
        return None
    return v if P_MIN < v <= P_MAX else None


def fix_amy_epw(amy_path: str | Path,
                tmy_path: Optional[str | Path] = None,
                year: Optional[int] = None,
                set_start_day: bool = True,
                elevation_m: Optional[float] = None) -> Dict[str, Any]:
    """Repair an AMY EPW in place. The original is kept as <name>.raw.epw."""
    amy_path = Path(amy_path)
    report: Dict[str, Any] = {"file": amy_path.name}

    if not amy_path.is_file():
        report["status"] = "skipped"
        report["reason"] = f"not found: {amy_path}"
        return report

    header, rows = _read(amy_path)
    if len(header) < HEADER_LINES or not rows:
        report["status"] = "skipped"
        report["reason"] = "unexpected EPW structure"
        return report

    # Year and elevation from the file itself when not supplied
    first = rows[0].split(",")
    if year is None:
        try:
            year = int(float(first[0]))
        except Exception:
            year = None
    loc = header[0].split(",")
    if elevation_m is None and len(loc) >= 10:
        try:
            elevation_m = float(loc[9])
        except Exception:
            elevation_m = 0.0

    report["year"] = year
    report["rows"] = len(rows)

    # ---- 1. pressure -----------------------------------------------------
    donor: Dict[tuple, float] = {}
    if tmy_path and Path(tmy_path).is_file():
        try:
            _, trows = _read(Path(tmy_path))
            for ln in trows:
                p = ln.split(",")
                if len(p) <= IDX_PRESSURE:
                    continue
                v = _valid_pressure(p[IDX_PRESSURE])
                if v is not None:
                    donor[(p[IDX_MONTH].strip(), p[IDX_DAY].strip(),
                           p[IDX_HOUR].strip())] = v
        except Exception as e:
            log(f"could not read TMY pressure column ({e})")

    fallback = pressure_from_elevation(elevation_m or 0.0)
    fixed_from_tmy = fixed_from_elev = 0
    bad_before = 0

    out_rows = []
    for ln in rows:
        p = ln.split(",")
        if len(p) > IDX_PRESSURE:
            if _valid_pressure(p[IDX_PRESSURE]) is None:
                bad_before += 1
                key = (p[IDX_MONTH].strip(), p[IDX_DAY].strip(), p[IDX_HOUR].strip())
                if key in donor:
                    p[IDX_PRESSURE] = f"{donor[key]:.0f}"
                    fixed_from_tmy += 1
                else:
                    p[IDX_PRESSURE] = f"{fallback:.0f}"
                    fixed_from_elev += 1
        out_rows.append(",".join(p))

    report["pressure"] = {
        "invalid_before": bad_before,
        "repaired_from_tmy": fixed_from_tmy,
        "repaired_from_elevation": fixed_from_elev,
        "elevation_m": round(float(elevation_m or 0.0), 1),
        "fallback_pa": round(fallback, 0),
    }

    # ---- 2. leap-year flag ----------------------------------------------
    leap_fixed = False
    if year and _is_leap(int(year)):
        parts = header[4].split(",")
        if len(parts) >= 2 and parts[1].strip().lower() != "yes":
            parts[1] = "Yes"
            header[4] = ",".join(parts)
            leap_fixed = True
    report["leap_year_flag_set"] = leap_fixed

    # ---- 3. data-period start weekday -----------------------------------
    dow_set = None
    if set_start_day and year:
        parts = header[7].split(",")
        if len(parts) >= 5:
            want = WEEKDAY_NAMES[datetime.date(int(year), 1, 1).weekday()]
            if parts[4].strip().lower() != want.lower():
                dow_set = {"from": parts[4].strip(), "to": want}
                parts[4] = want
                header[7] = ",".join(parts)
    report["data_period_start_day"] = dow_set

    if not (bad_before or leap_fixed or dow_set):
        report["status"] = "unchanged"
        return report

    backup = amy_path.with_suffix(".raw.epw")
    if not backup.exists():
        shutil.copy2(amy_path, backup)
        report["backup"] = backup.name

    with open(amy_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(header + out_rows) + "\n")

    report["status"] = "repaired"
    log(f"{amy_path.name}: pressure {bad_before} invalid "
        f"({fixed_from_tmy} from TMY, {fixed_from_elev} from elevation)"
        + (", leap flag set" if leap_fixed else "")
        + (f", start day -> {dow_set['to']}" if dow_set else ""))
    return report
