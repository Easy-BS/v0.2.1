# -*- coding: utf-8 -*-
# ./CALI_flow/nodes/amy_helpers.py
#
# Two small helpers for amy_weather.py:
#
#   find_cached_amy()          reuse an AMY EPW generated on an earlier run,
#                              before any network access happens
#   set_runperiod_start_day()  align the IDF RunPeriod weekday with the
#                              actual calendar year of the measured data
#
# Why the weekday matters: with a typical-year file the day-of-week alignment
# is arbitrary, but with year-specific weather the simulated calendar should
# match the real one, or weekday and weekend schedules fall on the wrong days
# for the entire run. 1 January 2024 was a Monday; the IDF written by
# apply_rfh() declares Tuesday.
#
# Setting the field explicitly is preferable to leaving it blank. A blank
# field makes EnergyPlus take the weekday from the EPW header, and diyepw
# copies that header from the TMY template, so it reflects the typical year
# rather than the actual one.

from __future__ import annotations

import datetime
import os
import re
import sys
from pathlib import Path
from typing import Optional

WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]


def log(msg: str):
    print(f"[amy-helpers] {msg}", file=sys.stderr)


# --------------------------------------------------------------------------
# Reuse a previously generated AMY file
# --------------------------------------------------------------------------

def find_cached_amy(outdir: str | Path, wmo: int, year: int) -> Optional[str]:
    """Return a previously generated AMY EPW for this station and year.

    diyepw names its output <country>_<state>_<city>.<station>_AMY_<year>.epw,
    so the station number and year are enough to identify it. Checking here
    avoids downloading the ISD files only to have diyepw discover the file
    already exists.
    """
    outdir = Path(outdir)
    if not outdir.is_dir():
        return None

    pattern = re.compile(rf"\.{int(wmo)}_AMY_{int(year)}\.epw$", re.I)
    matches = [p for p in outdir.glob("*.epw") if pattern.search(p.name)]
    if not matches:
        # Fall back to a looser match in case the station number is padded
        matches = [p for p in outdir.glob(f"*_AMY_{int(year)}.epw")
                   if str(int(wmo)) in p.name]
    if not matches:
        return None

    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    chosen = matches[0]
    if chosen.stat().st_size == 0:
        return None
    log(f"reusing cached AMY file {chosen.name}")
    return str(chosen.resolve())


# --------------------------------------------------------------------------
# Align the RunPeriod weekday with the calendar year
# --------------------------------------------------------------------------

def weekday_for_year(year: int, month: int = 1, day: int = 1) -> str:
    """EnergyPlus weekday name for a given date."""
    return WEEKDAY_NAMES[datetime.date(int(year), month, day).weekday()]


def set_runperiod_start_day(idf_path: str | Path, year: int,
                            idd_path: Optional[str] = None) -> dict:
    """Set Day_of_Week_for_Start_Day on every RunPeriod to match the year.

    Returns a small report. The IDF is modified in place; a timestamped
    backup is written alongside it the first time.
    """
    idf_path = Path(idf_path)
    report = {"idf": idf_path.name, "year": int(year)}

    if not idf_path.is_file():
        report["status"] = "skipped"
        report["reason"] = f"IDF not found: {idf_path}"
        return report

    try:
        from eppy.modeleditor import IDF
    except ImportError as e:
        report["status"] = "skipped"
        report["reason"] = f"eppy not available: {e}"
        return report

    target_day = weekday_for_year(year)
    report["weekday"] = target_day

    try:
        if idd_path:
            IDF.setiddname(str(idd_path))
        idf = IDF(str(idf_path))
    except Exception as e:
        # setiddname raises if called twice with different paths; that is
        # harmless when the IDD is already set to the same file.
        try:
            idf = IDF(str(idf_path))
        except Exception:
            report["status"] = "failed"
            report["error"] = str(e)
            return report

    runperiods = idf.idfobjects.get("RUNPERIOD", [])
    if not runperiods:
        report["status"] = "skipped"
        report["reason"] = "no RunPeriod object in the IDF"
        return report

    before = []
    changed = 0
    for rp in runperiods:
        current = (getattr(rp, "Day_of_Week_for_Start_Day", "") or "").strip()
        before.append(current or "(blank)")
        if current.lower() != target_day.lower():
            rp.Day_of_Week_for_Start_Day = target_day
            changed += 1

    report["previous"] = before

    if changed == 0:
        report["status"] = "unchanged"
        return report

    backup = idf_path.with_suffix(".pre_dow.idf")
    if not backup.exists():
        try:
            backup.write_bytes(idf_path.read_bytes())
            report["backup"] = backup.name
        except Exception:
            pass

    try:
        idf.saveas(str(idf_path))
    except Exception as e:
        report["status"] = "failed"
        report["error"] = str(e)
        return report

    report["status"] = "updated"
    report["runperiods_changed"] = changed
    log(f"RunPeriod start day set to {target_day} for {year} "
        f"(was {', '.join(before)})")
    return report
