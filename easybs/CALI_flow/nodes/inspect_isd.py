# -*- coding: utf-8 -*-
"""
inspect_isd_run.py

Same diagnostic as inspect_isd.py, but with the path set below so it can be
run directly from Spyder, IPython or an editor with no command line and no
SystemExit.

Edit ISD_PATH, then press Run.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd

# ==========================================================================
# EDIT THIS. Either the folder holding the .gz files, or one file.
# ==========================================================================
ISD_PATH = r"D:\xiguan_liang\RFH_E+\LLM_EP_demo\CALI_flow\weather\amy\_isd_cache"

# Months relevant to a heating-season calibration
HEATING_MONTHS = {1, 2, 3, 4, 10, 11, 12}
# ==========================================================================

COLS = ["year", "month", "day", "hour", "air_temp", "dew_point", "pressure",
        "wind_dir", "wind_speed", "sky_cover", "precip_1h", "precip_6h"]


def read_isd(path: Path) -> pd.DataFrame:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        df = pd.read_csv(f, sep=r"\s+", header=None, names=COLS)
    df = df.replace(-9999, pd.NA)
    df["ts"] = pd.to_datetime(
        dict(year=df.year, month=df.month, day=df.day, hour=df.hour),
        errors="coerce")
    return df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)


def gaps(df: pd.DataFrame, column: str = "air_temp") -> pd.DataFrame:
    """Reindex to a complete hourly series and list the runs of missing values."""
    year = int(df["year"].iloc[0])
    full = pd.date_range(f"{year}-01-01 00:00", f"{year}-12-31 23:00", freq="h")
    s = df.set_index("ts")[column].reindex(full)

    missing = s.isna()
    if not missing.any():
        return pd.DataFrame(columns=["start", "end", "hours", "months"])

    grp = (missing != missing.shift()).cumsum()[missing]
    out = []
    for _, idx in s.index.to_series()[missing].groupby(grp):
        out.append({
            "start": idx.iloc[0],
            "end": idx.iloc[-1],
            "hours": len(idx),
            "months": sorted({d.month for d in idx}),
        })
    return pd.DataFrame(out).sort_values("hours", ascending=False)


def report(path: Path) -> dict:
    df = read_isd(path)
    year = int(df["year"].iloc[0])
    leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    total_hours = 8784 if leap else 8760

    print(f"\n=== {path.name} ===")
    print(f"rows present : {len(df):,} of {total_hours:,}")

    g = gaps(df, "air_temp")
    summary = {"file": path.name, "year": year, "missing_hours": 0,
               "largest_gap": 0, "heating_missing_hours": 0,
               "heating_largest_gap": 0}

    if g.empty:
        print("air temperature: no missing hours")
        return summary

    heat = g[g["months"].apply(lambda m: bool(set(m) & HEATING_MONTHS))]
    summary.update({
        "missing_hours": int(g["hours"].sum()),
        "largest_gap": int(g["hours"].max()),
        "heating_missing_hours": int(heat["hours"].sum()) if not heat.empty else 0,
        "heating_largest_gap": int(heat["hours"].max()) if not heat.empty else 0,
    })

    print(f"air temperature missing hours : {summary['missing_hours']:,} "
          f"({100 * summary['missing_hours'] / total_hours:.1f}%)")
    print(f"largest gap : {summary['largest_gap']} hours")

    print("\n  gaps of 24 hours or more:")
    big = g[g["hours"] >= 24]
    if big.empty:
        print("    none")
    else:
        for _, r in big.iterrows():
            flag = "  <-- HEATING SEASON" if set(r["months"]) & HEATING_MONTHS else ""
            print(f"    {r['start']:%Y-%m-%d %H:%M} to {r['end']:%Y-%m-%d %H:%M}  "
                  f"{int(r['hours']):4d} h{flag}")

    print(f"\n  missing hours in heating months : {summary['heating_missing_hours']:,}")
    print(f"  largest gap in heating months   : {summary['heating_largest_gap']} hours")
    return summary


def run(path_str: str = ISD_PATH):
    target = Path(path_str)
    if not target.exists():
        print(f"Path not found: {target}")
        print("Locate the files with:  dir /s /b <project root>\\*471080*.gz")
        return []

    files = sorted(target.glob("*.gz")) if target.is_dir() else [target]
    if not files:
        print(f"No .gz files in {target}")
        return []

    results = []
    for f in files:
        try:
            results.append(report(f))
        except Exception as e:
            print(f"\n=== {f.name} ===\n  could not read: {e}")

    if results:
        print("\n--- summary ---")
        print(pd.DataFrame(results).to_string(index=False))
    return results


results = run()