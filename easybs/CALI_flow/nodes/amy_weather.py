# -*- coding: utf-8 -*-
"""
Created on Mon Aug 09 15:04:45 2026

@author:     Xiguan Liang @SKKU
"""

# ./CALI_flow/nodes/amy_weather.py
from __future__ import annotations

import importlib
import json
import math
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import pandas as pd
import requests

from state_schema import SimulationState

try:
    from .amy_helpers import find_cached_amy, set_runperiod_start_day
except ImportError:
    from amy_helpers import find_cached_amy, set_runperiod_start_day
    
try:
    from .amy_epw_fix import fix_amy_epw
except ImportError:
    from amy_epw_fix import fix_amy_epw
    
try:
    from haversine import haversine as _haversine
except Exception:
    _haversine = None

try:
    import diyepw
    from diyepw import analyze_noaa_isd_lite_file
    # diyepw/__init__.py re-exports the function create_amy_epw_file under the
    # same name as its module, shadowing it. Reach the module explicitly.
    _CAEF = importlib.import_module("diyepw.create_amy_epw_file")
    _HAS_DIYEPW = True
except Exception:
    diyepw = None
    _CAEF = None
    _HAS_DIYEPW = False


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

USER_AGENT = "EasyBS-amy-weather/1.0"
DOWNLOAD_TIMEOUT = 60
DEFAULT_CITY = "Seoul, South Korea"

# ISD quality gate. These are reproducibility parameters: report the values
# used in the manuscript.
AMY_MAX_MISSING_ROWS = 700      # out of 8760 hours, roughly 8 percent
AMY_MAX_INTERPOLATE = 6
AMY_MAX_IMPUTE = 130 #48

ONEBUILDING_XLSX = [
    "https://climate.onebuilding.org/sources/Region2_Asia_TMYx_EPW_Processing_locations.xlsx",
    "https://climate.onebuilding.org/sources/Region1_Africa_TMYx_EPW_Processing_locations.xlsx",
    "https://climate.onebuilding.org/sources/Region3_South_America_TMYx_EPW_Processing_locations.xlsx",
    "https://climate.onebuilding.org/sources/Region4_USA_TMYx_EPW_Processing_locations.xlsx",
    "https://climate.onebuilding.org/sources/Region4_Canada_TMYx_EPW_Processing_locations.xlsx",
    "https://climate.onebuilding.org/sources/Region4_NA_CA_Caribbean_TMYx_EPW_Processing_locations.xlsx",
    "https://climate.onebuilding.org/sources/Region5_Southwest_Pacific_TMYx_EPW_Processing_locations.xlsx",
    "https://climate.onebuilding.org/sources/Region6_Europe_TMYx_EPW_Processing_locations.xlsx",
]

PREFERRED_SUFFIX_ORDER = [
    ".TMYx.2009-2023", ".TMYx.2007-2021", ".TMYx.2004-2018", ".TMYx", ".RMY2024",
]


def log(msg: str):
    print(f"[amy-weather] {msg}", file=sys.stderr)


# --------------------------------------------------------------------------
# EPW helpers
# --------------------------------------------------------------------------

def parse_epw_header(epw_path: str) -> dict:
    with open(epw_path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().strip()
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 10 or parts[0].upper() != "LOCATION":
        raise RuntimeError(f"Unexpected EPW header in {epw_path}")
    return {
        "city": parts[1], "state": parts[2], "country": parts[3], "source": parts[4],
        "wmo": parts[5], "latitude": float(parts[6]), "longitude": float(parts[7]),
        "timezone": float(parts[8]), "elevation": float(parts[9]),
    }


def _epw_frame(epw_path: str) -> pd.DataFrame:
    df = pd.read_csv(epw_path, skiprows=8, header=None, low_memory=False)
    return pd.DataFrame({
        "month": pd.to_numeric(df[1], errors="coerce"),
        "t": pd.to_numeric(df[6], errors="coerce"),
    })


def hdd18(epw_path: str) -> float:
    """Annual heating degree days, base 18 C, from hourly dry-bulb data."""
    f = _epw_frame(epw_path)
    daily = f.groupby(f.index // 24)["t"].mean()
    return float((18.0 - daily).clip(lower=0).sum())


def monthly_hdd18(epw_path: str) -> Dict[int, float]:
    """Monthly heating degree days, base 18 C."""
    f = _epw_frame(epw_path)
    f["d"] = f.index // 24
    daily = f.groupby("d").agg(month=("month", "first"), t=("t", "mean"))
    daily["hdd"] = (18.0 - daily["t"]).clip(lower=0)
    return {int(k): round(float(v), 1) for k, v in daily.groupby("month")["hdd"].sum().items()}


# --------------------------------------------------------------------------
# TMY template resolution
# --------------------------------------------------------------------------

def _looks_like_epw(p: Optional[str]) -> bool:
    return bool(p) and str(p).lower().endswith(".epw") and os.path.isfile(str(p))


def _sidecar_epw(idf_path: Optional[str]) -> Optional[str]:
    """Multi_flow writes <idf>.weather.json next to the IDF it prepared.

    The calibration usually runs on a downstream IDF (for example the
    _RFH variant), whose name no longer matches that sidecar, so any
    sidecar in the same folder is accepted, newest first.
    """
    if not idf_path:
        return None
    folder = Path(idf_path).parent

    exact = Path(idf_path).with_suffix(".weather.json")
    candidates = [exact] if exact.is_file() else []
    candidates += sorted(
        (p for p in folder.glob("*.weather.json") if p != exact),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )

    for side in candidates:
        try:
            p = json.load(open(side)).get("epw")
            if _looks_like_epw(p):
                return p
        except Exception:
            continue
    return None


def _scan_for_tmy(idf_path: Optional[str], outdir: Path) -> Optional[str]:
    """Look for a TMY EPW in the usual weather folders."""
    roots = [outdir, Path("weather"), Path("..") / "weather"]
    if idf_path:
        roots += [Path(idf_path).parent / "weather",
                  Path(idf_path).parent.parent / "weather"]

    found = []
    for root in roots:
        try:
            if root.is_dir():
                found += [p for p in root.glob("*.epw")]
        except Exception:
            continue
    if not found:
        return None
    # prefer TMYx, then the most recently written file
    found.sort(key=lambda p: (_score_suffix(p.name), -p.stat().st_mtime))
    return str(found[0])


def _hv_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    if _haversine:
        return float(_haversine(a, b))
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def _geocode(city: str) -> Tuple[float, float]:
    r = requests.get("https://nominatim.openstreetmap.org/search",
                     params={"q": city, "format": "json", "limit": 1},
                     headers={"User-Agent": USER_AGENT}, timeout=DOWNLOAD_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if not data:
        raise RuntimeError(f"Could not geocode: {city}")
    return float(data[0]["lat"]), float(data[0]["lon"])


def _score_suffix(name: str) -> int:
    for i, suf in enumerate(PREFERRED_SUFFIX_ORDER):
        if suf.lower() in name.lower():
            return i
    return len(PREFERRED_SUFFIX_ORDER) + 1


def download_tmy_for_city(city: str, outdir: Path) -> str:
    """Last-resort TMY download from climate.onebuilding.org."""
    outdir.mkdir(parents=True, exist_ok=True)
    cache = outdir / "_index"
    cache.mkdir(parents=True, exist_ok=True)
    lat, lon = _geocode(city)
    log(f"geocoded {city} -> {lat:.4f}, {lon:.4f}")

    frames = []
    for url in ONEBUILDING_XLSX:
        local = cache / url.split("/")[-1]
        try:
            if not (local.exists() and local.stat().st_size > 0):
                r = requests.get(url, headers={"User-Agent": USER_AGENT},
                                 timeout=DOWNLOAD_TIMEOUT)
                r.raise_for_status()
                local.write_bytes(r.content)
            df = pd.read_excel(local, engine="openpyxl")
            cols = {c.lower(): c for c in df.columns}
            lat_c = next((cols[k] for k in cols if "lat" in k), None)
            lon_c = next((cols[k] for k in cols if "lon" in k), None)
            url_c = next((cols[k] for k in cols if "url" in k or "link" in k), None)
            file_c = next((cols[k] for k in cols if "file" in k and "epw" in k), None)
            if not (lat_c and lon_c and url_c):
                continue
            frames.append(pd.DataFrame({
                "lat": pd.to_numeric(df[lat_c], errors="coerce"),
                "lon": pd.to_numeric(df[lon_c], errors="coerce"),
                "url": df[url_c].astype(str),
                "epw_file": df[file_c].astype(str) if file_c else "",
            }).dropna(subset=["lat", "lon"]))
        except Exception as e:
            log(f"  index {local.name} unavailable ({e})")

    if not frames:
        raise RuntimeError("No OneBuilding station index could be loaded.")

    idx = pd.concat(frames, ignore_index=True)
    idx["dist_km"] = [_hv_km((lat, lon), (a, b)) for a, b in zip(idx["lat"], idx["lon"])]
    best = idx.sort_values("dist_km").iloc[0]
    log(f"nearest station {best['dist_km']:.1f} km away")

    base = str(best["url"]).rstrip("/")
    hint = str(best.get("epw_file") or "").strip()
    candidates: List[str] = []
    if base.lower().endswith((".zip", ".epw")):
        candidates.append(base)
    else:
        if hint:
            candidates.append(urljoin(base + "/", hint))
        r = requests.get(base + "/", headers={"User-Agent": USER_AGENT},
                         timeout=DOWNLOAD_TIMEOUT)
        r.raise_for_status()
        hrefs = re.findall(r'href=[\'"]([^\'"]+)[\'"]', r.text, flags=re.I)
        files = [h.split("?")[0] for h in hrefs
                 if h.split("?")[0].lower().endswith((".zip", ".epw"))]
        files.sort(key=lambda n: (_score_suffix(n), n))
        candidates += [urljoin(base + "/", f) for f in files]

    last_err = None
    for u in candidates:
        try:
            local = outdir / u.split("/")[-1]
            if not (local.exists() and local.stat().st_size > 0):
                with requests.get(u, stream=True, headers={"User-Agent": USER_AGENT},
                                  timeout=DOWNLOAD_TIMEOUT) as resp:
                    resp.raise_for_status()
                    with open(local, "wb") as fh:
                        for chunk in resp.iter_content(chunk_size=131072):
                            if chunk:
                                fh.write(chunk)
            if local.suffix.lower() == ".zip":
                with zipfile.ZipFile(local) as z:
                    names = [n for n in z.namelist() if n.lower().endswith(".epw")]
                    if not names:
                        raise RuntimeError("no .epw inside archive")
                    names.sort(key=lambda n: (_score_suffix(n), n))
                    target = outdir / Path(names[0]).name
                    with z.open(names[0]) as src, open(target, "wb") as dst:
                        dst.write(src.read())
                    return str(target)
            return str(local)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not download a TMY EPW: {last_err}")


def resolve_tmy_template(state: Dict[str, Any], outdir: Path) -> Tuple[str, str]:
    """Find a TMY EPW to use as the AMY template.

    Returns (path, source_label). Tried in order:
      1. state["tmy_epw_path"]
      2. the .weather.json sidecar written by Multi_flow next to the IDF
      3. state["epw_path"], the file the calibration would otherwise use
      4. download from climate.onebuilding.org for the configured city
    """
    p = state.get("tmy_epw_path")
    if _looks_like_epw(p):
        return str(p), "state.tmy_epw_path"

    p = _sidecar_epw(state.get("idf_path"))
    if p:
        return p, "idf sidecar"

    p = state.get("epw_path")
    if _looks_like_epw(p):
        return str(p), "state.epw_path"

    p = _scan_for_tmy(state.get("idf_path"), outdir)
    if p:
        return p, "weather folder scan"

    city = state.get("city") or DEFAULT_CITY
    return download_tmy_for_city(city, outdir), "onebuilding download"


# --------------------------------------------------------------------------
# Year resolution
# --------------------------------------------------------------------------

def resolve_year(state: Dict[str, Any]) -> Optional[int]:
    """Determine the calendar year of the measured data."""
    for key in ("measured_year", "amy_year"):
        v = state.get(key)
        if v:
            try:
                y = int(str(v)[:4])
                if 1900 < y < 2100:
                    return y
            except Exception:
                pass

    # Keys of the form "YYYY-MM", should the extraction schema change
    for k in (state.get("measured_monthly_kwh") or {}):
        m = re.match(r"^(\d{4})[-/]", str(k))
        if m:
            return int(m.group(1))

    # Last resort: a 4-digit year in the request text
    text = state.get("user_input") or ""
    years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", text)]
    return max(years) if years else None


# --------------------------------------------------------------------------
# AMY generation
# --------------------------------------------------------------------------

def get_global_noaa_isd_lite_file(
    wmo_index: int,
    year: int,
    output_dir: Optional[str] = None,
    allow_downloads: bool = True,
) -> str:
    """Return a NOAA ISD-Lite file for any global WMO station.

    diyepw's catalog builder currently filters the NOAA directory to station
    identifiers beginning with "7", which excludes stations such as Seoul
    (WMO 471080). This function bypasses that catalog and resolves the actual
    NOAA filename directly from the yearly ISD-Lite directory.

    The signature intentionally mirrors diyepw.get_noaa_isd_lite_file() so it
    can be monkey-patched into diyepw.create_amy_epw_file.
    """
    wmo = str(int(wmo_index)).zfill(6)
    year = int(year)
    outdir = Path(output_dir or ".")
    outdir.mkdir(parents=True, exist_ok=True)

    # First reuse any previously downloaded file for this WMO/year. NOAA ISD
    # Lite filenames have the form USAF-WBAN-YEAR.gz, e.g.
    # 471080-99999-2024.gz.
    local_matches = sorted(outdir.glob(f"{wmo}-*-{year}.gz"))
    if local_matches:
        for p in local_matches:
            if p.is_file() and p.stat().st_size > 0:
                return str(p)

    if not allow_downloads:
        raise FileNotFoundError(
            f"No cached NOAA ISD-Lite file for WMO {wmo}, year {year}"
        )

    base = f"https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{year}/"

    log(f"downloading NOAA ISD-Lite directory for {year}")
    r = requests.get(
        base,
        headers={"User-Agent": USER_AGENT},
        timeout=DOWNLOAD_TIMEOUT,
    )
    r.raise_for_status()

    # Match any WBAN component rather than assuming 99999.
    pattern = rf'href=["\']({re.escape(wmo)}-[^"\'/]+-{year}\.gz)["\']'
    names = sorted(set(re.findall(pattern, r.text, flags=re.I)))
    if not names:
        # Some directory listings may use plain text rather than quoted hrefs.
        names = sorted(set(re.findall(
            rf'\b({re.escape(wmo)}-\d{{5}}-{year}\.gz)\b',
            r.text,
            flags=re.I,
        )))

    if not names:
        raise RuntimeError(
            f"NOAA ISD-Lite directory for {year} contains no file "
            f"for WMO {wmo}"
        )

    # A WMO/USAF can theoretically have more than one WBAN entry. Prefer
    # 99999 (the common placeholder for international stations), otherwise
    # use the first deterministic match.
    remote_name = next(
        (name for name in names if f"{wmo}-99999-{year}.gz" == name),
        names[0],
    )
    target = outdir / remote_name
    url = urljoin(base, remote_name)

    log(f"downloading NOAA ISD-Lite {remote_name}")
    with requests.get(
        url,
        stream=True,
        headers={"User-Agent": USER_AGENT},
        timeout=DOWNLOAD_TIMEOUT,
    ) as resp:
        resp.raise_for_status()
        with open(target, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=131072):
                if chunk:
                    fh.write(chunk)

    if not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError(f"Downloaded ISD-Lite file is empty: {target}")

    return str(target)


class _DiyepwOverride:
    """Patch diyepw's TMY and NOAA lookups during AMY generation.

    TMY:
        Force diyepw to use the already resolved local TMY EPW.

    NOAA ISD-Lite:
        Replace diyepw's North-America-limited catalog lookup with the global
        downloader above. Patching the function in the create_amy_epw_file
        module is important because create_amy_epw_file() calls it internally.
    """

    def __init__(self, tmy_path: str, isd_dir: str):
        self.tmy_path = os.path.abspath(tmy_path)
        self.isd_dir = os.path.abspath(isd_dir)
        self._original_tmy = None
        self._original_isd = None

    def __enter__(self):
        if not os.path.isfile(self.tmy_path):
            raise FileNotFoundError(f"TMY EPW not found: {self.tmy_path}")

        self._original_tmy = _CAEF.get_tmy_epw_file
        self._original_isd = _CAEF.get_noaa_isd_lite_file

        _CAEF.get_tmy_epw_file = (
            lambda wmo_index, output_dir=None, allow_downloads=False:
            self.tmy_path
        )

        _CAEF.get_noaa_isd_lite_file = (
            lambda wmo_index, year, output_dir=None, allow_downloads=False:
            get_global_noaa_isd_lite_file(
                wmo_index=wmo_index,
                year=year,
                output_dir=output_dir or self.isd_dir,
                allow_downloads=allow_downloads,
            )
        )
        return self

    def __exit__(self, *exc):
        _CAEF.get_tmy_epw_file = self._original_tmy
        _CAEF.get_noaa_isd_lite_file = self._original_isd
        return False


def build_amy_epw(wmo: int, year: int, tmy_epw: str,
                  outdir: Path) -> Tuple[Optional[str], Dict[str, Any]]:
    """Create one AMY EPW. Returns (path or None, report)."""
    outdir.mkdir(parents=True, exist_ok=True)
    isd_cache = outdir / "_isd_cache"
    isd_cache.mkdir(parents=True, exist_ok=True)

    tmy_hdd = hdd18(tmy_epw)
    report: Dict[str, Any] = {
        "wmo": wmo,
        "year": year,
        "tmy_file": os.path.basename(tmy_epw),
        "tmy_hdd18": round(tmy_hdd, 1),
        "tmy_monthly_hdd18": monthly_hdd18(tmy_epw),
        "max_missing_rows_allowed": AMY_MAX_MISSING_ROWS,
    }
    cached = find_cached_amy(outdir, wmo, year)
    if cached:
        report["epw_repair"] = fix_amy_epw(cached, tmy_epw, year=year)
        amy_hdd = hdd18(cached)
        report.update({
            "status": "ok",
            "source": "cache",
            "amy_file": os.path.basename(cached),
            "amy_hdd18": round(amy_hdd, 1),
            "amy_vs_tmy_pct": round(100.0 * (amy_hdd - tmy_hdd) / tmy_hdd, 1),
            "amy_monthly_hdd18": monthly_hdd18(cached),
        })
        return cached, report

    try:
        with _DiyepwOverride(tmy_epw, str(isd_cache)):
            # diyepw shifts the ISD series from UTC into local time and so
            # needs the following year as well.
            for y in (year, year + 1):
                f = get_global_noaa_isd_lite_file(
                    wmo_index=wmo,
                    year=y,
                    output_dir=str(isd_cache),
                    allow_downloads=True,
                )
                a = analyze_noaa_isd_lite_file(f)
                report[f"isd_{y}_missing_rows"] = int(a["total_rows_missing"])
                report[f"isd_{y}_max_gap"] = int(a["max_consec_rows_missing"])

            path = diyepw.create_amy_epw_file(
                wmo_index=wmo, year=year,
                amy_epw_dir=str(outdir), amy_dir=str(isd_cache),
                allow_downloads=True,
                max_records_to_interpolate=AMY_MAX_INTERPOLATE,
                max_records_to_impute=AMY_MAX_IMPUTE,
                max_missing_amy_rows=AMY_MAX_MISSING_ROWS,
            )
    except Exception as e:
        report["status"] = "failed"
        report["error"] = str(e)
        return None, report
    report["epw_repair"] = fix_amy_epw(path, tmy_epw, year=year)
    amy_hdd = hdd18(path)
    report.update({
        "status": "ok",
        "amy_file": os.path.basename(path),
        "amy_hdd18": round(amy_hdd, 1),
        "amy_vs_tmy_pct": round(100.0 * (amy_hdd - tmy_hdd) / tmy_hdd, 1),
        "amy_monthly_hdd18": monthly_hdd18(path),
    })
    return path, report


# --------------------------------------------------------------------------
# LangGraph node
# --------------------------------------------------------------------------

def amy_weather(state: SimulationState) -> SimulationState:
    """Build an AMY EPW for the measured year and set epw_path to it.

    On any failure the node records the reason in weather_report and leaves
    epw_path unchanged, so the calibration still runs.
    """
    outdir = Path(state.get("weather_outdir") or "weather")

    if not _HAS_DIYEPW:
        log("diyepw is not installed; keeping the existing weather file.")
        return {**state, "weather_report": {
            "status": "skipped", "reason": "diyepw not installed (pip install diyepw)"}}

    year = resolve_year(state)
    if not year:
        log("no measured year found; keeping the existing weather file.")
        return {**state, "weather_report": {
            "status": "skipped", "reason": "no measured year in state or request text"}}

    try:
        tmy_path, tmy_source = resolve_tmy_template(dict(state), outdir)
    except Exception as e:
        log(f"no TMY template available ({e}); keeping the existing weather file.")
        return {**state, "weather_report": {
            "status": "skipped", "reason": f"no TMY template: {e}"}}

    try:
        header = parse_epw_header(tmy_path)
    except Exception as e:
        return {**state, "weather_report": {
            "status": "skipped", "reason": f"unreadable TMY header: {e}"}}

    wmo = int(re.sub(r"\D", "", header.get("wmo") or "") or 0)
    if not wmo:
        log("the TMY header carries no WMO index; keeping the existing weather file.")
        return {**state, "weather_report": {
            "status": "skipped", "reason": "no WMO index in the TMY header",
            "tmy_file": os.path.basename(tmy_path)}}

    log(f"station {header.get('city')} WMO {wmo}, template from {tmy_source}, year {year}")
    amy_path, report = build_amy_epw(wmo, year, tmy_path, outdir / "amy")
    report["tmy_source"] = tmy_source

    if not amy_path:
        log(f"AMY generation failed: {report.get('error')}")
        return {**state, "weather_report": report}

    log(f"AMY {year}: {os.path.basename(amy_path)}  "
        f"HDD18 {report['amy_hdd18']:,.0f} ({report['amy_vs_tmy_pct']:+.1f}% vs TMY)")

    report["runperiod_start_day"] = set_runperiod_start_day(
        state.get("idf_path"), year, state.get("idd_path"))
    return {
        **state,
        "epw_path": os.path.abspath(amy_path),
        "tmy_epw_path": os.path.abspath(tmy_path),
        "weather_report": report,
        "message": (f"AMY weather for {year}: {os.path.basename(amy_path)} "
                    f"(HDD18 {report['amy_vs_tmy_pct']:+.1f}% vs typical year)"),
    }
