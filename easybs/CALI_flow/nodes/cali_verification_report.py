# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 12:02:12 2026

@author: Xiguan Liang @SKKU
"""

# ./CALI_flow/nodes/cali_verification_report.py
#
# Called at the end of every calibration stage.
# Produces:
#   1. A printed monthly comparison table with formulas and intermediate values
#   2. A bar chart (measured vs simulated gas, per month) saved to disk at 300 dpi
#
# The table is designed so every number can be reproduced in Excel without
# running any code: all inputs and intermediate steps are shown.

from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


# --------------------------------------------------------------------------
# Core metric functions, written step-by-step so formulas can be audited
# --------------------------------------------------------------------------

def monthly_residuals(measured: Dict[str, float],
                      simulated_fuel: Dict[int, float]) -> Dict[int, dict]:
    """Return one dict per month with all intermediate values."""
    rows = {}
    for mm in sorted(int(k) for k in measured):
        meas = float(measured[str(mm)])
        sim = float(simulated_fuel.get(mm, float("nan")))
        resid = sim - meas
        rows[mm] = {
            "month": mm,
            "measured_kwh": meas,
            "simulated_kwh": sim,
            "residual_kwh": resid,
            "residual_pct": 100.0 * resid / meas if meas else float("nan"),
        }
    return rows


def compute_and_explain(rows: Dict[int, dict]) -> dict:
    """Compute CVRMSE and NMBE, returning every intermediate value."""
    months = sorted(rows)
    n = len(months)
    if n < 2:
        return {}

    meas_vals = [rows[m]["measured_kwh"] for m in months]
    sim_vals  = [rows[m]["simulated_kwh"] for m in months]
    resid_vals = [rows[m]["residual_kwh"] for m in months]

    mean_meas  = sum(meas_vals) / n
    sum_resid  = sum(resid_vals)
    sum_sq_resid = sum(r ** 2 for r in resid_vals)
    mse        = sum_sq_resid / (n - 1)
    rmse       = math.sqrt(mse)
    cvrmse_pct = 100.0 * rmse / mean_meas
    nmbe_pct   = 100.0 * sum_resid / ((n - 1) * mean_meas)

    return {
        "n": n,
        "mean_measured_kwh": mean_meas,
        "sum_residuals": sum_resid,
        "sum_sq_residuals": sum_sq_resid,
        "mse": mse,
        "rmse": rmse,
        "cvrmse_pct": cvrmse_pct,
        "nmbe_pct": nmbe_pct,
    }


# --------------------------------------------------------------------------
# Printed report
# --------------------------------------------------------------------------

def print_verification_report(stage_name: str,
                               idf_path: str, epw_path: str,
                               measured: Dict[str, float],
                               simulated_heat: Dict[int, float],
                               boiler_eff: float) -> Dict[int, dict]:
    """Print the full comparison table including formula steps.

    Returns the monthly rows dict so the caller can chain or plot.
    """
    simulated_fuel = {mm: v / boiler_eff for mm, v in simulated_heat.items()}
    rows = monthly_residuals(measured, simulated_fuel)
    stats = compute_and_explain(rows)
    months = sorted(rows)

    sep = "=" * 88
    print(f"\n{sep}")
    print(f"  CALIBRATION VERIFICATION REPORT  —  {stage_name}")
    print(sep)
    print(f"  IDF  : {idf_path}")
    print(f"  EPW  : {epw_path}")
    print(f"  Boiler efficiency applied: η = {boiler_eff:.4f}")
    print(f"  Simulated gas = EnergyPlus heat output ÷ η")
    print(sep)

    # Column header
    hdr = (f"{'Month':>5}  {'Measured':>12}  {'Sim Heat':>12}  "
           f"{'Sim Gas':>12}  {'Residual':>12}  {'Resid %':>8}")
    print(hdr)
    print(f"  {'':5}  {'(kWh gas)':>12}  {'(kWh heat)':>12}  "
          f"{'(kWh gas)':>12}  {'(Sim-Meas)':>12}  {'of meas':>8}")
    print("-" * 88)

    for mm in months:
        r = rows[mm]
        heat = simulated_heat.get(mm, float("nan"))
        print(f"  {MONTH_NAMES[mm]:>5}  {r['measured_kwh']:>12.2f}  {heat:>12.2f}  "
              f"  {r['simulated_kwh']:>11.2f}  {r['residual_kwh']:>+12.2f}  "
              f"{r['residual_pct']:>+7.1f}%")

    print("-" * 88)
    if stats:
        print(f"\n  n (months used) = {stats['n']}")
        print(f"  Mean measured   = {stats['mean_measured_kwh']:.4f} kWh")
        print(f"  Σ residuals     = {stats['sum_residuals']:+.4f} kWh")
        print(f"  Σ residuals²    = {stats['sum_sq_residuals']:.4f}")
        print()
        print(textwrap.dedent(f"""\
          Formula (ASHRAE Guideline 14):
            CVRMSE = 100 × √[Σ(sim_gas − meas)² / (n − 1)] / mean(meas)
                   = 100 × √[{stats['sum_sq_residuals']:.4f} / {stats['n'] - 1}] / {stats['mean_measured_kwh']:.4f}
                   = 100 × √[{stats['mse']:.4f}] / {stats['mean_measured_kwh']:.4f}
                   = 100 × {stats['rmse']:.4f} / {stats['mean_measured_kwh']:.4f}
                   = {stats['cvrmse_pct']:.4f} %

            NMBE   = 100 × Σ(sim_gas − meas) / [(n − 1) × mean(meas)]
                   = 100 × {stats['sum_residuals']:+.4f} / [{stats['n'] - 1} × {stats['mean_measured_kwh']:.4f}]
                   = {stats['nmbe_pct']:+.4f} %

          ASHRAE thresholds (monthly): CVRMSE ≤ 15%  |  NMBE ≤ ±5%
          Result: CVRMSE = {stats['cvrmse_pct']:.3f} % {'✓' if stats['cvrmse_pct'] <= 15 else '✗'}   |  NMBE = {stats['nmbe_pct']:+.3f} % {'✓' if abs(stats['nmbe_pct']) <= 5 else '✗'}
        """))
    print(sep)

    return rows


# --------------------------------------------------------------------------
# Bar chart
# --------------------------------------------------------------------------

def save_comparison_chart(rows: Dict[int, dict],
                           stats: dict,
                           stage_name: str,
                           out_path: str | Path,
                           idf_name: str = "",
                           epw_name: str = "") -> Path:
    """Grouped bar chart: measured vs simulated fuel, residual as line.

    Saves at 300 dpi. Returns the output path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    months = sorted(rows)
    month_labels = [MONTH_NAMES[m] for m in months]
    meas_vals = [rows[m]["measured_kwh"] for m in months]
    sim_vals  = [rows[m]["simulated_kwh"] for m in months]
    resid_vals = [rows[m]["residual_kwh"] for m in months]

    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.08})

    # ── Upper panel: grouped bars ─────────────────────────────────────────
    x = list(range(len(months)))
    w = 0.38
    bar_meas = ax1.bar([xi - w / 2 for xi in x], meas_vals, width=w,
                       color="#2166ac", alpha=0.85, label="Measured")
    bar_sim  = ax1.bar([xi + w / 2 for xi in x], sim_vals,  width=w,
                       color="#d6604d", alpha=0.85, label="Simulated")

    ax1.set_ylabel("Monthly heating (kWh)", fontsize=11)
    ax1.legend(framealpha=0.9, fontsize=10)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    # Title: include key metrics
    cvr = stats.get("cvrmse_pct", float("nan"))
    nmb = stats.get("nmbe_pct", float("nan"))
    title_main = f"{stage_name}"
    title_metrics = (f"CVRMSE = {cvr:.3f}%   NMBE = {nmb:+.3f}%")
    ax1.set_title(f"{title_main}\n{title_metrics}", fontsize=11, pad=8)

    # ── Lower panel: residual bar + zero line ─────────────────────────────
    colours = ["#4dac26" if r >= 0 else "#d01c8b" for r in resid_vals]
    ax2.bar(x, resid_vals, width=0.55, color=colours, alpha=0.80)
    ax2.axhline(0, color="black", linewidth=0.9, linestyle="--")
    ax2.set_ylabel("Residual\n(Sim − Meas, kWh)", fontsize=10)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+,.0f}"))
    ax2.set_xticks(x)
    ax2.set_xticklabels(month_labels, fontsize=11)

    # ── Annotation: value labels on bars ─────────────────────────────────
    for xi, (m, s) in zip(x, zip(meas_vals, sim_vals)):
        ax1.text(xi - w / 2, m + max(meas_vals) * 0.01,
                 f"{m:,.0f}", ha="center", va="bottom", fontsize=7.5,
                 color="#2166ac", fontweight="bold")
        ax1.text(xi + w / 2, s + max(meas_vals) * 0.01,
                 f"{s:,.0f}", ha="center", va="bottom", fontsize=7.5,
                 color="#d6604d", fontweight="bold")

    # ── Footer: file provenance ───────────────────────────────────────────
    footer = []
    if idf_name:
        footer.append(f"IDF: {idf_name}")
    if epw_name:
        footer.append(f"EPW: {epw_name}")
    footer.append(
        f"Formula: CVRMSE = 100×√[Σ(sim−meas)²/(n−1)]/mean(meas); "
        f"NMBE = 100×Σ(sim−meas)/[(n−1)×mean(meas)]"
    )
    fig.text(0.5, -0.02, "\n".join(footer),
             ha="center", va="top", fontsize=7.5, color="dimgray",
             wrap=True)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# Convenience wrapper called from each stage's main()
# --------------------------------------------------------------------------

def run_verification(stage_name: str,
                     idf_path: str, epw_path: str,
                     measured: Dict[str, float],
                     simulated_heat: Dict[int, float],
                     boiler_eff: float,
                     chart_out: Optional[str | Path] = None) -> dict:
    """Print the report and optionally save the chart.

    Returns the stats dict so the caller can check metrics.
    """
    rows = print_verification_report(
        stage_name, idf_path, epw_path,
        measured, simulated_heat, boiler_eff)
    stats = compute_and_explain(rows)

    if chart_out is not None:
        saved = save_comparison_chart(
            rows, stats, stage_name, chart_out,
            idf_name=Path(idf_path).name,
            epw_name=Path(epw_path).name)
        print(f"[INFO] Chart saved to: {saved.resolve()}")

    return stats
