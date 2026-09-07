# v0.2.1
Fixed some known issues.

# Easy-BS — RFH Calibration Framework

An LLM-agent framework that generates EnergyPlus radiant floor heating models
from plain-language building descriptions and calibrates them against metered
consumption, using parameter priors selected from the building's stated
structure type and construction era.

<!-- Replace the version tag and badge URLs with your own before publishing -->
![version](https://img.shields.io/badge/release-v0.2.0-blue)
![EnergyPlus](https://img.shields.io/badge/EnergyPlus-8.9.0-green)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

---
<p align="center">
  <img src="docs/Fig1.svg" width="400">
</p>

## What this release changes

The changes below address both problems.

| | Previous release | This release |
|---|---|---|
| Weather data | Typical meteorological year | Actual meteorological year, built per calendar year |
| Measured data | 2021–2024 monthly average | Single calendar year, matched to the weather |
| Fuel accounting | Simulated heat compared to metered gas | Heat converted to fuel by boiler efficiency |
| Decision variables | Dimensionless conductivity multipliers | Assembly U-values |
| Search bounds | Fixed, hand-specified | Selected from a prior library keyed to structure and era |
| Plausibility penalties | Applied to both objectives | Removed; bounds are physical by construction |
| Envelope constructions | EnergyPlus sample slab and a single-material roof | Layered assemblies matched to the stated vintage |
| Independent validation | None | Six priors checked against in-situ measurement |
| Final metrics | CVRMSE 13.56%, NMBE −2.13% | **CVRMSE 10.97%, NMBE −3.01%** |

---

## Headline result

For a 1969 masonry dwelling in Seoul, described to the framework only as
`masonry, built 1969, windows replaced with double glazing`:

| Parameter | Prior selected | Calibrated | Measured independently | In prior |
|---|---|---|---|---|
| Wall U-value, W/(m²·K) | 1.40 – 2.10 | 1.486 | 1.810 | yes |
| Roof U-value, W/(m²·K) | 1.20 – 2.10 | 1.357 | 1.605 | yes |
| Floor U-value, W/(m²·K) | 1.20 – 2.10 | 1.310 | 1.705 | yes |
| Window U-value, W/(m²·K) | 2.20 – 3.60 | 2.417 | 2.820 | yes |
| Infiltration, ACH | 0.25 – 0.60 | 0.345 | 0.357 | yes |
| Boiler efficiency | 0.75 – 0.88 | 0.865 | 0.820 (nameplate) | yes |

The measurements were obtained by in-situ heat flux and tracer gas decay.
They were **not** supplied to the framework and take no part in the
calibration. Every prior brackets its measured value, and the calibrated
infiltration rate falls within 3.4% of the tracer gas result.

Final agreement with seven months of metered gas consumption:
**CVRMSE 10.97%, NMBE −3.01%**, satisfying the monthly criteria of
ASHRAE Guideline 14 (≤15% and ≤±5%).

---

## Changes in detail

### 1. Year-specific weather

The previous version paired multi-year averaged consumption with typical-year
weather. The calendar year used in this release proved 15.6% milder than the
typical year (HDD18 2,226 against 2,648), so the earlier pairing introduced a
systematic mismatch that the optimizer absorbed into envelope parameters.

A new node builds an AMY EPW file automatically from the NOAA Integrated
Surface Database for the station nearest the site, using the typical-year file
as a template for the solar components that the hourly surface record does not
carry. The node reports data completeness, degree-day comparison, and the
source of every substituted value.

```
CALI_flow/nodes/amy_weather.py       AMY generation and caching
CALI_flow/nodes/amy_epw_fix.py       repairs pressure column and leap-year flag
CALI_flow/nodes/amy_helpers.py       cache lookup, calendar alignment
```

### 2. Fuel accounting

The measured record is gas purchased; EnergyPlus reports heat delivered.
The previous version compared the two directly, an error of approximately 18%
at the calibrated boiler efficiency. Simulated heat is now divided by the
boiler efficiency before comparison, and that efficiency is itself a bounded
calibration parameter.

### 3. Parameter priors keyed to the building

Search bounds are no longer specified by hand. A prior library maps structure
class and construction era to admissible ranges for each envelope element,
with era boundaries following the regulatory history of Korean residential
insulation requirements. The agent extracts the structure type, construction
year and glazing status from the user's request and selects the applicable
entry.

```python
PRIOR_LIBRARY[("masonry", "pre1980")] = {
    "u_wall":    (1.40, 2.10),
    "u_roof":    (1.20, 2.10),
    "u_floor":   (1.20, 2.10),
    "infil_ach": (0.25, 0.60),
}
```

Because every reachable point is physically admissible for the stated class of
building, the plausibility penalties of the previous version were removed. The
objectives are now the calibration metrics themselves, so reported values are
the values the optimizer minimised.

### 4. U-value parameterization

Decision variables are assembly U-values rather than dimensionless multipliers
on a baseline conductivity. Each construction retains its real layers and
thermal mass; one massless layer per assembly carries the uncertain
resistance, solved from the target U-value. Bounds are therefore stated in the
same unit as measurements and as code requirements.

### 5. Realistic constructions

The generated model previously used the EnergyPlus sample radiant slab and a
single-material placeholder for the roof and non-radiant floors. Both are
replaced by layered assemblies appropriate to the stated vintage. Independent
measurement confirmed that the case building has no under-slab insulation,
contradicting the assumed build-up.

### 6. Reproducibility

Every stage now writes a verification report giving the per-month comparison,
the formula expansion with substituted values, and the source IDF and weather
file. Any reported metric can be recomputed in a spreadsheet from the released
artifacts.

```
cali_verification_report.py      per-month table and 300 dpi comparison chart
extract_stage_progression.py     parameter progression across all stages
plot_pareto_stages.py            Pareto fronts and convergence, per stage
```

---



## Repository layout

```
easybs/
├── RFH_flow/            radiant floor heating model generation
│   └── nodes/
│       ├── text_normalize.py        deterministic input normalization
│       ├── llm_router.py            intent classification, schema-constrained
│       ├── rfh_adder.py             zone resolution and construction
│       └── rfh_lib.py               EnergyPlus object assembly
├── CALI_flow/           staged calibration
│   └── nodes/
│       ├── amy_weather.py           year-specific weather
│       ├── Cali_Envelope.py         stage 1: envelope, infiltration, boiler
│       ├── Cali_Tset_Occ.py         stage 2: occupancy and setpoints
│       └── Cali_Tset_Detail.py      stages 3 and 4: schedule refinement
└── Multi_flow/          multi-zone geometry generation
```

---

## Requirements

```
EnergyPlus 8.9.0
Python 3.11+
numpy>=2.0   pandas>=2.2   pymoo>=0.6.1   eppy   geomeppy   diyepw
```

`diyepw` requires NumPy 2. If the host environment is pinned to NumPy 1,
`amy_weather.py` will run the weather module through a separate interpreter;
set `EASYBS_AMY_PYTHON` to that interpreter's path.

---

## Reproducing the reported result

```bash
# 1. Generate the multi-zone geometry and the RFH model
python Multi_flow/run_graph.py --config config.json

# 2. Run the staged calibration; the AMY file is built automatically
python CALI_flow/run_graph.py --config cali_runtime_config.json

# 3. Regenerate the tables and figures
python extract_stage_progression.py
python plot_pareto_stages.py
```

Expected output: `CVRMSE = 10.967%`, `NMBE = −3.008%`.

To verify independently, simulate the released `After2_Cali_RFH.idf` against
`KOR_SO_Seoul.WS.471080_AMY_2024.epw`, divide the monthly
`DistrictHeating:Facility` output by 3.6 × 10⁶ to obtain kWh, then divide by
the boiler efficiency recorded in `stage1_result.json` to obtain equivalent
gas consumption.

---

## Citation


---

## License

MIT. See [LICENSE](LICENSE).
