# RIME-X Reproduction: Regional Climate Impact Emulator

> Reproducing the methodology presented in the **RIME-X v1.0** framework for regional climate impact emulation using ISIMIP3b, CMIP6, and MAGICC7.

---

## Overview

This repository documents the reproduction of the paper:

> **RIME-X v1.0: Combining Simple Climate Models, Earth System Models, and Climate Impact Models into a Unified Statistical Emulator for Regional Climate Indicators**

The objective is to reproduce the complete Climate Impact Explorer (CIE) workflow, understand every processing step, verify the published methodology, and generate regional climate impact projections.

This work is intended for research and educational purposes and is **not an official implementation** of the RIME-X project.

---

## Objectives

- Reproduce the complete RIME-X preprocessing workflow
- Understand the internal architecture of the framework
- Reproduce published regional impact maps
- Validate reproduced results against the Climate Impact Explorer (CIE)
- Document the entire pipeline for future researchers

---

# Workflow

The complete RIME-X workflow is illustrated below.

```
CMIP6 Climate Simulations
            │
            ▼
21-Year Running Mean
            │
            ▼
Global Warming Level Detection
            │
            ▼
Extract ISIMIP Impact Simulations
            │
            ▼
Group Simulations by Warming Level
            │
            ▼
Weighted Quantile Calculation
            │
            ▼
Quantile Maps (.nc)
            │
            ▼
MAGICC7 Global Temperature Ensemble
            │
            ▼
Interpolation
            │
            ▼
Predicted Impact Maps
            │
            ▼
Climate Impact Explorer Visualization
```




# Methodology

The reproduction follows the original workflow implemented in RIME-X.

## 1. Climate Simulations

Input climate simulations are obtained from

- CMIP6
- ISIMIP3b

Variables include:

- Temperature
- Precipitation
- Cooling Degree Days
- Heating Degree Days
- Crop Yield
- Water Availability
- Additional climate impact indicators

---

## 2. Running Mean

For each simulation,

- annual values are computed
- a **21-year centered running mean** is calculated

This smooths internal climate variability before matching warming levels.

---

## 3. Warming-Level Matching

Every simulation is matched with global warming levels.

Example

| Warming Level | Year |
|---------------|------|
| 1.5°C | 2032 |
| 2.0°C | 2054 |
| 2.5°C | 2071 |

The warming level file links each climate model to the year at which the target warming level is reached.

---

## 4. Impact Collection

For every warming level:

- corresponding ISIMIP simulations are extracted
- transformed into anomalies
- grouped by warming level

---

## 5. Quantile Calculation

Each warming level contains multiple climate model realizations.

RIME-X computes

- weighted quantiles
- model weighting
- optional equal model weighting

Result:

```
warming level
        ↓
distribution
        ↓
5%
25%
50%
75%
95%
```

---

## 6. Quantile Maps

The quantiles are stored as NetCDF files.

Dimensions

```
warming_level
quantile
latitude
longitude
```

Example

```
cooling_degree_days_annual_quantilemaps.nc
```

---

## 7. MAGICC7 Integration

The regional quantile maps are combined with

- MAGICC7 global temperature ensembles

using

```
RegularGridInterpolator
```

to predict

```
Impact(year, latitude, longitude)
```

---

## 8. Validation

The reproduced outputs are compared against

- Climate Impact Explorer
- published RIME-X figures

Validation includes

- visual comparison
- anomaly maps
- regional averages
- RMSE
- MAE

---

# Current Progress

✅ Repository setup

✅ Environment configured

✅ Quantile map generation understood

✅ Warming-level interpolation reproduced

✅ Regional averaging reproduced

✅ Quantile prediction pipeline understood

✅ Climate Impact Explorer workflow analyzed

✅ Validation scripts explored

🔄 Exact MAGICC7 reproduction (requires original ensemble data)

---

# Example Outputs

Current work reproduces regional quantile maps such as

- Cooling Degree Days
- Heating Degree Days
- Temperature
- Precipitation

for

- Vanuatu
- Italy
- Global regions

using ISIMIP3b climate impacts.

---

# Requirements

Python ≥ 3.9

Major packages

```
numpy
pandas
xarray
scipy
geopandas
matplotlib
rioxarray
netCDF4
xesmf
tqdm
```

Install

```bash
pip install -r requirements.txt
```

---

# Running

Generate warming-level quantile maps

```bash
python make_quantile_maps.py \
    --indicator cooling_degree_days \
    --regional \
    --season annual
```

Generate predictions

```python
prediction = make_quantilemap_prediction(
    quantile_map,
    gmt_ensemble
)
```

---

# Verification

The reproduced outputs are verified by comparing them against the Climate Impact Explorer.

Comparison includes

- spatial maps
- regional averages
- quantiles
- anomaly maps

---

# Challenges

The original Climate Impact Explorer uses a proprietary MAGICC7 ensemble (600 members), which is not publicly distributed.

Without these data, the quantile maps can be reproduced exactly, but year-specific projections cannot be matched perfectly.

---

# Acknowledgements

This work reproduces the methodology developed by the RIME-X authors.

If you use this repository, please also cite the original work.

```
Werning et al.

RIME-X v1.0:
Combining Simple Climate Models,
Earth System Models,
and Climate Impact Models
into a Unified Statistical Emulator
for Regional Climate Indicators.
```

---

# Disclaimer

This repository is an independent research reproduction and is **not affiliated with or endorsed by the original RIME-X developers or Climate Analytics**.

The goal is to improve transparency, reproducibility, and understanding of the RIME-X framework.

---

# Author

**Imran Ul Haq**

Research Engineer  
Weather & Climate Services (WCS), Pakistan

Research Interests

- Climate Informatics
- Climate Impact Modeling
- Machine Learning
- Earth System Science
- Scientific Reproducibility
