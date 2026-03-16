# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

3D-VIS is a scientific Streamlit web application that predicts 3D nanostructures of nanoporous carbons from experimental gas adsorption isotherms, using a kernel of pre-calculated atomistic structures. Published in Carbon journal (2023), DOI: 10.1016/j.carbon.2023.118431.

## Running the App

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

No build step, test suite, or linter is configured.

## Architecture

### Core Flow

The app (`streamlit_app.py`) implements a linear analysis pipeline:

1. **Kernel selection** — user picks gas/temperature (N₂ 77 K or CO₂ 298.15 K)
2. **Data loading** — experimental isotherm uploaded (TSV or Belsorp .DAT), or default example used
3. **Data cleaning** — low-pressure noise removal via slider; interpolation to kernel pressure grid
4. **Inverse solve** — Non-Negative Least Squares (`scipy.optimize.nnls`) fits experimental data as a linear combination of kernel isotherms
5. **Results** — contribution plots, 3D structure PNGs, simulated TEM images, morphological statistics
6. **PSD output** — Pore Size Distribution calculated via Pascal-triangle smoothing; exported as TSV

### Key Data Sources (`kernel.xlsx`)

All pre-calculated isotherms and structural parameters live in `kernel.xlsx`:
- Sheet `Details`: structural parameters for each kernel structure (density, temp, surface area, etc.)
- Sheet `N2 77 K 1CLJ_2D-NLDFT`: N₂ isotherm matrix (93 pressure points × 110 structures)
- Sheet `CO2 298 K`: CO₂ isotherm matrix
- Sheet `Poreblazer PSDs_2`: pore size distributions for each structure

Supporting assets:
- `structures/` — 78 `.xyz` atomic coordinate files
- `rendered structures/` — PNG renders of each structure
- `simulated TEM/` — TIF simulated TEM images

### Belsorp .DAT Format

The app has special parsing for Belsorp instrument files (see `examples/*.DAT`). A "Force P₀" control overrides the saturation pressure read from these files.

### Notebook

`3D_calculation.ipynb` mirrors the app's algorithms for offline/batch analysis and research iteration. It uses the same kernel data and NNLS approach.
