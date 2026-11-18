# Kernel database changelog

Versions follow [Pride Versioning](https://pridever.org) (`PROUD.DEFAULT.SHAME`); see
`manifest.toml` for the rules. Each released version has a git tag `<key>/v<version>`,
e.g. `n2_77k/v1.1.0` or `structures/v1.0.0`. Only the current version of each kernel is
offered in the app.

Dates before 2026-11-18 are reconstructed from git history.

## Structure library

### structures 1.0.0 — Carbon 2023
- The 78 atomistic structures of Vallejos-Burgos et al., Carbon 215 (2023) 118431,
  https://doi.org/10.1016/j.carbon.2023.118431. All at revision 1.
- 009 and 013 are excluded (low density, never formed a solid framework).

Known problems, to be fixed in structures 1.0.1 (shame):
- About half of the structures were not fully energy-minimized.
- 073 was truncated: its box is 77.26 × 77.28 × 64.88 Å instead of cubic, and its atoms
  span only 49.5 Å in z.

Planned for structures 1.1.0: 009 and 013 regenerated at higher density, so they are
connected in 3D.

## N₂ at 77 K (`n2_77k`)

### 1.1.0 — 2024-05-30 (commit 22ace40)
- Large pores modeled with 2D-NLDFT slit pores (31 columns, 10–415 Å) instead of the
  Kelvin equation. Not part of the paper.

### 1.0.0 — 2023-08-02 — Carbon 2023 paper (tag `n2_77k/v1.0.0`)
- The kernel used in the paper: the 78 atomistic structures plus large pores from the
  Kelvin equation. Kept in `legacy/n2_77k_kelvin_v1.0.0_carbon2023.csv`.

## CO₂ at 298.15 K (`co2_298k`)

### 0.1.0 — 2025-02-05 (commits 09c7ce6, 2b02ced)
- First release on the Carbon 2023 structures. Unpublished.

## CO₂ at 273 K (`co2_273k`)

### 0.1.0 — 2026-06-29 (commit 657d6bf)
- First release on the Carbon 2023 structures. Unpublished.

## H₂ at 77 K (`h2_77k`)

### 0.1.0 — 2026-06-29 (commit 657d6bf)
- First release on the Carbon 2023 structures, 0.01–120 bar. Unpublished.

## Ar at 87 K (`ar_87k`)

### 0.1.0 — 2026-09-14 (commit ea12eee)
- Carbon 2023 atomistic structures only. Unpublished.
- The 11 Kelvin large-pore columns (workbook CB–CL) are left out: their Kelvin constant is
  about 4.5× too small. Full sheet in `legacy/ar_87k_full_sheet_with_kelvin_cb-cl.csv`.
  Regenerating them will be 0.1.1 (shame).
