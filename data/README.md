# Kernel database

Everything the app loads lives here as CSV, versioned per structure library and per kernel.

```
manifest.toml            what the app loads, versions, labels, excluded structures
versions.lock.json       sha256 of every data file for each released version
CHANGELOG.md             what changed in each version, and why
structures/
  structures.csv         one row per atomistic structure (id, rev, rev_note, properties)
  poreblazer_psd.csv     Poreblazer PSD of each structure (pore width in Angstrom)
kernels/<key>/
  isotherm.csv           pressure + one column per structure ID
  dft_pores.csv          DFT large-pore columns: id and pore width (N₂ only)
legacy/                  verbatim dumps of workbook sheets the app does not read
```

## IDs and revisions

- Atomistic structures: `001`, `002`, … The paper's "structure #25" is `025`. An ID is never
  reused or renumbered, even when the structure is retired.
- Fixing a structure (same recipe, mistake corrected) keeps its ID and increases `rev`,
  with the reason in `rev_note`. A deliberately different recipe gets a new ID.
- DFT large-pore columns: `DFT2D-<pore width in Angstrom>`, e.g. `DFT2D-100`.

## Releasing a change

1. Edit `kernel.xlsx` (the authoring workbook), then export it:
   `.venv/bin/python tools/export_kernels.py`
2. Check what changed: `.venv/bin/python tools/data_lock.py`. A changed file of an already
   released version is an error: bump that version in `manifest.toml`
   (PROUD.DEFAULT.SHAME, see the manifest header).
3. A new structure library version means every kernel must be recomputed and pinned to it
   (`structures = "..."` in each kernel); the lock check fails otherwise.
4. Describe the change in `CHANGELOG.md`.
5. Record the new versions: `.venv/bin/python tools/data_lock.py --record`
6. Commit, and tag each new version: `git tag n2_77k/v1.1.1`, `git tag structures/v1.0.1`.

## legacy/

Sheets of `kernel.xlsx` the app does not use, with empty rows and columns dropped; the
header row is the original (0-based) column position.

| File | Workbook sheet |
|---|---|
| `n2_77k_kelvin_v1.0.0_carbon2023.csv` | `N2 77 K 1CLJ` (N₂ kernel 1.0.0, used in the paper) |
| `n2_77k_kelvin_rc1_with_desorption.csv` | `N2 77 K 1CLJ_rc1` |
| `ar_87k_full_sheet_with_kelvin_cb-cl.csv` | `Ar 87 K 1CLJ`, including the broken CB–CL columns |
| `isosteric_heat_n2_77k.csv` | `Isosteric heat` |
| `poreblazer_psds.csv` | `Poreblazer PSDs` (the app uses `Poreblazer PSDs_2`) |
| `n2_nldft_psds.csv`, `n2_2d-nldft_psds.csv`, `n2_qsdft_psds.csv` | DFT PSDs of each structure |
