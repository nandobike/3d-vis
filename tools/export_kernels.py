"""
Export the kernel workbook (kernel.xlsx) to the versioned CSV files under data/.

The CSV files are the source of truth read by the app. kernel.xlsx is only an authoring
format: edit it, run this script, then run tools/data_lock.py to check whether the data
changed and needs a version bump in data/manifest.toml.

    python tools/export_kernels.py [path/to/kernel.xlsx]

Every column is written under its stable ID: atomistic structures are "001", "002", ...
and DFT large-pore columns are "DFT2D-<pore width in Angstrom, 3 digits>". Floats are
written with Python's repr, so reading them back with float_precision="round_trip"
reproduces the workbook values bit for bit.

Sheets the app does not read are dumped verbatim (as a raw cell grid) to data/legacy/
so nothing is lost when the workbook is no longer the source of truth.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / 'data'

N_ATOMISTIC = 78
#First data row in every isotherm sheet (rows above hold titles, column numbers, DFT widths)
ISOTHERM_FIRST_ROW = 8
#Row holding the 1..N column numbers in isotherm sheets, and the DFT pore widths
ISOTHERM_NUMBER_ROW = 4
ISOTHERM_WIDTH_ROW = 2

#kernel key -> (sheet, number of pressure points, has 2D-NLDFT large-pore columns)
KERNEL_SHEETS = {
    'n2_77k': ('N2 77 K 1CLJ_2D-NLDFT', 93, True),
    'co2_298k': ('CO2 298 K', 50, False),
    'co2_273k': ('CO2 273 K', 50, False),
    'h2_77k': ('H2 77 K', 35, False),
    #The Ar sheet's 11 trailing Kelvin columns (Excel CB-CL) are broken and not exported
    #here; the full sheet is kept in data/legacy/.
    'ar_87k': ('Ar 87 K 1CLJ', 64, False),
}

PSD_SHEET = 'Poreblazer PSDs_2'
PSD_FIRST_ROW = 6
PSD_ROWS = 210
PSD_LABEL_ROW = 5

#Sheets exported verbatim to data/legacy/ (not read by the app)
LEGACY_SHEETS = {
    'N2 77 K 1CLJ': 'n2_77k_kelvin_v1.0.0_carbon2023.csv',
    'N2 77 K 1CLJ_rc1': 'n2_77k_kelvin_rc1_with_desorption.csv',
    'Ar 87 K 1CLJ': 'ar_87k_full_sheet_with_kelvin_cb-cl.csv',
    'Isosteric heat': 'isosteric_heat_n2_77k.csv',
    'Poreblazer PSDs': 'poreblazer_psds.csv',
    'N2 NLDFT PSDs': 'n2_nldft_psds.csv',
    'N2 2D-NLDFT': 'n2_2d-nldft_psds.csv',
    'N2 QSDFT PSDs': 'n2_qsdft_psds.csv',
}


def structure_id(number):
    return f'{number:03d}'


def dft_id(width_angstrom):
    return f'DFT2D-{int(round(width_angstrom)):03d}'


def write_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format=None, lineterminator='\n')
    print(f'  wrote {path.relative_to(REPO)}  {df.shape}')


def read_sheet(workbook, sheet):
    return pd.read_excel(workbook, sheet_name=sheet, header=None, engine='openpyxl')


def export_structures(workbook, details, psd_sheet):
    atomistic = details.iloc[:N_ATOMISTIC].copy()
    numbers = atomistic.index.to_numpy()
    if not np.array_equal(numbers, np.arange(1, N_ATOMISTIC + 1)):
        raise ValueError('Details rows 1..78 are not numbered 1..78')
    labels = psd_sheet.iloc[PSD_LABEL_ROW, 1:N_ATOMISTIC + 1].tolist()
    meta = pd.DataFrame({
        'id': [structure_id(n) for n in numbers],
        'rev': 1,
        'rev_note': 'Carbon 2023',
        #Poreblazer run label as written in the workbook (e.g. "01_4")
        'poreblazer_label': [l if isinstance(l, str) else f'{int(l)}' for l in labels],
    })
    atomistic = atomistic.reset_index(drop=True)
    write_csv(pd.concat([meta, atomistic], axis=1), DATA / 'structures' / 'structures.csv')

    psd = psd_sheet.iloc[PSD_FIRST_ROW:PSD_FIRST_ROW + PSD_ROWS, :N_ATOMISTIC + 1]
    psd_numbers = psd_sheet.iloc[ISOTHERM_NUMBER_ROW, 1:N_ATOMISTIC + 1].astype(int).tolist()
    if psd_numbers != list(range(1, N_ATOMISTIC + 1)):
        raise ValueError(f'{PSD_SHEET}: column numbers are not 1..78')
    #Columns past the atomistic ones (DFT) must be all zero: the loader pads them back
    dft_part = psd_sheet.iloc[PSD_FIRST_ROW:PSD_FIRST_ROW + PSD_ROWS, N_ATOMISTIC + 1:]
    if (dft_part.to_numpy(dtype=float) != 0).any():
        raise ValueError(f'{PSD_SHEET}: DFT columns are not all zero')
    psd = psd.copy()
    psd.columns = ['pore_width_A'] + [structure_id(n) for n in psd_numbers]
    write_csv(psd.astype(float), DATA / 'structures' / 'poreblazer_psd.csv')


def export_kernel(workbook, key, sheet, rows, has_dft, details):
    grid = read_sheet(workbook, sheet)
    numbers = grid.iloc[ISOTHERM_NUMBER_ROW, 1:].dropna().astype(int).tolist()
    n_columns = len(numbers) if has_dft else N_ATOMISTIC
    if numbers[:n_columns] != list(range(1, n_columns + 1)):
        raise ValueError(f'{sheet}: column numbers are not 1..{n_columns}')

    block = grid.iloc[ISOTHERM_FIRST_ROW:ISOTHERM_FIRST_ROW + rows, :n_columns + 1]
    if block.isna().any().any():
        raise ValueError(f'{sheet}: missing values in the isotherm block')
    if len(grid.iloc[ISOTHERM_FIRST_ROW:, 0].dropna()) != rows:
        raise ValueError(f'{sheet}: expected {rows} pressure points')

    ids = [structure_id(n) for n in range(1, N_ATOMISTIC + 1)]
    if has_dft:
        widths = grid.iloc[ISOTHERM_WIDTH_ROW, N_ATOMISTIC + 1:n_columns + 1].astype(float)
        moment1 = details['moment1'].iloc[N_ATOMISTIC:n_columns].astype(float)
        if not np.array_equal(widths.to_numpy(), moment1.to_numpy()):
            raise ValueError(f'{sheet}: DFT widths differ from Details moment1')
        dft_ids = [dft_id(w) for w in widths]
        ids += dft_ids
        #Each DFT column's pore width, which the Details sheet stores as moment1
        write_csv(pd.DataFrame({'id': dft_ids, 'pore_width_A': widths.to_numpy()}),
                  DATA / 'kernels' / key / 'dft_pores.csv')

    block = block.copy().astype(float)
    block.columns = ['pressure'] + ids
    write_csv(block, DATA / 'kernels' / key / 'isotherm.csv')


def slug_check(name):
    if not re.fullmatch(r'[a-z0-9_.\-]+\.csv', name):
        raise ValueError(name)


def export_legacy(workbook):
    for sheet, name in LEGACY_SHEETS.items():
        slug_check(name)
        grid = read_sheet(workbook, sheet).dropna(how='all').dropna(axis=1, how='all')
        write_csv(grid, DATA / 'legacy' / name)


def main():
    workbook = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / 'kernel.xlsx'
    print(f'Exporting {workbook}')
    details = pd.read_excel(workbook, sheet_name='Details', header=1, index_col=1,
                            engine='openpyxl').drop(columns='Unnamed: 0')
    details = details.dropna(how='all')
    psd_sheet = read_sheet(workbook, PSD_SHEET)
    export_structures(workbook, details, psd_sheet)
    for key, (sheet, rows, has_dft) in KERNEL_SHEETS.items():
        export_kernel(workbook, key, sheet, rows, has_dft, details)
    export_legacy(workbook)
    print('Done. Now run: python tools/data_lock.py')


if __name__ == '__main__':
    main()
