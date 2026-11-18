"""
Read the versioned kernel database under data/ (see data/manifest.toml).

The readers return the workbook-shaped tables the app has always used: structures are
numbered 1..N by position (atomistic first, then DFT), so column k of an isotherm table
and row k of the structures table describe the same structure. The stable IDs ("025",
"DFT2D-100") live in the CSV headers and the manifest.
"""
import hashlib
import tomllib
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / 'data'
MANIFEST = DATA_DIR / 'manifest.toml'
LOCK = DATA_DIR / 'versions.lock.json'

#Columns of structures.csv that are bookkeeping, not structural properties
STRUCTURE_META_COLUMNS = ['id', 'rev', 'rev_note', 'poreblazer_label']


def read_csv(relative_path):
    #round_trip parsing reproduces the written floats exactly
    return pd.read_csv(DATA_DIR / relative_path, float_precision='round_trip',
                       dtype={'id': str})


def load_manifest():
    with open(MANIFEST, 'rb') as fh:
        return tomllib.load(fh)


def structure_ids(manifest):
    return read_csv(manifest['structures']['file'])['id'].tolist()


def excluded_positions(manifest):
    """1-based positions of the excluded structures (e.g. [9, 13])."""
    ids = structure_ids(manifest)
    return [ids.index(sid) + 1 for sid in manifest['structures']['excluded']]


def kernel_configs(manifest):
    """
    Map each kernel's display name to its config: the manifest entry plus
    structures (all columns) and structures_model (atomistic columns).
    """
    n_atomistic = len(structure_ids(manifest))
    configs = {}
    for kernel in manifest['kernels']:
        n_columns = len(pd.read_csv(DATA_DIR / kernel['isotherm'], nrows=0).columns) - 1
        configs[kernel['name']] = {
            **kernel,
            'structures': n_columns,
            'structures_model': n_atomistic,
            'pressure_xlim': tuple(kernel['pressure_xlim']),
        }
    return configs


def read_structures(manifest, config, n_structures):
    """
    Structural parameters, one row per structure numbered 1..n_structures (index '#').
    DFT rows carry only moment1 (pore width in Angstrom), as in the workbook.
    """
    atomistic = read_csv(manifest['structures']['file']).drop(columns=STRUCTURE_META_COLUMNS)
    frames = [atomistic]
    if n_structures > len(atomistic):
        dft = read_csv(config['dft_pores'])
        frames.append(pd.DataFrame({'moment1': dft['pore_width_A']}))
    df = pd.concat(frames, ignore_index=True).iloc[:n_structures]
    df.index = pd.RangeIndex(1, n_structures + 1, name='#')
    return df


def read_isotherm(config, n_structures):
    """Column 0 is the pressure grid; columns 1..n_structures are the kernel isotherms."""
    df = read_csv(config['isotherm']).iloc[:, :n_structures + 1]
    df.columns = range(n_structures + 1)
    return df


def read_poreblazer_psd(manifest, n_structures):
    """
    Column 0 is the pore width (Angstrom); columns 1..n_structures are the Poreblazer
    PSDs. DFT structures have no Poreblazer PSD, so their columns are zero.
    """
    df = read_csv(manifest['structures']['poreblazer_psd'])
    n_atomistic = df.shape[1] - 1
    df = df.iloc[:, :n_structures + 1].copy()
    for position in range(n_atomistic + 1, n_structures + 1):
        df[position] = 0.0
    df.columns = range(n_structures + 1)
    return df


def file_sha256(relative_path):
    #Line endings are normalized so a Windows checkout hashes the same
    content = (DATA_DIR / relative_path).read_bytes().replace(b'\r\n', b'\n')
    return hashlib.sha256(content).hexdigest()


def versioned_files(manifest):
    """
    Map 'name@version' -> the data files that version covers, for the structure
    library and every kernel.
    """
    structures = manifest['structures']
    entries = {f"structures@{structures['version']}":
               [structures['file'], structures['poreblazer_psd']]}
    for kernel in manifest['kernels']:
        files = [kernel['isotherm']] + ([kernel['dft_pores']] if 'dft_pores' in kernel else [])
        entries[f"{kernel['key']}@{kernel['version']}"] = files
    return entries
