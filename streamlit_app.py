import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.ticker as ticker
import pandas as pd
from scipy.optimize import nnls

#some old versions of scipy do not have the cumulative trapezoid function
try:
    from scipy.integrate import cumulative_trapezoid
except ImportError:
      st.warning("Old SciPy detected: cumulative PSD unavailable.")


#Constants
ANNEALING_TIME_S = 2e-9 #2 nanoseconds in seconds
KB_EV_PER_K = 8.617e-5 #eV/K Boltzmann constant
ACTIVATION_ENERGY_EV = 6.0 #eV
#1 cm3 of liquid N2 at 77 K holds 646.5 cm3 STP of gas (22414 cm3 STP/mol / 34.67 cm3/mol
#liquid molar volume). Used to derive the DFT structures' extensive properties from their
#isotherm normalization; only the N2 77 K kernel has DFT structures.
CM3_STP_PER_CM3_LIQUID_N2 = 646.5

#Excel file holding every kernel (isotherms, structural details, PSDs)
EXCEL_DATABASE = r'kernel.xlsx'

#Kernel definitions. Each entry describes one simulated adsorption kernel in EXCEL_DATABASE.
# - isotherm_sheet / isotherm_rows: where to read the isotherm matrix and how many pressure points.
# - structures: total number of structure columns in the kernel.
# - structures_model: how many of those are atomistic. The remaining (structures - structures_model)
#   are DFT structures appended at the end. They have no .xyz/render/TEM assets and no
#   counterpart in DFT-free kernels (e.g. CO2), so they cannot be transferred during conversion.
#The first structures_model atomistic structures share the same indices across all kernels.
KERNELS = {
    "N₂ at 77 K": {
        "isotherm_sheet": "N2 77 K 1CLJ_2D-NLDFT",
        "isotherm_rows": 93,
        "structures": 109,
        "structures_model": 78,
        "pressure_unit": "P/P₀",
        "pressure_xlim": (1e-8, 1.4),
    },
    "CO₂ at 298.15 K": {
        "isotherm_sheet": "CO2 298 K",
        "isotherm_rows": 50,
        "structures": 78,
        "structures_model": 78,
        "pressure_unit": "bar",
        "pressure_xlim": (1e-7, 2),
    },
    "CO₂ at 273 K": {
        "isotherm_sheet": "CO2 273 K",
        "isotherm_rows": 50,
        "structures": 78,
        "structures_model": 78,
        "pressure_unit": "bar",
        "pressure_xlim": (1e-7, 2),
    },
    "H₂ at 77 K": {
        "isotherm_sheet": "H2 77 K",
        "isotherm_rows": 35,
        "structures": 78,
        "structures_model": 78,
        "pressure_unit": "bar",
        "pressure_xlim": (5e-3, 200),
    },
    #The Ar sheet also carries 11 trailing columns (Excel CB-CL) built from a carbon black
    #t-curve plus a Kelvin filling step. Their Kelvin constant is ~4.5x too small, so every
    #width >= 8 A condenses in the last pressure step, and their widths (7-40 A) do not match
    #the Details DFT rows (10-60 A). They are excluded until regenerated.
    "Ar at 87 K": {
        "isotherm_sheet": "Ar 87 K 1CLJ",
        "isotherm_rows": 64,
        "structures": 78,
        "structures_model": 78,
        "pressure_unit": "P/P₀",
        "pressure_xlim": (1e-8, 1.4),
    },
}


def find_range(contents):
    """
    Takes the file content as input
    Returns the line location of the adsorption branches
    """
    string_find = b'"No."\t"Pe/kPa"\t"P0/kPa"\t"Vd/ml"\t"V/ml(STP) g-1"'
    string_find2 = b'"No."\t"Pe/kPa"\t"P0/kPa"\t"Vd/ml"\t"V/ml(STP)\x81Eg-1"'
    #print(contents[35] == b'"No."\t"Pe/kPa"\t"P0/kPa"\t"Vd/ml"\t"V/ml(STP)\x81Eg-1"')
    start_adsorption, start_desorption = \
        [i for i,x in enumerate(contents) if ((x==string_find) or (x==string_find2))]
    string_find = b'0\t0\t0\t0\t0'
    end_adsorption, end_desorption = [i for i,x in enumerate(contents) if x==string_find]    
    return (start_adsorption, end_adsorption, start_desorption, end_desorption)

def read_branch(contents, branch) -> "np.ndarray":
    """
    Reads Belsorp exported files
    Returns the content of a certain adsorption or desorption branch.
    """
    start_adsorption, end_adsorption, start_desorption, end_desorption = find_range(contents)
    if branch == 'adsorption':
        skip_header = start_adsorption+1
        max_rows = end_adsorption-start_adsorption-1
    elif branch == 'desorption':
        skip_header = start_desorption+1
        max_rows = end_desorption-start_desorption-1
    isotherm = np.genfromtxt(contents,
                             delimiter='\t',
                             skip_header=skip_header,
                             max_rows=max_rows,
                             encoding='shift-jis',
                             usecols=(1,2,4))
    #return np.column_stack((isotherm[:,0], isotherm[:,1], isotherm[:,2]))
    return isotherm


def parse_pasted_isotherm(text) -> "tuple[np.ndarray, int]":
    """
    Parses an isotherm pasted as text, typically copied from two Excel columns.
    Columns may be separated by tabs, semicolons, commas or spaces. Decimal commas
    (e.g. "0,015") are accepted when the columns are not comma-separated.
    Lines that are not two numbers (headers, blank lines) are skipped.
    Returns the (n, 2) array and the number of skipped non-blank lines.
    """
    rows = []
    skipped = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if '\t' in line:
            fields = line.split('\t')
        elif ';' in line:
            fields = line.split(';')
        elif len(line.split()) >= 2:
            fields = line.split()
        else:
            fields = line.split(',')
        fields = [f.strip() for f in fields if f.strip()]
        try:
            if len(fields) < 2:
                raise ValueError
            rows.append([float(f.replace(',', '.')) for f in fields[:2]])
        except ValueError:
            skipped += 1
    return np.array(rows, dtype=float).reshape(-1, 2), skipped


def parse_pasted_column(text) -> "tuple[np.ndarray, int]":
    """
    Parses a single column of numbers pasted as text, one value per line.
    Decimal commas are accepted. Lines that are not a number (headers) are skipped.
    Returns the values and the number of skipped non-blank lines.
    """
    values = []
    skipped = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(line.replace(',', '.')))
        except ValueError:
            skipped += 1
    return np.array(values, dtype=float), skipped


def plot_top_structures(solution, offset=0):
    """                                                                                                                               
    Plot a 2x3 grid of the top 3 contributing kernel structures, ranked by solution weight.
                                                                                                                                    
    Each column shows one structure: the top render (PNG) in row 0 and the simulated TEM                                              
    image (TIF) with a 2 nm scale bar in row 1. DFT-only structures (index > structures_model)                                        
    display a text placeholder instead of images.

    Parameters
    ----------
    solution : np.ndarray
        NNLS solution vector of length `structures`, one weight per kernel structure.
    offset : int, optional
        Skip the top `offset` contributors before selecting 3 to display.
        0 (default) shows ranks 1-3; 3 shows ranks 4-6.
    """

    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    for i in range(3):
        structure_render = np.argsort(solution)[-1 - i - offset] + 1
        if structure_render <= structures_model:
            render = img.imread(f'rendered structures/{structure_render:03d}.png')
            ax[0,i].imshow(render)
            ax[0,i].set_axis_off()
        else:
            ax[0,i].set_xticks([])
            ax[0,i].set_yticks([])
            ax[0,i].text(0.14, 0.5, f"Structure modeled via DFT\nPore size = {df_structures['moment1'][structure_render]/10:.1f} nm")

        ax[0,i].title.set_text(f'Structure {structure_render:02d} ({solution[structure_render-1]/sum(solution)*100:.1f}%)')

        if structure_render <= structures_model:
            render = img.imread(f'simulated TEM/{structure_render:02d}.tif')
            ax[1,i].imshow(render, cmap='gist_gray')
            with open(f'structures/{structure_render:03d}.xyz') as fh:
                next(fh)
                lattice_size = float(next(fh).split(" ")[0])
            bar_length_pixels = 20 * render.shape[0] / lattice_size
            pixels_sim_tem = render.shape[0]
            ax[1,i].text(50, pixels_sim_tem*0.91, "2 nm", color='white', fontsize=15, fontweight='bold')
            ax[1,i].plot([50, 50+bar_length_pixels], [pixels_sim_tem*0.95, pixels_sim_tem*0.95], '-', lw=5, color='white')
            ax[1,i].set_axis_off()
        else:
            ax[1,i].set_xticks([])
            ax[1,i].set_yticks([])
            ax[1,i].text(0.38, 0.5, "No image")
    st.pyplot(fig)


def calculate_isotherm(solution):
    # This function sums the contributions of every kernel structure
    # in order to calculate the predicted isotherm.
    isotherm = np.zeros(np_pressure_gcmc.size) #create an empty vector
    for i in range(structures):
        isotherm = isotherm + solution[i] * np.array(np_isotherm[:,i])
    return isotherm


def PascalTriangle(n):
    # This calculates a Pascal triangle that will be used for smoothing
    # https://www.askpython.com/python/examples/pascals-triangle-using-python
    # https://danielmuellerkomorowska.com/2020/06/02/smoothing-data-by-rolling-average-with-numpy/
    trow = [1]
    y = [0]
    for x in range(n):
        trow=[left+right for left,right in zip(trow+y, y+trow)]
    return trow


def load_kernel(config, n_structures):
    """
    Load a kernel's structural parameters and isotherm matrix from EXCEL_DATABASE.

    Parameters
    ----------
    config : dict
        One KERNELS entry (isotherm sheet name, row count, structure counts).
    n_structures : int
        Number of structure columns to load. Pass structures_model to drop the trailing
        DFT structures, yielding a DFT-free kernel whose fit can be converted to any
        other kernel without losing weight.

    Returns
    -------
    df_structures : pandas.DataFrame
        Per-structure parameters, one row per structure (n_structures rows). For DFT
        structures the Details sheet holds only moment1 (pore width in Angstrom); their
        extensive properties (pore volume, surface area) are derived here from the
        isotherm normalization so they share the isotherms' per-gram basis.
    df_isotherm : pandas.DataFrame
        Column 0 is the pressure grid; columns 1..n_structures are the kernel isotherms.
    """
    df_structures = pd.read_excel(EXCEL_DATABASE,
                    sheet_name='Details',
                    header=1,
                    nrows=n_structures,
                    index_col=1,
                    engine='openpyxl')

    df_isotherm = pd.read_excel(EXCEL_DATABASE,
                    sheet_name=config['isotherm_sheet'],
                    header=None,
                    skiprows=8,
                    nrows=config['isotherm_rows'],
                    usecols=range(0, n_structures + 1),
                    engine='openpyxl')

    #Structures 9 and 13 had low density and never formed a solid framework.
    #Zeroing their isotherms removes them from the regression.
    df_isotherm[9] = 0
    df_isotherm[13] = 0

    #DFT structures are normalized so each isotherm column saturates at 1000 cm3 STP/g,
    #which is not the per-gram basis of any real wall material. Derive their extensive
    #properties from that same normalization so weight*property products are consistent:
    #pore volume V = saturation/646.5 (complete filling with liquid N2) and slit-geometry
    #surface area A = 2V/w, with w = moment1 (pore width in Angstrom).
    n_model = config['structures_model']
    if n_structures > n_model:
        dft_rows = df_structures.index[n_model:]
        dft_saturation = df_isotherm.iloc[-1, n_model + 1:n_structures + 1].to_numpy(dtype=float)
        dft_volume = dft_saturation / CM3_STP_PER_CM3_LIQUID_N2
        dft_width = df_structures['moment1'].iloc[n_model:].to_numpy(dtype=float)
        df_structures.loc[dft_rows, 'Helium volume in cm^3/g'] = dft_volume
        df_structures.loc[dft_rows, 'Geometric (point accessible) volume in cm^3/g'] = dft_volume
        #2e4 converts (cm3/g) / Angstrom to m2/g
        df_structures.loc[dft_rows, 'Total surface area m^2/g'] = 2e4 * dft_volume / dft_width

    return df_structures, df_isotherm


def build_second_difference(n_bins):
    """
    Discrete second-difference (curvature) operator, shape (n_bins-2, n_bins).

    Row i holds the stencil [1, -2, 1] at columns i, i+1, i+2. The Poreblazer
    width grid is uniform (0.25 A spacing) so the plain stencil measures true
    curvature; a non-uniform grid would need spacing-aware coefficients.
    """
    operator = np.zeros((n_bins - 2, n_bins))
    for i in range(n_bins - 2):
        operator[i, i] = 1.0
        operator[i, i + 1] = -2.0
        operator[i, i + 2] = 1.0
    return operator


def fit_weights(kernel_matrix, target, penalty_matrix, lam):
    """
    Non-negative least squares fit of the kernel to the experimental isotherm,
    optionally regularized so the combined PSD is smooth.

    Solves  min_f ||K f - N||^2 + lam_eff ||(L P) f||^2  s.t. f >= 0  by
    appending sqrt(lam_eff) * (L P) rows to the design matrix and zeros to the
    target, then calling plain NNLS (Tikhonov regularization in general form).

    Scaling: lam is dimensionless. The penalty block is rescaled by
    ||K||_F / ||L P||_F, so lam = 1 gives the isotherm and smoothness blocks
    equal Frobenius norm. Cite lam together with this rule to reproduce a fit.

    Parameters
    ----------
    kernel_matrix : (M, S) kernel isotherms K.
    target : (M,) experimental isotherm N interpolated to the kernel pressures.
    penalty_matrix : (W-2, S) precomputed L @ P, where P is the helium-volume
        scaled PSD basis on the shared width grid.
    lam : float, regularization strength. 0 reduces exactly to plain NNLS.

    Returns
    -------
    solution : (S,) non-negative weights f.
    fit_residual : float, ||K f - N||. Comparable across lam values.
    penalty_value : float, ||(L P) f||, curvature of the combined PSD (unscaled).
    """
    kernel_matrix = np.asarray(kernel_matrix, dtype=float)
    penalty_matrix = np.asarray(penalty_matrix, dtype=float)
    if lam > 0:
        scale = np.linalg.norm(kernel_matrix) / np.linalg.norm(penalty_matrix)
        design = np.vstack([kernel_matrix, np.sqrt(lam) * scale * penalty_matrix])
        rhs = np.concatenate([target, np.zeros(penalty_matrix.shape[0])])
        solution, _ = nnls(design, rhs)
        fit_residual = np.linalg.norm(kernel_matrix @ solution - target)
    else:
        solution, fit_residual = nnls(kernel_matrix, target)
    penalty_value = np.linalg.norm(penalty_matrix @ solution)
    return solution, fit_residual, penalty_value


def lambda_sweep(kernel_matrix, target, penalty_matrix, lambdas):
    """Fit once per lambda; returns (fit_residuals, penalty_values) for the L-curve."""
    fit_residuals = np.empty(len(lambdas))
    penalty_values = np.empty(len(lambdas))
    for i, lam in enumerate(lambdas):
        _, fit_residuals[i], penalty_values[i] = \
            fit_weights(kernel_matrix, target, penalty_matrix, lam)
    return fit_residuals, penalty_values


def lcurve_corner(fit_residuals, penalty_values):
    """
    Index of the L-curve corner: the sweep point farthest from the straight
    line joining the two endpoints in normalized log-log space, counting only
    points that bulge toward the origin (the convex side, where the corner of
    an L lives).

    The signed distance matters: penalty-free structures (DFT) let the penalty
    keep dropping at large lambda while the fit residual explodes, which bends
    the tail of the curve away from the origin. An unsigned distance would pick
    that tail as the "corner".
    """
    def normalize01(values):
        values = np.log10(np.maximum(values, 1e-30))
        span = values.max() - values.min()
        return (values - values.min()) / span if span > 0 else np.zeros_like(values)
    x = normalize01(fit_residuals)
    y = normalize01(penalty_values)
    dx, dy = x[-1] - x[0], y[-1] - y[0]
    chord = np.hypot(dx, dy)
    if chord == 0:
        return 0
    #positive = below the chord, toward the origin (true corner side);
    #negative = the concave side, never a corner
    distance = (dy * (x - x[0]) - dx * (y - y[0])) / chord
    return int(np.argmax(distance))





st.title('3D-VIS Isotherm Analysis')

multi = '''This app predicts the 3D nanostructure of a porous carbon from an experimental gas adsorption isotherm,                    
  by fitting it as a linear combination of pre-calculated atomistic kernel isotherms (Non-Negative Least Squares).                      
  Results include the contributing structures, morphological statistics, and pore size distribution.
                                                                                                                                        
  **Supported measurements:** N₂ at 77 K, Ar at 87 K, CO₂ at 298.15 K, CO₂ at 273 K, and H₂ at 77 K (high pressure, up to 120 bar).

  **Supported file formats:** tab-separated values (P/P₀ vs cm³/g) or Belsorp `.DAT` export files. Data can also be pasted directly, e.g. copied from Excel.

  **Workflow:** Select kernel → Upload isotherm → Clean data → Analyze → Export PSD.

  ---

  Based on the paper:

  *F. Vallejos-Burgos et al.* **3D nanostructure prediction of porous carbons via gas adsorption**, Carbon, Volume 215, 2023, 118431.

  [Read the paper](https://doi.org/10.1016/j.carbon.2023.118431) · [Download
  citation](https://raw.githubusercontent.com/nandobike/3d-vis/main/S0008622323006760.bib)

  Questions? Suggestions? Complaints? Shoot us an email: fvb@vallejos.cl



'''
st.markdown(multi)

st.image('summary.jpg', caption='Summary of the 3D-VIS method')




st.divider()
st.header('Kernel Selection')
st.markdown('Select a kernel. The kernel is the set of the simulated adsorption isotherms. '
            'It must match the experimental conditions of the measurement to be uploaded next.')

kernel_radio = st.radio(
    "Select Simulated Kernel",
    list(KERNELS.keys()),
    index=0
)
kernel_config = KERNELS[kernel_radio]

#DFT structures can't be transferred to DFT-free kernels (e.g. CO2). Let the user exclude
#them so the fit uses only the atomistic structures shared by every kernel, which makes the
#Adsorbate Conversion below lossless.
has_dft = kernel_config['structures'] != kernel_config['structures_model']
include_dft = st.checkbox(
    "Include DFT structures in the fit",
    value=True,
    disabled=not has_dft,
    help="DFT structures model larger pores but have no counterpart in DFT-free "
         "kernels such as CO₂. Uncheck for a fit that can be converted to any adsorbate "
         "without losing weight. Disabled when the selected kernel has no DFT structures."
)

#Structures available in the kernel (atomistic only when DFT is excluded)
structures = kernel_config['structures'] if (has_dft and include_dft) else kernel_config['structures_model']

#Structures that are calculated with atomistic model (the rest are DFT)
structures_model = kernel_config['structures_model']

#Load structural parameters and calculated adsorption isotherms
df_structures, df_isotherm = load_kernel(kernel_config, structures)

#By default exclude DFT structures with pores smaller than 5 nm: the atomistic GCMC
#structures (cells up to ~5 nm) already model that range realistically, and the point of
#the method is to "see" the real structure, which too much DFT weight prevents.
DFT_MIN_PORE_NM_DEFAULT = 5.0
dft_min_pore_nm = DFT_MIN_PORE_NM_DEFAULT
if has_dft and include_dft:
    with st.expander("DFT pore-size cutoff"):
        st.markdown('DFT structures with pores smaller than this cutoff are excluded from '
                    'the fit. The atomistic structures already cover pores up to about '
                    '5 nm (their cell size), so small-pore DFT structures only compete '
                    'with them and hide the real structure. Set to 0 to keep every DFT '
                    'structure.')
        dft_min_pore_nm = st.slider("Exclude DFT structures with pores smaller than (nm):",
                                    0.0, 42.0, DFT_MIN_PORE_NM_DEFAULT, 0.5)
        dft_excluded = [s for s in range(structures_model + 1, structures + 1)
                        if df_structures['moment1'][s] < dft_min_pore_nm * 10]
        #Zeroing their isotherms removes them from the regression, same mechanism as the
        #unformed structures 9 and 13.
        for s in dft_excluded:
            df_isotherm[s] = 0
        st.caption(f"{len(dft_excluded)} of {structures - structures_model} DFT structures "
                   f"excluded; {structures - structures_model - len(dft_excluded)} remain "
                   f"in the fit.")


#Read pore size distributions and load into dataframe
df_PSD_pb = pd.read_excel(EXCEL_DATABASE,
                sheet_name='Poreblazer PSDs_2', #_2 for ultranarrow pores ~1 A
                header=None,
                skiprows=6,
                nrows=210,
                usecols=range(0,structures+1),
                engine='openpyxl')
#Convert pore size distribution data to a numpy array
np_PSD_pb = np.array(df_PSD_pb)[:,1:]

#PSD basis for the smoothness regularization. It must match the PSD the app reports
#(PSD_solution below), which scales each structure's Poreblazer PSD by its helium volume,
#so the same scaling is applied here.
np_psd_basis = np_PSD_pb.astype(float) * np.array(df_structures['Helium volume in cm^3/g'])
#Structures 9 and 13 have zeroed isotherms (never formed a solid framework). Zero their
#PSDs too, otherwise the penalty could assign them weight purely to smooth the combined
#PSD without affecting the isotherm fit.
np_psd_basis[:, 8] = 0
np_psd_basis[:, 12] = 0
#DFT structures have all-zero columns in the Poreblazer sheet (their PSD is a
#single-size spike handled separately), so the smoothness penalty cannot see them and
#some weight may drift toward them at large lambda.
np_penalty_matrix = build_second_difference(np_psd_basis.shape[0]) @ np_psd_basis

#Create a boolean that means that there is DFT isotherms. Only True if structures_model not equals to structures
dft_present = (structures != structures_model)


st.divider()
st.header('Isotherm Data Load')
_pressure_col_desc = "relative pressure (P/P₀)" if kernel_config['pressure_unit'] == "P/P₀" else "pressure in bar"
st.markdown(f'Upload your isotherm as a text file, or paste it directly from Excel, either as two columns together or as the pressure and adsorbed amount columns in separate fields. The data must only contain datapoints in ascending pressure order. Two columns separated by tabs, first for {_pressure_col_desc}, second for adsorbed amount in cc STP/g. See an example [here](https://raw.githubusercontent.com/nandobike/3d-vis/main/examples/a20_lao.tsv)')

input_method = st.radio("Isotherm input",
                        ["Upload file", "Paste two columns", "Paste pressure and amount separately"],
                        horizontal=True)

#load experimental isotherm
#It must be a tab-separated file with two columns.
#First column is relative pressure and second column adsorbed volume in units cc STP/g
file = None
exp_iso = None #Set here only for pasted data; files are read below
skipped_lines = 0
EXAMPLE_ISOTHERM_FILE = "examples/a20_lao.tsv"
if input_method != "Upload file": #Pre-fill the paste fields with the example isotherm
    with open(EXAMPLE_ISOTHERM_FILE) as f:
        example_rows = [line.strip().split('\t') for line in f if line.strip()]
if input_method == "Upload file":
    file = st.file_uploader("Upload isotherm file")
elif input_method == "Paste two columns":
    pasted_text = st.text_area(
        "Paste isotherm data",
        value="\n".join("\t".join(row) for row in example_rows),
        height=250,
        placeholder="0.0001\t120.5\n0.0005\t180.2\n0.001\t210.7\n...",
        help="Two columns: pressure and adsorbed amount (cc STP/g). Copy both columns from "
             "Excel and paste here. Header rows and blank lines are ignored.")
    if pasted_text.strip():
        exp_iso, skipped_lines = parse_pasted_isotherm(pasted_text)
else:
    col_pressure, col_amount = st.columns(2)
    with col_pressure:
        pressure_text = st.text_area(
            f"Pressure ({kernel_config['pressure_unit']})",
            value="\n".join(row[0] for row in example_rows),
            height=250,
            placeholder="0.0001\n0.0005\n0.001\n...",
            help="One value per line, e.g. a column copied from Excel. "
                 "Header rows and blank lines are ignored.")
    with col_amount:
        amount_text = st.text_area(
            "Adsorbed amount (cc STP/g)",
            value="\n".join(row[1] for row in example_rows),
            height=250,
            placeholder="120.5\n180.2\n210.7\n...",
            help="One value per line, in the same order as the pressures. "
                 "Header rows and blank lines are ignored.")
    if pressure_text.strip() or amount_text.strip():
        pressures, skipped_pressure = parse_pasted_column(pressure_text)
        amounts, skipped_amount = parse_pasted_column(amount_text)
        if len(pressures) != len(amounts):
            st.error(f"Read {len(pressures)} pressure values but {len(amounts)} adsorbed "
                     f"amounts. Both fields must have the same number of values.")
            st.stop()
        exp_iso = np.column_stack((pressures, amounts))
        skipped_lines = skipped_pressure + skipped_amount

if exp_iso is not None: #Pasted data
    if exp_iso.shape[0] < 2:
        st.error("Could not read at least two data points from the pasted text. "
                 "Paste numeric values for pressure and adsorbed amount.")
        st.stop()
    st.write(f"Read {exp_iso.shape[0]} data points from the pasted text.")
    if skipped_lines:
        st.caption(f"{skipped_lines} non-numeric line(s) skipped (e.g. headers).")
elif file is None: #Nothing provided: load the example
    file = EXAMPLE_ISOTHERM_FILE
    st.write(f"No data was provided. Loading a default isotherm file: {file}")
    exp_iso = np.genfromtxt(file, delimiter="\t") #Load example. Originally a20_lao.tsv
else: #Read uploaded file
    st.write(f'A file was uploaded: {file.name} as {file.type}')
    #read first line and remove whitespace
    first_line = next(file).strip()
    if first_line == b'====================': #If a Belsorp file is loaded
        st.write('This file seems to be Belsorp format, will attempt load.')
        contents = []
        for line in file:
            #print(line)
            #contents.append(line.decode('shift-jis').rstrip()) #this works new version
            contents.append(line.rstrip())
        col1, col2 = st.columns(2)
        with col1:
            force_p0 = st.checkbox("Force P₀", value=False,
                                   help="Saturation pressure will be forced to be a user-defined " \
                                     "value instead of reading it from the file.")

        with col2:
            p0_forced_value = st.number_input("P₀ forced", value=100.0, min_value=0.0,
                                        help="A value of 100 will convert kPa to bar as required by kernel.",
                                        disabled=not(force_p0)
                                        )
            #st.write("P0 is", p0_forced_value)
        
        exp_iso = read_branch(contents, 'adsorption')
        if force_p0:
            exp_iso = np.column_stack((exp_iso[:,0]/p0_forced_value, exp_iso[:,2]))
        else:
            exp_iso = np.column_stack((exp_iso[:,0]/exp_iso[:,1], exp_iso[:,2]))
        #st.write(exp_iso) #Debug
    else: #Now the standard file
        file.seek(0) #Rewind: the format check above consumed the first line
        exp_iso = np.genfromtxt(file, delimiter="\t") #load isotherm file into numpy array

#Warn (without stopping) when the data is not ascending, e.g. a desorption branch was
#included. The interpolation to the kernel grid assumes increasing pressures.
#Equal consecutive adsorbed amounts (a plateau) are not flagged, only decreases.
for column, name, not_ascending in ((0, "Pressures", np.diff(exp_iso[:,0]) <= 0),
                                    (1, "Adsorbed amounts", np.diff(exp_iso[:,1]) < 0)):
    if np.any(not_ascending):
        first_bad = int(np.argmax(not_ascending)) + 2 #1-based number of the offending point
        st.warning(f"{name} are not in ascending order ({int(np.sum(not_ascending))} "
                   f"point(s), first at point {first_bad}: "
                   f"{exp_iso[first_bad-2, column]:g} → {exp_iso[first_bad-1, column]:g}). "
                   f"Only the adsorption branch should be used, otherwise the "
                   f"interpolation to the kernel may be wrong.")


st.divider()
st.header('Data Cleaning and Validation')
st.write('Usually it is necessary to remove a few experimental points from the very low pressures since they are very inaccurate. Look at the error in the fitted isotherm plot in the Results section to know how many to remove.')
#Remove some initial experimental points where the experimental data is usually flawed
#points_to_remove = 13 #for a20_lao
points_to_remove = st.slider("Use the slider below to remove initial points from the isotherm:",
                             0,
                             np.shape(exp_iso)[0],
                             0)

st.write(f'Now the points from {points_to_remove} to {np.shape(exp_iso)[0]} will be used in the calculation')

x_axis_scale = st.radio(
    "Select x-axis scaling for the plot below",
    ["Logarithmic", "Linear"])


np_isotherm = np.array(df_isotherm)[points_to_remove:,1:]
np_pressure_gcmc = np.array(df_isotherm)[points_to_remove:,0]

exp_iso_interp = np.interp(np_pressure_gcmc, exp_iso[:,0], exp_iso[:,1]) #interpolate isotherm to points of the kernel

fig, ax = plt.subplots(figsize=(7,4))
ax.plot(exp_iso[:,0], exp_iso[:,1],label='Experimental', marker='o', linestyle='none')
ax.set_xlabel("Relative pressure P/P$_0$" if kernel_config['pressure_unit'] == "P/P₀" else "Pressure (bar)")
ax.set_ylabel("Adsorbed amount (cm$^3$/g)")
ax.set_ylim(bottom=0)  # adjust the bottom leaving top unchanged
ax.plot(np_pressure_gcmc, exp_iso_interp,
         label='Experimental interpolated',
         marker='x',
         markersize=4,
         linestyle='none')
if x_axis_scale == 'Logarithmic':
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))
    ax.set_xlim(left=kernel_config['pressure_xlim'][0], right=kernel_config['pressure_xlim'][1])

ax.set_title('Experimental Isotherm and Interpolation to Kernel')
ax.legend()
ax.grid(color='aliceblue')
#plt.show()
st.pyplot(fig)




st.divider()
st.header('Analysis Results')
st.write('Here the results of fitting the experimental isotherm with the kernel isotherms. Look at the error plot and go back to remove highly inaccurate points if necessary.')

#Widget state keys let the "Use suggested λ" button in the diagnostics expander turn
#regularization on and set the strength programmatically.
if 'regularize' not in st.session_state:
    st.session_state['regularize'] = False
if 'log_lambda' not in st.session_state:
    st.session_state['log_lambda'] = 0.0

regularize = st.checkbox(
    "Apply PSD-smoothness regularization",
    key='regularize',
    help="Penalizes the curvature of the combined pore size distribution so that "
         "structures with nearly identical isotherms cannot trade weight freely under "
         "noise and produce spiky PSDs. Unchecked reproduces the plain NNLS fit.")

if regularize:
    log_lambda = st.slider(
        "Regularization strength log₁₀(λ)",
        min_value=-4.0, max_value=4.0, step=0.01,
        key='log_lambda',
        help="λ is dimensionless: the penalty rows are rescaled so that λ = 1 gives the "
             "isotherm-fit and PSD-smoothness terms equal weight (equal Frobenius norms). "
             "Use the L-curve in the diagnostics below to pick a value.")
    lambda_reg = 10.0 ** log_lambda
    if dft_present:
        st.caption("Note: DFT structures are not covered by the smoothness penalty (their "
                   "PSD is a single-size spike handled separately), so a large λ can shift "
                   "weight toward them. Watch the DFT part in Morphological Information, or "
                   "uncheck \"Include DFT structures\" above.")
else:
    lambda_reg = 0.0

with st.expander("Regularization diagnostics (conditioning and L-curve)"):
    _nonzero_cols = np.linalg.norm(np_isotherm.astype(float), axis=0) > 0
    _cond_K = np.linalg.cond(np_isotherm[:, _nonzero_cols].astype(float))
    st.text(f"Condition number of kernel K (nonzero columns) = {_cond_K:.4g}\n"
            f"Condition number of KᵀK = {_cond_K**2:.4g}")
    if st.checkbox("Run λ sweep and plot the L-curve"):
        sweep_lambdas = np.logspace(-4, 4, 25)
        sweep_residuals, sweep_penalties = lambda_sweep(np_isotherm, exp_iso_interp,
                                                        np_penalty_matrix, sweep_lambdas)
        corner = lcurve_corner(sweep_residuals, sweep_penalties)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.loglog(sweep_residuals, sweep_penalties, marker='o', markersize=4, color='tab:blue')
        ax.loglog(sweep_residuals[corner], sweep_penalties[corner], marker='*', markersize=15,
                  linestyle='none', color='tab:red',
                  label=f'Corner: λ = {sweep_lambdas[corner]:.3g}')
        for i in range(0, len(sweep_lambdas), 6):
            ax.annotate(f'λ={sweep_lambdas[i]:.1g}', (sweep_residuals[i], sweep_penalties[i]),
                        fontsize=7, textcoords='offset points', xytext=(5, 5))
        _, current_res, current_pen = fit_weights(np_isotherm, exp_iso_interp,
                                                  np_penalty_matrix, lambda_reg)
        current_label = (f'Selected λ = {lambda_reg:.3g}' if lambda_reg > 0
                         else 'Selected λ = 0 (regularization off)')
        ax.loglog(current_res, current_pen, marker='D', markersize=9, linestyle='none',
                  markerfacecolor='none', markeredgewidth=2, color='tab:green',
                  label=current_label)
        ax.set_xlabel('Isotherm fit residual (cm³/g)')
        ax.set_ylabel('PSD curvature penalty')
        ax.set_title('L-curve: pick λ near the corner')
        ax.legend()
        ax.grid(color='aliceblue')
        st.pyplot(fig)
        st.text(f"Suggested λ at the corner of the L-curve = {sweep_lambdas[corner]:.3g} "
                f"(log₁₀ λ = {np.log10(sweep_lambdas[corner]):.2f})")

        def use_suggested_lambda(log_lambda_value):
            st.session_state['regularize'] = True
            st.session_state['log_lambda'] = log_lambda_value

        st.button("Use suggested λ as the regularization strength",
                  on_click=use_suggested_lambda,
                  args=(round(float(np.log10(sweep_lambdas[corner])), 2),))

#Use non-negative least squares to find the coefficients that fit the experimental isotherm
#from the kernel isotherms, optionally with the PSD-smoothness penalty (lambda_reg = 0
#reduces exactly to plain nnls(np_isotherm, exp_iso_interp)).
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html
solution, residual, penalty_value = fit_weights(np_isotherm, exp_iso_interp,
                                                np_penalty_matrix, lambda_reg)





# Plot experimental datapoints and show the fit
x_axis_scale = st.radio(
    "Select x-axis scaling for the plots below",
    ["Logarithmic", "Linear"],
    key='log fit')
log_scale_plot = (x_axis_scale == "Logarithmic") #use True if you want to plot using logarithmic scale in x

fig, ax = plt.subplots(2, gridspec_kw={'height_ratios': [1, 3]}, dpi=120) #, figsize=(3,3)

# Top plot for error
ax[0].set_title('Experimental Data and Fitted Isotherm')
ax[0].plot(np_pressure_gcmc, exp_iso_interp-calculate_isotherm(solution), marker='o', linestyle='solid', color='tab:orange')
if log_scale_plot:
    ax[0].set_xscale('log')
    ax[0].xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))

ax[0].set_ylabel("Error (cm$^3$/g)")
ax[0].grid(color='aliceblue')


if log_scale_plot:
    ax[0].set_xlim(left=1e-8, right=1.4)
else:
    ax[0].set_xlim(left=-0.02, right=1)

ax[0].axes.get_xaxis().set_ticks([])

# Bottom plot of isotherm and fitted isotherm
ax[1].plot(exp_iso[:,0], exp_iso[:,1],
           label='Experimental',
           marker='o',
           linestyle='none',
           color='tab:orange')
ax[1].plot(np_pressure_gcmc, calculate_isotherm(solution),
           label='Solution',
           linestyle='solid',
           color='black')
if log_scale_plot:
    ax[1].set_xscale('log')
    ax[1].xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))

ax[1].set_xlabel("Relative pressure P/P$_0$")
ax[1].set_ylabel("Adsorbed amount (cm$^3$/g)")
ax[1].legend()
ax[1].grid(color='aliceblue')

ax[1].set_ylim(bottom=0)
if log_scale_plot:
    ax[1].set_xlim(left=1e-8, right=1.4)
else:
    ax[1].set_xlim(left=-0.02, right=1)

st.pyplot(fig)

st.text(f"Residual total= {residual:.3f} cc/g") #norm of residuals = sqrt of sum (error^2)
st.text(f"Residual per point = {residual/np_pressure_gcmc.size:.3f} cc/g") #norm of residuals = sqrt of sum (error^2)
if lambda_reg > 0:
    st.text(f"PSD curvature penalty = {penalty_value:.4g} (λ = {lambda_reg:.3g})")

debug = st.checkbox("Check this box to show contributions of structures to full isotherm. Usually for debugging purposes.")

if debug:
    st.write('The plot below is usually for debugging purposes and show the contributions to the isotherm of the different structures.')
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(np_pressure_gcmc, calculate_isotherm(solution),
           label='Solution',
           linestyle='solid',
           color='black',
           linewidth=3)
    #ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(bottom=1)

    top_n = 15
    contribution_string = ""
    for i in range(top_n):
        struct = np.argsort(solution)[::-1][i]
        if struct+1 > structures_model:
            dft_deco = ' DFT'
            linestyle = 'dashed'
            alpha=0.7
        else:
            dft_deco = ''
            linestyle = 'solid'
            alpha=0.9
        contribution_string = f"S #{struct+1} {dft_deco}"

        ax.plot(np_pressure_gcmc, solution[struct] * np.array(np_isotherm[:,struct]),
                linestyle=linestyle, label=contribution_string,
                alpha=alpha)
    ax.legend(prop={'size': 6}, loc='lower right')
    ax.set_xlabel("Relative pressure P/P$_0$" if kernel_config['pressure_unit'] == "P/P₀" else "Pressure (bar)")
    ax.set_ylabel("Adsorbed amount (cm$^3$/g)")
    st.pyplot(fig)







st.write('For the figures below, the plot on the left shows the contribution of figures of the kernel ' +
        'and the right hand side shows the contribution plotted in the density-temperature space where the ' +
        'kernel atomic structures were generated. The size of the marker is proportional to the contribution.')

#plt.figure(figsize=(10, 6))
fig, ax = plt.subplots(1,2, figsize=(11,4))
ax[0].bar(range(1, structures+1), solution*100)
ax[0].set_xlabel("Structure number")
ax[0].set_ylabel("Contribution (%)")
ax[0].set_title('Contribution of each structure')
#Density-Temperature space plot
ax[1].scatter(df_structures['System density, g/cm^3'][:structures_model], #DFT structures not included
              df_structures['T(K)'][:structures_model],
              s=solution[:structures_model]*2000,
              alpha=0.8)
ax[1].set_xlabel('Density (g/cc)')
ax[1].set_ylabel('Simulated annealing temperature (K)')
ax[1].set_title(r'Contribution of each structure in $\rho$-T space')
st.pyplot(fig)




st.divider()
st.header('Contribution of Kernel')
st.write('These are the top contributor structures of the kernel to fit the experimental adsorption isotherm')
if dft_present:
    st.markdown("DFT structures are calculated based on [*Jagiello and Oliver's* paper](https://doi.org/10.1016/j.carbon.2012.12.011)")
# Print top contributions
top_n = 15
contribution_string = ""
for i in range(top_n):
# Use this if all the range is desired: for i in range(structures):
    struct = np.argsort(solution)[::-1][i]
    if struct+1 > structures_model:
        dft_deco = ' (DFT structure)'
    else:
        dft_deco = ''
    contribution_string += f"Structure #{struct+1}:\t{solution[struct]*100:0.3f}% {dft_deco}\n"

contribution_string += "-"*34
contribution_string += f"\nSum     =     {solution.sum()*100:.3f}%"
st.text(contribution_string)
st.write("These values are not normalized (and should not be). This means that a sum over 100% means that pore walls of the experimental samples are thicker than in the molecular models.")




st.header('Microstructure Visualization')
st.write('These are the top 3 structures that contribute to your isotherm, both atomic structure and simulated TEM image.')
# Create a figure of a render and simulated TEM with the top contributors
plot_top_structures(solution, offset=0)

show_more_structures = st.checkbox("Show next 3 structures")
if show_more_structures:
    plot_top_structures(solution, offset=3)






st.divider()
st.header('Morphological Information')
st.write('Below are textural statistics predicted using 3D-VIS for the isotherm provided.')

#Some structures are DFT, not atomistic; several statistics below exclude them
sum_solution = np.sum(solution)
sum_solution_model = np.sum(solution[:structures_model])
sum_solution_dft = np.sum(solution[structures_model:])

total_area = np.sum(df_structures['Total surface area m^2/g']*solution)
simulation_temperature = np.sum((df_structures['T(K)']*solution)[:structures_model])/sum_solution_model

temp_exp = 1/simulation_temperature - KB_EV_PER_K / ACTIVATION_ENERGY_EV * np.log(ANNEALING_TIME_S/3600)
temp_exp = 1/temp_exp

text_results_info = f"Sum of solution = {sum_solution:.3f}\n"
text_results_info += f"Sum of solution only atomistic = {sum_solution_model:.3f}\n"
if dft_present:
    text_results_info += f"Sum of solution only DFT = {sum_solution_dft:.3f}\n"
    text_results_info += f"DFT part = {sum_solution_dft/sum_solution*100:.2f}%\n"
text_results_info += f"Density g/cc (excludes DFT) = " \
                     f"{np.sum((df_structures['System density, g/cm^3']*solution)[:structures_model]):.4f}\n"
text_results_info += f"He volume cc/g (excludes DFT) = " \
      f"{np.sum((df_structures['Helium volume in cm^3/g']*solution)[:structures_model]):.4f}\n"
text_results_info += f"Geometric (point accessible) volume in cm³/g = " \
      f"{np.sum(df_structures['Geometric (point accessible) volume in cm^3/g']*solution):.4f}\n"
#print(f"Probe-occupiable volume cc/g = {sum(df_structures['V PO cm3/g']*solution):.4f}")
#print(f"Accessible area m2/g = {int(sum(df_structures[' Accessible surface area per mass in m^2/g']*solution)):d}")
text_results_info += f"Total area m²/g = {int(total_area):d}\n"
text_results_info += f"Simulation temperature K (excludes DFT) = {simulation_temperature:.0f}\n"
text_results_info += f"Equivalent graphitization temperature K (excludes DFT) = {temp_exp:.0f}"
st.text(text_results_info)



st.divider()
st.header('Pore Size Distribution (PSD)')



# Calculate PSD
PSD_solution = (np.array(df_structures['Helium volume in cm^3/g']) * solution * np_PSD_pb).sum(axis=1)

#Smooth the PSD
smooth_kernel_size = 10 # Increase this for smoother results, 70 is for paper
smooth_kernel = np.array(PascalTriangle(smooth_kernel_size))
smooth_kernel = smooth_kernel / smooth_kernel.sum()
PSD_solution_smooth = np.convolve(PSD_solution, smooth_kernel, mode='same')
#First 3 points are not zero, but should not plot, we can use NaNs
PSD_solution_smooth[0:3] = np.nan

if dft_present:
    #DFT pore sizes live on a 1 A grid from 0 to past the largest pore, with margin so
    #the smoothing kernel does not clip the largest peak. DFT pores smaller than the end
    #of the Poreblazer grid (52.4 A) overlay the atomistic PSD range on the plot.
    psd_dft_size = np.arange(0,
                             df_structures['moment1'].iloc[-1] + 30,
                             1,
                             dtype=float)

    #Each DFT structure contributes its pore volume (derived in load_kernel from the
    #isotherm normalization) as a spike integrating to volume*weight on the 1 A grid.
    psd_dft = np.zeros_like(psd_dft_size)
    for index, value in df_structures['moment1'][structures_model:].items():
        index_pore = np.searchsorted(psd_dft_size, value)
        psd_dft[index_pore] += df_structures['Helium volume in cm^3/g'][index] * solution[index-1]

    smooth_kernel_size = 40 # Increase this for smoother results, cannot be larger than kernel
    smooth_kernel = np.array(PascalTriangle(smooth_kernel_size))
    smooth_kernel = smooth_kernel / smooth_kernel.sum()
    PSD_dft_smooth = np.convolve(psd_dft, smooth_kernel, mode='same')

# Plot PSD
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(8,4))

for i in range(2):
    ax[i].plot(df_PSD_pb[0]/10, PSD_solution*10, color='lavender') # Use light color for original PSD solution
    ax[i].plot(df_PSD_pb[0]/10,
               PSD_solution_smooth*10,
               linewidth=3,
               label='Atomic model',
               color="tab:blue")
    ax[i].set_xlabel("Pore size (nm)")

if dft_present:
    ax[1].plot(psd_dft_size/10,
            PSD_dft_smooth*10,
            linewidth=3, label='DFT', color='darkseagreen')

ax[1].legend()
ax[0].set_ylabel("Pore volume -dV(r)/dr")
ax[0].set_xlim([0,6])
ax[1].set_xlim([0.05,100])
ax[1].set_xscale('log')
fig.suptitle('PSD from atomic structures')
fig.tight_layout()
st.pyplot(fig)        




st.header('Surface Area and PSD')

fig, ax = plt.subplots()
ax.plot(df_PSD_pb[0]/10,
         PSD_solution*10/(df_PSD_pb[0]/10),
         label='Raw',
         color='peachpuff')
ax.plot(df_PSD_pb[0]/10,
         PSD_solution_smooth*10/(df_PSD_pb[0]/10),
         label='Smooth',
         color='tab:blue',
         linewidth=3)

ax.set_xlabel('Pore size (nm)')
ax.set_ylabel('-dA/dr')
ax.legend()
ax.set_title('Surface area as function of pore size')
st.pyplot(fig)        




cum_area = cumulative_trapezoid(PSD_solution*10/(df_PSD_pb[0]/10), x=df_PSD_pb[0]/10, initial=0)
cum_area /= cum_area[-1]
cum_area *= total_area

pore_range_area_tuple = st.slider(
    "Move the slider below to select pore range to calculate specific surface area within a custom pore range (nm):",
    np.min(df_PSD_pb[0]/10), np.max(df_PSD_pb[0]/10), (1.40, 3.00))

pore_range_area = np.array([float(pore_range_area_tuple[0]), float(pore_range_area_tuple[1])]) #Enter pore range here

ssa_pore_range_area = np.interp(pore_range_area, df_PSD_pb[0]/10, cum_area)
area_between = ssa_pore_range_area[-1] - ssa_pore_range_area[0]



fig, ax = plt.subplots()
#add horizontal lines
ax.plot([0, pore_range_area[-1]], [ssa_pore_range_area[-1]]*2, linestyle='-', color='tab:orange')
ax.plot([0, pore_range_area[0]], [ssa_pore_range_area[0]]*2, linestyle='-', color='tab:orange')
#add vertical lines
ax.plot([pore_range_area[0]]*2, [0, ssa_pore_range_area[0]], linestyle='--', color='tab:orange')
ax.plot([pore_range_area[-1]]*2, [0, ssa_pore_range_area[-1]], linestyle='--', color='tab:orange')
#Add arrows
ax.plot(0, ssa_pore_range_area[-1], marker='<', color='tab:orange')
ax.plot(0, ssa_pore_range_area[0], marker='<', color='tab:orange')
#Add cummulative surface area plot
ax.plot(df_PSD_pb[0]/10, cum_area, linewidth=3)

ax.set_xlabel('Pore size (nm)')
ax.set_ylabel(r'Surface area (m$^2$/g)')
filter_area_plot = (df_PSD_pb[0]/10 > pore_range_area[0]) & (df_PSD_pb[0]/10 < pore_range_area[1])
#print(filter_area_plot.sum())

#Fill small area without curve
ax.fill_between([0, pore_range_area[0]], [ssa_pore_range_area[0]]*2, [ssa_pore_range_area[-1]]*2, color='oldlace')
#Fill small area with curve
ax.fill_between((df_PSD_pb[0]/10)[filter_area_plot], cum_area[filter_area_plot], ssa_pore_range_area[-1],  color='oldlace')

#the classic:


ax.text(np.mean([0, pore_range_area[0]]),
        ssa_pore_range_area.mean(),
        rf'{area_between:.1f} m$^2$/g',
        horizontalalignment='center',
        verticalalignment='center')


st.pyplot(fig)        
st.text(f'Area between {pore_range_area[0]} nm and {pore_range_area[1]} nm is {area_between:.1f} m²/g')





cum_psd = cumulative_trapezoid(PSD_solution*10, x=df_PSD_pb[0]/10, initial=0)
psd_export = np.array([df_PSD_pb[0]/10,
                        PSD_solution*10,
                        PSD_solution_smooth*10,
                        cum_psd,
                        cum_area,
                        PSD_solution_smooth*10/(df_PSD_pb[0]/10)]).T
#header_psd_export = f"Pore size (nm)\tPSD\tSmoothed PSD\tCumulative PSD\tCumulative SSA"
export_string = f"Pore size (nm)\tPSD\tSmoothed PSD\tCumulative PSD\tCumulative SSA\tSSA Smoothed\r\n"


for i in range(psd_export.shape[0]):
    export_string += f"{psd_export[i,0]:.4f}\t{psd_export[i,1]:.7f}\t{psd_export[i,2]:.7f}\t{psd_export[i,3]:.7f}\t{psd_export[i,4]:.2f}\t{psd_export[i,5]:.4f}\r\n"

st.download_button(
    label="Download PSD data as tab-separated values",
    data=export_string,
    file_name="export_PSD.tsv",
    mime="text/plain",
)




st.divider()
st.header('Adsorbate Conversion')
st.write('Because the fit expresses your sample as a combination of kernel structures, the same '
         'structural weights can predict how the sample would adsorb a different gas, using that '
         "gas's pre-computed kernel. Only the atomistic structures shared across kernels can be "
         'transferred; if the fit above still includes DFT structures, uncheck "Include DFT '
         'structures" in the Kernel Selection for a faithful conversion.')

target_options = [name for name in KERNELS if name != kernel_radio]

if not target_options:
    st.info('No other kernel is available to convert to.')
else:
    target_name = st.selectbox("Predict the isotherm for:", target_options)
    target_config = KERNELS[target_name]

    #Only the leading atomistic structures share indices across kernels, so only those weights
    #can be transferred. Any DFT weight in the source fit has no target counterpart.
    n_transfer = min(structures_model, target_config['structures_model'])

    _, df_target_iso = load_kernel(target_config, target_config['structures'])
    np_target_iso = np.array(df_target_iso)[:, 1:]
    np_target_pressure = np.array(df_target_iso)[:, 0]

    #Predicted isotherm = same structural weights applied to the target kernel's isotherms
    predicted_isotherm = np_target_iso[:, :n_transfer] @ solution[:n_transfer]

    dropped_weight = solution[n_transfer:].sum()
    if dropped_weight > 1e-9:
        st.warning(f"{dropped_weight / solution.sum() * 100:.1f}% of the fit comes from DFT "
                   f"structures with no counterpart in the {target_name} kernel; they were "
                   f"excluded from this prediction. Uncheck \"Include DFT structures\" above "
                   f"and re-run for a faithful conversion.")

    convert_x_axis_scale = st.radio(
        "Select x-axis scaling for the plot below",
        ["Logarithmic", "Linear"],
        key='log convert')

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np_target_pressure, predicted_isotherm,
            label=f'Predicted {target_name}',
            marker='o',
            markersize=4,
            linestyle='solid',
            color='tab:green')
    ax.set_xlabel("Relative pressure P/P$_0$" if target_config['pressure_unit'] == "P/P₀" else "Pressure (bar)")
    ax.set_ylabel("Adsorbed amount (cm$^3$/g)")
    ax.set_ylim(bottom=0)
    if convert_x_axis_scale == 'Logarithmic':
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))
    ax.set_title(f'Predicted {target_name} isotherm')
    ax.legend()
    ax.grid(color='aliceblue')
    st.pyplot(fig)

    #Export predicted isotherm as tab-separated values
    convert_export = "Relative pressure\tAdsorbed amount (cm3/g STP)\r\n"
    for pressure, amount in zip(np_target_pressure, predicted_isotherm):
        convert_export += f"{pressure:.8e}\t{amount:.7f}\r\n"
    st.download_button(
        label=f"Download predicted {target_name} isotherm as tab-separated values",
        data=convert_export,
        file_name="predicted_isotherm.tsv",
        mime="text/plain",
    )
