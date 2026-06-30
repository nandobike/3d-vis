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

#Excel file holding every kernel (isotherms, structural details, PSDs)
EXCEL_DATABASE = r'kernel.xlsx'

#Kernel definitions. Each entry describes one simulated adsorption kernel in EXCEL_DATABASE.
# - isotherm_sheet / isotherm_rows: where to read the isotherm matrix and how many pressure points.
# - structures: total number of structure columns in the kernel.
# - structures_model: how many of those are atomistic. The remaining (structures - structures_model)
#   are DFT/Kelvin structures appended at the end. They have no .xyz/render/TEM assets and no
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
        Per-structure parameters, one row per structure (n_structures rows).
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

    return df_structures, df_isotherm





st.title('3D-VIS Isotherm Analysis')

multi = '''This app predicts the 3D nanostructure of a porous carbon from an experimental gas adsorption isotherm,                    
  by fitting it as a linear combination of pre-calculated atomistic kernel isotherms (Non-Negative Least Squares).                      
  Results include the contributing structures, morphological statistics, and pore size distribution.
                                                                                                                                        
  **Supported measurements:** N₂ at 77 K, CO₂ at 298.15 K, CO₂ at 273 K, and H₂ at 77 K (high pressure, up to 120 bar).

  **Supported file formats:** tab-separated values (P/P₀ vs cm³/g) or Belsorp `.DAT` export files.

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
    help="DFT (Kelvin) structures model larger pores but have no counterpart in DFT-free "
         "kernels such as CO₂. Uncheck for a fit that can be converted to any adsorbate "
         "without losing weight. Disabled when the selected kernel has no DFT structures."
)

#Structures available in the kernel (atomistic only when DFT is excluded)
structures = kernel_config['structures'] if (has_dft and include_dft) else kernel_config['structures_model']

#Structures that are calculated with atomistic model (not through Kelvin equation)
structures_model = kernel_config['structures_model']

#Load structural parameters and calculated adsorption isotherms
df_structures, df_isotherm = load_kernel(kernel_config, structures)


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

#Create a boolean that means that there is DFT isotherms. Only True if structures_model not equals to structures
dft_present = (structures != structures_model)


st.divider()
st.header('Isotherm Data Load')
_pressure_col_desc = "relative pressure (P/P₀)" if kernel_config['pressure_unit'] == "P/P₀" else "pressure in bar"
st.markdown(f'Upload your isotherm as a text file. The file must only contain datapoints in ascending pressure order. Two columns separated by tabs, first for {_pressure_col_desc}, second for adsorbed amount in cc STP/g. See an example [here](https://raw.githubusercontent.com/nandobike/3d-vis/main/examples/a20_lao.tsv)')

#load experimental isotherm
#It must be a tab-separated file with two columns.
#First column is relative pressure and second column adsorbed volume in units cc STP/g
file = st.file_uploader("Upload isotherm file")

if file is None:
    file = "examples/a20_lao.tsv"
    st.write(f"No file was uploaded. Loading a default isotherm file: {file}")

else:
    st.write(f'A file was uploaded: {file.name} as {file.type}')



if isinstance(file, str): #True if file is the example. False if it is an uploaded file.
    exp_iso = np.genfromtxt(file, delimiter="\t") #Load example. Originally a20_lao.tsv
else: #Read uploaded file
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
        exp_iso = np.genfromtxt(file, delimiter="\t") #load isotherm file into numpy array


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

#Use non-negative least squares to find the coefficients that fit the experimental isotherm from the kernel isotherms
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html
solution, residual = nnls(np_isotherm, exp_iso_interp)





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
            kelvin_deco = ' DFT'
            linestyle = 'dashed'
            alpha=0.7
        else:
            kelvin_deco = ''
            linestyle = 'solid'
            alpha=0.9
        contribution_string = f"S #{struct+1} {kelvin_deco}"

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
ax[1].scatter(df_structures['System density, g/cm^3'][:structures_model], #Kelvin structures not included
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
        kelvin_deco = ' (DFT structure)'
    else:
        kelvin_deco = ''
    contribution_string += f"Structure #{struct+1}:\t{solution[struct]*100:0.3f}% {kelvin_deco}\n"

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

#Some structures need to be not considered since are not atomistic but Kelvin
sum_solution = np.sum(solution)
sum_solution_model = np.sum(solution[:structures_model])
sum_solution_kelvin = np.sum(solution[structures_model:])

total_area = np.sum(df_structures['Total surface area m^2/g']*solution)
simulation_temperature = np.sum((df_structures['T(K)']*solution)[:structures_model])/sum_solution_model

temp_exp = 1/simulation_temperature - KB_EV_PER_K / ACTIVATION_ENERGY_EV * np.log(ANNEALING_TIME_S/3600)
temp_exp = 1/temp_exp

text_results_info = f"Sum of solution = {sum_solution:.3f}\n"
text_results_info += f"Sum of solution only atomistic = {sum_solution_model:.3f}\n"
if dft_present:
    text_results_info += f"Sum of solution only DFT = {sum_solution_kelvin:.3f}\n"
    text_results_info += f"DFT part = {sum_solution_kelvin/sum_solution*100:.2f}%\n"
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
#Create a vector of the PSD sizes in Angstrom for Kelvin
#May need to increase the range if there are larger pores in the kernel
psd_kelvin_size = np.arange(df_PSD_pb[0].iloc[-1],
                            df_structures['moment1'].iloc[-1]+1, #this may need to increase
                            1,
                            dtype=float)

#Create a vector of zeros to store what the PSD for Kelvin will be
psd_kelvin = np.zeros_like(psd_kelvin_size)


for index, value in df_structures['moment1'][structures_model:].items():
    index_pore = np.searchsorted(psd_kelvin_size, value)
    psd_kelvin[index_pore] = df_structures['Helium volume in cm^3/g'][index] * solution[index-1]

smooth_kernel_size = 40 # Increase this for smoother results, cannot be larger than kernel
smooth_kernel = np.array(PascalTriangle(smooth_kernel_size))
smooth_kernel = smooth_kernel / smooth_kernel.sum()

if dft_present:
    PSD_kelvin_smooth = np.convolve(psd_kelvin, smooth_kernel, mode='same')

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
    ax[1].plot(np.append(df_PSD_pb[0]/10, psd_kelvin_size[0]/10)[-2:],
            np.append(PSD_solution_smooth*10, 0)[-2:],
            color="tab:blue",
            linewidth=2, linestyle=(0, (1, 1)) )

if dft_present:    
    ax[1].plot(psd_kelvin_size/10,
            PSD_kelvin_smooth*10/50, #usually too big
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
