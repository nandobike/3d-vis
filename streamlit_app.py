import streamlit as st

# Import all necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.ticker as ticker

import pandas as pd
from scipy.optimize import nnls
#from os import path

#some old versions of scipy do not have the cumulative trapezoid function
try:
    from scipy.integrate import cumulative_trapezoid
except:
    print("Old Scipy, cumulative PSD will throw error later")






def find_range(contents): #Reads Belsorp
    string_find = b'"No."\t"Pe/kPa"\t"P0/kPa"\t"Vd/ml"\t"V/ml(STP) g-1"'
    string_find2 = b'"No."\t"Pe/kPa"\t"P0/kPa"\t"Vd/ml"\t"V/ml(STP)\x81Eg-1"'
    #print(contents[35] == b'"No."\t"Pe/kPa"\t"P0/kPa"\t"Vd/ml"\t"V/ml(STP)\x81Eg-1"')
    start_adsorption, start_desorption = \
        [i for i,x in enumerate(contents) if ((x==string_find) or (x==string_find2))]
    string_find = b'0\t0\t0\t0\t0'
    end_adsorption, end_desorption = [i for i,x in enumerate(contents) if x==string_find]    
    return (start_adsorption, end_adsorption, start_desorption, end_desorption)

def read_branch(contents, branch): #Reads Belsorp
    #Finds the start and end of branches
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
    return np.column_stack((isotherm[:,0]/isotherm[:,1], isotherm[:,2]))











st.title('3D-VIS Isotherm Analysis')

multi = '''This is the app that solves the microstructure of a porous carbon based on the paper:

*F. Vallejos-Burgos et al.* **3D nanostructure prediction of porous carbons via gas adsorption**, Carbon, Volume 215, 2023, 118431.

[Here is our paper published in Carbon](https://doi.org/10.1016/j.carbon.2023.118431)

If you use the program please remember to cite us.

Questions? Suggestions? Complaints? Shoot us an email: fvb@vallejos.cl
'''
st.markdown(multi)

st.image('summary.jpg', caption='Summary of the 3D-VIS method')









#Structures available in the kernel
structures = 89
#structures = 78 #This needs to be added if using models only

#Structures that are calculated with atomistic model (not through Kelvin equation)
structures_model = 78

#Excel filename with the kernel data
print('Load kernel')
excel_database = r'kernel.xlsx'


#Load structural parameters into a Pandas dataframe
df_structures = pd.read_excel(excel_database,
                   sheet_name='Details',
                   header=1,
                   nrows=structures,
                   index_col=1,
                   engine='openpyxl')

#Load calculated adsorption isotherms into a dataframe
df_isotherm = pd.read_excel(excel_database,
                   #sheet_name='N2 77 K 1CLJ',
                   #sheet_name='Ar 87 K 1CLJ', #comment above and uncomment this to use Ar 87 K kernel
                   sheet_name='N2 77 K 1CLJ_rc1',
                   header=None,
                   skiprows=8,
                   #nrows=64,
                   nrows=93,
                   usecols=range(0,structures+1),
                   engine='openpyxl')

#These are structures with low density that did not form a solid framework.
#By making their isotherms equal to zero, they are removed from the regression
df_isotherm[9] = 0
df_isotherm[13] = 0

#Read pore size distributions and load into dataframe
df_PSD_pb = pd.read_excel(excel_database,
                   sheet_name='Poreblazer PSDs_2', #_2 for ultranarrow pores ~1 A
                   header=None,
                   skiprows=6,
                   nrows=210,
                   usecols=range(0,structures+1),
                   engine='openpyxl')
#Convert pore size distribution data to a numpy array
np_PSD_pb = np.array(df_PSD_pb)[:,1:]




st.divider()
st.header('Isotherm Data Load')
st.markdown('Upload your isotherm as a text file. The file must only contain datapoints in ascending pressure order. Two columns separated by tabs, first for relative pressure, second for adsorbed amount in cc STP/g. See an example [here](https://raw.githubusercontent.com/nandobike/3d-vis/main/examples/a20_lao.tsv)')
#load experimental isotherm
#It must be a tab-separated file with two columns.
#First column is relative pressure and second column adsorbed volume in units cc STP/g
file = st.file_uploader("Upload isotherm file")
if file is None:
    file = "examples/a20_lao.tsv"
    st.write(f"No file was uploaded. Loading a default isotherm file: {file}")
    #file_parameters = read_parameters(file)
    #file_name_print = file
else:
    #for line in filename:
    #    st.write(line.decode('cp1252').rstrip())
    #file_parameters = read_parameters_uploaded(file)
    #file_name_print = file.name
    #1+1
    st.write(f'A file was uploaded: {file.name} as {file.type}')

#print('-------------------')
#print(file)
#print(file.type)



if isinstance(file, str):
    exp_iso = np.genfromtxt(file, delimiter="\t") #load isotherm file into numpy array

#elif (file.type == 'application/octet-stream'): #This is now not needed
elif next(file) == b'====================\r\n': #If a Belsorp file is loaded
    st.write('This file seems to be Belsorp format, will attempt load.')
    contents = []
    for line in file:
        #print(line)
        #contents.append(line.decode('shift-jis').rstrip()) #this works new version
        contents.append(line.rstrip())
    #print(contents) #DEBUG
    exp_iso = read_branch(contents, 'adsorption')
else: #Now the standard file
    exp_iso = np.genfromtxt(file, delimiter="\t") #load isotherm file into numpy array







base_exp_filename = 'Isotherm_data' #path.splitext(experimental_isotherm_file)[0]



st.divider()
st.header('Data Cleaning and Validation')
st.write('Usually it is necessary to remove a few experimental points from the very low pressures since they are very inaccurate. Look at the error in the fitted isotherm plot in the Results section to know how many to remove.')
#Remove some initial experimental points where the experimental data is usually flawed
#points_to_remove = 13 #for a20_lao
points_to_remove = st.slider("Use the slider below to remove initial points from the isotherm:", 0, np.shape(exp_iso)[0], 0)

st.write(f'Now the points from {points_to_remove} to {np.shape(exp_iso)[0]} will be used in the calculation')

np_isotherm = np.array(df_isotherm)[points_to_remove:,1:]
np_pressure_gcmc = np.array(df_isotherm)[points_to_remove:,0]

exp_iso_interp = np.interp(np_pressure_gcmc, exp_iso[:,0], exp_iso[:,1]) #interpolate isotherm to points of the kernel

fig, ax = plt.subplots(figsize=(7,4))
ax.plot(exp_iso[:,0], exp_iso[:,1],label='Experimental', marker='o', linestyle='none')
ax.set_xlabel("Relative pressure P/P$_0$")
ax.set_ylabel("Adsorbed amount (cm$^3$/g)")
ax.set_ylim(bottom=0)  # adjust the bottom leaving top unchanged
ax.set_xlim(left=1e-8, right=1.4)
ax.plot(np_pressure_gcmc, exp_iso_interp,
         label='Experimental interpolated',
         marker='x',
         markersize=4,
         linestyle='none')
ax.set_xscale('log')
ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))
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

def calculate_isotherm(solution):
    # This function sums the contributions of every kernel structure
    # in order to calculate the predicted isotherm.
    isotherm = np.zeros(np_pressure_gcmc.size) #create an empty vector
    for i in range(structures):
        isotherm = isotherm + solution[i] * np.array(np_isotherm[:,i])
    return isotherm




# Plot experimental datapoints and show the fit
log_scale_plot = True #use True if you want to plot using logarithmic scale in x
fig, ax = plt.subplots(2, gridspec_kw={'height_ratios': [1, 3]}, dpi=120) #, figsize=(3,3)

# Top plot for error
ax[0].set_title('Experimental Data and Fitted Isotherm')
ax[0].plot(np_pressure_gcmc, exp_iso_interp-calculate_isotherm(solution), marker='o', linestyle='solid', color='tab:orange')
if log_scale_plot:
    ax[0].set_xscale('log')
    ax[0].xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))

ax[0].set_ylabel("Error (cm$^3$/g)")
ax[0].grid(color='aliceblue')

#ax[0].set_ylabel("Error")
if log_scale_plot:
    ax[0].set_xlim(left=1e-8, right=1.4)
    
else:
    ax[0].set_xlim(left=-0.02, right=1)

ax[0].axes.get_xaxis().set_ticks([])
#ax[0].axes.get_yaxis().set_ticks([])

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
#ax[1].set_xlabel("P/P$_0$")
ax[1].set_ylabel("Adsorbed amount (cm$^3$/g)")
#ax[1].set_ylabel("Adsorption")
ax[1].legend()
ax[1].grid(color='aliceblue')

ax[1].set_ylim(bottom=0)
if log_scale_plot:
    ax[1].set_xlim(left=1e-8, right=1.4)
else:
    ax[1].set_xlim(left=-0.02, right=1)
#if log_scale_plot:
#    ax[1].set_xlim(left=1e-7, right=1.4)
#ax[1].axes.get_yaxis().set_ticks([])
plt.ylim(bottom=0)
st.pyplot(fig)

st.text(f"Residual total= {residual:.3f} cc/g") #norm of residuals = sqrt of sum (error^2)
st.text(f"Residual per point = {residual/np_pressure_gcmc.size:.3f} cc/g") #norm of residuals = sqrt of sum (error^2)






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
st.write('These are the top contributor structurs of the kernel to fit the experimantal adsorption isotherm')
# Print top contributions
top_n = 15
contribution_string = ""
for i in range(top_n):
# Use this if all the range is desired: for i in range(structures):
    struct = np.argsort(solution)[::-1][i]
    if struct+1 > structures_model:
        kelvin_deco = ' (Kelvin structure)'
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
fig, ax = plt.subplots(2,3, figsize=(16,10))
for i in range(3):
    structure_render = np.argsort(solution)[-1-i]+1
    if structure_render <= structures_model:
        file_render = f'rendered structures/{structure_render:03d}.png'
        render = img.imread(file_render)
        ax[0,i].imshow(render)
        ax[0,i].set_axis_off()
    else:
        ax[0,i].set_xticks([])
        ax[0,i].set_yticks([])
        ax[0,i].text(0.14,0.5, "Structure modeled via Kelvin equation")
    
    ax[0,i].title.set_text(f'Structure {structure_render:02d} ({(solution[structure_render-1])/sum(solution)*100:.1f}%)')

    
    if structure_render <= structures_model:
        file_render = f'simulated TEM/{structure_render:02d}.tif'
        render = img.imread(file_render)
        ax[1,i].imshow(render, cmap='gist_gray')

        with open (f'structures/{structure_render:03d}.xyz') as fh:
            next(fh)
            lattice_size = float(next(fh).split(" ")[0])
        bar_length_pixels = 20*render.shape[0]/lattice_size
        pixels_sim_tem = render.shape[0]
        ax[1,i].text(50,
                     pixels_sim_tem*0.91,
                     "2 nm",
                     color='white',
                     fontsize=15,
                     fontweight='bold')
        ax[1,i].plot([50, 50+bar_length_pixels],
                     [pixels_sim_tem*0.95, pixels_sim_tem*0.95],
                     '-',
                     lw=5,
                     color='white')
        ax[1,i].set_axis_off()
    else:
        ax[1,i].set_xticks([])
        ax[1,i].set_yticks([])
        ax[1,i].text(0.42,0.5, "No image")
st.pyplot(fig)        
#Save figure
#if True: #set to True to save figure
#    plt.savefig(path.join("results", (base_exp_filename + "_render_TEM.tiff")),
#                bbox_inches='tight',
#                dpi=160)



st.divider()
st.header('Morphological Information')
st.write('Below are textural statistics predicted using 3D-VIS for the isotherm provided.')

#Some structures need to be not considered since are not atomistic but Kelvin
sum_solution = sum(solution)
sum_solution_model = sum(solution[:structures_model])
sum_solution_kelvin = sum(solution[structures_model:])

total_area = sum(df_structures['Total surface area m^2/g']*solution)
simulation_temperature = sum((df_structures['T(K)']*solution)[:structures_model])/sum_solution_model

time_sim = 2e-9 #2 nanoseconds in seconds
kB = 8.617e-5 #eV/K
activ_energy = 6 #eV
temp_exp = 1/simulation_temperature - kB / activ_energy * np.log(time_sim/3600)
temp_exp = 1/temp_exp

text_results_info = f"Sum of solution = {sum_solution:.3f}\n"
text_results_info += f"Sum of solution only atomistic = {sum_solution_model:.3f}\n"
text_results_info += f"Sum of solution only Kelvin = {sum_solution_kelvin:.3f}\n"
text_results_info += f"Kelvin part = {sum_solution_kelvin/sum_solution*100:.2f}%\n"
text_results_info += f"Density g/cc (excludes Kelvin) = " \
                     f"{sum((df_structures['System density, g/cm^3']*solution)[:structures_model]):.4f}\n"
text_results_info += f"He volume cc/g (excludes Kelvin) = " \
      f"{sum((df_structures['Helium volume in cm^3/g']*solution)[:structures_model]):.4f}\n"
text_results_info += f"Geometric (point accessible) volume in cm^3/g = " \
      f"{sum(df_structures['Geometric (point accessible) volume in cm^3/g']*solution):.4f}\n"
#print(f"Probe-occupiable volume cc/g = {sum(df_structures['V PO cm3/g']*solution):.4f}")
#print(f"Accessible area m2/g = {int(sum(df_structures[' Accessible surface area per mass in m^2/g']*solution)):d}")
text_results_info += f"Total area m2/g = {int(total_area):d}\n"
text_results_info += f"Simulation temperature K (excludes Kelvin) = {simulation_temperature:.0f}\n"
text_results_info += f"Equivalent graphitization temperature K (excludes Kelvin) = {temp_exp:.0f}"
st.text(text_results_info)



st.divider()
st.header('Pore Size Distribution (PSD)')

def PascalTriangle(n):
    # This calculates a Pascal triangle that will be used for smoothing
    # https://www.askpython.com/python/examples/pascals-triangle-using-python
    # https://danielmuellerkomorowska.com/2020/06/02/smoothing-data-by-rolling-average-with-numpy/
    trow = [1]
    y = [0]
    for x in range(n):
        trow=[left+right for left,right in zip(trow+y, y+trow)]
    return trow


# Calculate PSD
PSD_weighted = np_PSD_pb
PSD_solution = (np.array(df_structures['Helium volume in cm^3/g']) * solution * PSD_weighted).sum(axis=1)
#Smooth the PSD
smooth_kernel_size = 10 # Increase this for smoother results, 70 is for paper
smooth_kernel = np.array(PascalTriangle(smooth_kernel_size))
smooth_kernel = smooth_kernel / smooth_kernel.sum()
PSD_solution_smooth = np.convolve(PSD_solution, smooth_kernel, mode='same')
#First 3 points are not zero, but should not plot, we can use NaNs
PSD_solution_smooth[0:3] = np.nan
#Create a vector of the PSD sizes in Angstrom for Kelvin
psd_kelvin_size = np.arange(df_PSD_pb[0].iloc[-1],500,1,dtype=float)

#Create a vector of zeros to store what the PSD for Kelvin will be
psd_kelvin = np.zeros_like(psd_kelvin_size)

for index, value in df_structures['moment1'][structures_model:].items():
    #print(index, value)
    index_pore = np.searchsorted(psd_kelvin_size, value)
    psd_kelvin[index_pore] = df_structures['Helium volume in cm^3/g'][index] * solution[index-1]

smooth_kernel_size = 400 # Increase this for smoother results
smooth_kernel = np.array(PascalTriangle(smooth_kernel_size))
smooth_kernel = smooth_kernel / smooth_kernel.sum()
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

ax[1].plot(np.append(df_PSD_pb[0]/10, psd_kelvin_size[0]/10)[-2:],
           np.append(PSD_solution_smooth*10, 0)[-2:],
           color="tab:blue",
           linewidth=2, linestyle=(0, (1, 1)) )
    
#ax[1].plot(psd_kelvin_size/10, psd_kelvin)
ax[1].plot(psd_kelvin_size/10,
           PSD_kelvin_smooth*10,
           linewidth=3, label='Kelvin', color='darkseagreen')
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
         PSD_solution_smooth*10/(df_PSD_pb[0]/10),
         label='Smooth')
ax.plot(df_PSD_pb[0]/10,
         PSD_solution*10/(df_PSD_pb[0]/10),
         label='Raw')
ax.set_xlabel('Pore size (nm)')
ax.set_ylabel('-dA/dr')
ax.legend()
ax.set_title('Surface area as function of pore size')
st.pyplot(fig)        




cum_area = cumulative_trapezoid(PSD_solution*10/(df_PSD_pb[0]/10), x=df_PSD_pb[0]/10, initial=0)
cum_area /= cum_area[-1]
cum_area *= total_area
#pore_range_area = np.array([2.05, 3.84]) #Enter pore range here

pore_range_area_tuple = st.slider(
    "Select a pore range to calculate specific surface area:",
    np.min(df_PSD_pb[0]/10), np.max(df_PSD_pb[0]/10), (2.05, 3.84))

pore_range_area = np.array([float(pore_range_area_tuple[0]), float(pore_range_area_tuple[1])]) #Enter pore range here

ssa_pore_range_area = np.interp(pore_range_area, df_PSD_pb[0]/10, cum_area)
area_between = ssa_pore_range_area[-1] - ssa_pore_range_area[0]

fig, ax = plt.subplots()
ax.plot(df_PSD_pb[0]/10, cum_area)
ax.set_xlabel('Pore size (nm)')
ax.set_ylabel(r'Surface area (m$^2$/g)')
filter_area_plot = (df_PSD_pb[0]/10 > pore_range_area[0]) & (df_PSD_pb[0]/10 < pore_range_area[1])
#print(filter_area_plot.sum())
ax.fill_between((df_PSD_pb[0]/10)[filter_area_plot], cum_area[filter_area_plot], color='bisque')
ax.text(pore_range_area.mean(),
        ssa_pore_range_area[0]/2,
        rf'{area_between:.1f} m$^2$/g',
        horizontalalignment='center',
        verticalalignment='center')
        #transform=ax.transAxes)
st.pyplot(fig)        
st.text(f'Area between {pore_range_area[0]} nm and {pore_range_area[1]} nm is {area_between:.1f} m2/g')





cum_psd = cumulative_trapezoid(PSD_solution*10, x=df_PSD_pb[0]/10, initial=0)
psd_export = np.array([df_PSD_pb[0]/10,
                        PSD_solution*10,
                        PSD_solution_smooth*10,
                        cum_psd,
                        cum_area]).T
#header_psd_export = f"Pore size (nm)\tPSD\tSmoothed PSD\tCumulative PSD\tCumulative SSA"
export_string = f"Pore size (nm)\tPSD\tSmoothed PSD\tCumulative PSD\tCumulative SSA\r\n"

#print(psd_export.shape)

for i in range(psd_export.shape[0]):
    export_string += f"{psd_export[i,0]:.4f}\t{psd_export[i,1]:.7f}\t{psd_export[i,2]:.7f}\t{psd_export[i,3]:.7f}\t{psd_export[i,4]:.2f}\r\n"

#np.savetxt(path.join("results", (base_exp_filename + "_export_PSD.tsv")),
#            psd_export,
#            delimiter='\t',
#            header=header_psd_export,
#            comments="")

st.download_button(
    label="Download PSD data as tab-separated values",
    data=export_string,
    file_name="export_PSD.tsv",
    mime="text/plain",
)
