import streamlit as st

# Import all necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import pandas as pd
from scipy.optimize import nnls
#from os import path

#some old versions of scipy do not have the cumulative trapezoid function
try:
    from scipy.integrate import cumulative_trapezoid
except:
    print("Old Scipy, cumulative PSD will throw error later")

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



















st.write('Hello world!')
