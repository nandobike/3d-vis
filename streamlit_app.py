import streamlit as st

# Import all necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import pandas as pd
from scipy.optimize import nnls
from os import path

#some old versions of scipy do not have the cumulative trapezoid function
try:
    from scipy.integrate import cumulative_trapezoid
except:
    print("Old Scipy, cumulative PSD will throw error later")



st.write('Hello world!')
