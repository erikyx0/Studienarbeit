import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.ticker import FuncFormatter
import seaborn as sb  # type: ignore

import os 
import shutil 
from collections import OrderedDict


#from Latex_table import df_to_table

#%% Farben für die Plots definieren
import sys

# Pfad zum Hauptordner hinzufügen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))

from colors import colors_new
print(colors_new)

color1 = '#4878A8'
color2 = '#7E9680'
color3 = '#B3B3B3'
color4 = '#BC6C25'
color5 = "#960B0B"
color6 = "#077B1A"

color1 = colors_new[0]
color2 = colors_new[1]
color3 = colors_new[2]
color4 = colors_new[3]
color5 = colors_new[4]
color6 = colors_new[5]

colors = [color1,color2,color3,color4,color5,color6]

colors = colors_new

# Set the working directory to the location of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

"""
1 = 0.95/ 0.05
2 = 0.90/ 0.10
3 = 0.85/ 0.15
"""

#%% Daten einlesen 
#df_R1_PFR7 = pd.read_csv('Daten/1/21.soln_no_1_PFRC7.csv')
df_R2_PFR7 = pd.read_csv('Daten/kein CO2/2/21.soln_no_1_PFRC7.csv')
df_R3_PFR7 = pd.read_csv('Daten/kein CO2/3/21.soln_no_1_PFRC7.csv')

