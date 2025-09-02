import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
import unicodedata

import re 
import glob 

color1 = '#4878A8'
color2 = '#7E9680'
color3 = '#B3B3B3'
color4 = '#BC6C25'
color5 = "#960B0B"
color6 = "#077B1A"


colors = [color1,color2,color3,color4,color5,color6]

# Set the working directory to the location of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

#%% Daten einlesen 
data_end_para = pd.read_excel('Daten/Parameterstudie_CO2_1.xlsm', sheet_name= '6.PFRC2_end_point_vs_parameter')

# Vollständige Daten für alle PFRC2 Durchläufe (~200k)
file = r"Daten\Parameterstudie_CO2_1.xlsm"

# Runs laden
from excel_loader import load_runs
all_data = load_runs(file, runs=range(1,22))

#%% Daten filtern
run_number = data_end_para['Run_number_()']

x_H2_end = data_end_para[' Mole_fraction_H2_PFRC2_end_point_()']

plt.plot(run_number, x_H2_end, color=colors[0], marker='^', linestyle='-')
plt.grid()
plt.show()

#%% Berechnung und Plot der Massenströme 
# Gegeben sind Molenbruch und Gesamtmassenstrom 
data_end_para['Massentrom CO2'] = data_end_para[' Mole_fraction_CO2_PFRC2_end_point_()'] * data_end_para[' Exit_mass_flow_rate_PFRC2_end_point_(kg/sec)']
data_end_para['Massentrom H2'] = data_end_para[' Mole_fraction_H2_PFRC2_end_point_()'] * data_end_para[' Exit_mass_flow_rate_PFRC2_end_point_(kg/sec)']

plt.plot(run_number, data_end_para['Massentrom CO2'], color=colors[0], marker='o', linestyle='-', label='CO2')
plt.plot(run_number, data_end_para['Massentrom H2'], color=colors[1], marker='o', linestyle='-', label='H2')
plt.legend()
plt.grid()
plt.show()
plt.close("all")

#%% Komplexe PLots 

# Plot
plt.figure(figsize=(9,5))
for run, d in all_data.groupby("Run"):
    plt.plot(d["Distance_m"], d["H2"], label=f"Run {run}", marker = 'o')

plt.xlabel("Distance PFRC2 (m)")
plt.ylabel("Mole fraction H₂ (-)")
plt.grid(True)
plt.legend(title="Run", ncol=2)
plt.tight_layout()
plt.show()