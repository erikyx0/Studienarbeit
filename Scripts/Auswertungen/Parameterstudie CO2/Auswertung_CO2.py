#%% start
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
a = 1

#%% Daten einlesen 
data_end_para = pd.read_excel('Daten/Parameterstudie_CO2_1.xlsm', sheet_name= '6.PFRC2_end_point_vs_parameter')
data_max_para = pd.read_excel('Daten/Parameterstudie_CO2_1.xlsm', sheet_name= '7.PFRC2_max_point_vs_parameter')
data_min_para = pd.read_excel('Daten/Parameterstudie_CO2_1.xlsm', sheet_name= '8.PFRC2_min_point_vs_parameter')

# Vollständige Daten für alle PFRC2 Durchläufe (~200k)
file = r"Daten\Parameterstudie_CO2_1.xlsm"

# Runs laden
dfs_by_run: dict[int, pd.DataFrame] = {}

for i in range(1, 22):  # Runs 1..21
    sheetname = f"{i+8}.soln_no_1_PFRC2_Run#{i}"
    df = pd.read_excel(file, sheet_name=sheetname, engine="openpyxl")
    dfs_by_run[i] = df

#%% Daten filtern
run_number = data_end_para['Run_number_()']
co2_run_number = data_end_para[' Mass_Flow_Rate_C1_Inlet4_PSR_(C1)_(kg/sec)']

x_H2_end = data_end_para[' Mole_fraction_H2_PFRC2_end_point_()']


#%% Berechnung und Plot der Massenströme 
# Gegeben sind Molenbruch und Gesamtmassenstrom 
data_end_para['Massentrom CO2'] = data_end_para[' Mole_fraction_CO2_PFRC2_end_point_()'] * data_end_para[' Exit_mass_flow_rate_PFRC2_end_point_(kg/sec)']
data_end_para['Massentrom H2'] = data_end_para[' Mole_fraction_H2_PFRC2_end_point_()'] * data_end_para[' Exit_mass_flow_rate_PFRC2_end_point_(kg/sec)']
data_end_para['Massentrom CO'] = data_end_para[' Mole_fraction_CO_PFRC2_end_point_()'] * data_end_para[' Exit_mass_flow_rate_PFRC2_end_point_(kg/sec)']
data_end_para['Massentrom H2O'] = data_end_para[' Mole_fraction_H2O_PFRC2_end_point_()'] * data_end_para[' Exit_mass_flow_rate_PFRC2_end_point_(kg/sec)']

data_end_para['H2/CO2'] = data_end_para['Massentrom H2'] / data_end_para['Massentrom CO2']

#%% Plots 
# Runnumber Case 2: 7.18 (interpoliert) (entspricht 0.05455 kg/s) 
# Plot für einfache Ergebnisse 
plt.figure(figsize=(10,6))
plt.plot(co2_run_number[1:], data_end_para[' Mole_fraction_CO2_PFRC2_end_point_()'][1:], color=color1, label=r'CO$_2$', marker = 's') 
plt.plot(co2_run_number[1:], data_end_para[' Mole_fraction_CO_PFRC2_end_point_()'][1:], color=color2, label=r'CO', marker = 's') 
plt.plot(co2_run_number[1:], data_end_para[' Mole_fraction_H2_PFRC2_end_point_()'][1:], color=color4, label=r'H$_2$', marker = 's')
plt.plot(co2_run_number[1:], data_end_para[' Mole_fraction_H2O_PFRC2_end_point_()'][1:], color=color5, label=r'H$_2$O', marker = 's')
plt.grid()
plt.xlabel(r'CO$_2$ Massenstrom am Einlass (kg/s)')
plt.ylabel('Molenbruch am Ende des Reaktors')
plt.vlines(x=[0.05455], ymin = 0, ymax = 0.45, colors = "grey", linestyles="--", label="Komplexere Simulation")
plt.legend()
plt.savefig('Bilder/Parameterstudie_CO2_Molenbruch_Ende.png', dpi=300, bbox_inches='tight')
plt.close("all")

# Plot für Massenströme 
plt.figure(figsize=(10,6))
plt.plot(co2_run_number[1:], data_end_para['Massentrom CO2'][1:], color=color1, label=r'CO$_2$', marker = 's') 
plt.plot(co2_run_number[1:], data_end_para['Massentrom CO'][1:], color=color2, label=r'CO', marker = 's') 
plt.plot(co2_run_number[1:], data_end_para['Massentrom H2'][1:], color=color4, label=r'H$_2$', marker = 's')
plt.plot(co2_run_number[1:], data_end_para['Massentrom H2O'][1:], color=color5, label=r'H$_2$O', marker = 's')
plt.grid()
plt.xlabel(r'CO$_2$ Massenstrom am Einlass (kg/s)')
plt.ylabel('Massenstrom am Ende des Reaktors')
plt.vlines(x=[0.05455], ymin = 0, ymax = 0.1, colors = "grey", linestyles="--", label="Komplexere Simulation")
plt.legend()
plt.savefig('Bilder/Parameterstudie_CO2_Massenstrom_Ende.png', dpi=300, bbox_inches='tight')
plt.close("all")

# Plot für bestes H2 / CO2 ergebnis 
plt.figure(figsize=(10,6))


# Testbestimmung CO2 Massenstrom Einlass 
fig, ax1 = plt.subplots(figsize=(7,5))

# --- Erste Achse: Verhältnis H2/CO2 ---
ax1.plot(co2_run_number[1:], data_end_para['H2/CO2'][1:],
         color=color1, label=r'H$_2$/CO$_2$', marker='v')
ax1.set_xlabel(r'CO$_2$ Massenstrom am Einlass (kg/s)')
ax1.set_ylabel(r'Verhältnis H$_2$/CO$_2$ am Ende des Reaktors', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True)

# vertikale Linie
ax1.vlines(x=[0.05455], ymin=0, ymax=12.5,
           colors="grey", linestyles="--", label="Komplexere Simulation")

# --- Zweite Achse: CO2 Netto ---
ax2 = ax1.twinx()
ax2.plot(co2_run_number[1:], data_end_para['Massentrom CO2'][1:] - co2_run_number[1:],
         color=color5, label=r'CO$_2$ Netto', marker='o')
ax2.set_ylabel(r'CO$_2$ Bilanz (Ausstoß - Eintrag) [kg/s]', color=color5)
ax2.tick_params(axis='y', labelcolor=color5)

# --- gemeinsame Legende ---
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

plt.tight_layout()
plt.savefig('Bilder/Parameterstudie_CO2_Bilanz.png',
            dpi=300, bbox_inches='tight')
plt.close("all")

# Plot für Temperaturen 
plt.figure(figsize=(10,6))
plt.plot(co2_run_number[1:], data_end_para[' Surface_temperature_PFRC2_end_point_(K)'][1:], color=color1, label='Minimaltemperatur', marker = 's')
plt.plot(co2_run_number[1:], data_max_para[' Surface_temperature_PFRC2_max_point_(K)'][1:], color=color2, label='Maximaltemperatur', marker = 's') 
#plt.plot(co2_run_number[1:], data_min_para[' Surface_temperature_PFRC2_min_point_(K)'][1:], color=color3, label='Minimalwert', marker = 's')
plt.grid()
plt.xlabel(r'CO$_2$ Massenstrom am Einlass (kg/s)')
plt.ylabel('Temperatur (K)')
plt.vlines(x=[0.05455], ymin = 1350, ymax = 2200, colors = "grey", linestyles="--", label="Komplexere Simulation")
plt.legend()
plt.savefig('Bilder/Parameterstudie_CO2_Temperaturen.png', dpi=300, bbox_inches='tight')
plt.close("all")

#%% ermitteln CH4-Schlupf
plt.plot(co2_run_number[1:], data_end_para[' Mole_fraction_CH4_PFRC2_end_point_()'][1:], marker = 's')
plt.grid()
plt.xlabel(r'CO$_2$ Massenstrom am Einlass (kg/s)')
plt.ylabel(r'Molenbruch CH$_4$ am Ende des Reaktors')
plt.vlines(x=[0.05455], ymin = 0, ymax = data_end_para[' Mole_fraction_CH4_PFRC2_end_point_()'][1:].max(), colors = "grey", linestyles="--", label="Komplexere Simulation")
#plt.show()
plt.savefig('Bilder/Parameterstudie_CO2_CH4_Schlupf.png', dpi=300)