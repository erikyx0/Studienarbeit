#%% start
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import unicodedata
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

import re 
import glob
import sys

# Test1
# Pfad zum Hauptordner hinzufügen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))

from colors import colors_new

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

data_end_para['H2/CO'] = data_end_para['Massentrom H2'] / data_end_para['Massentrom CO']

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
plt.vlines(x=[0.0842], ymin = 0, ymax = data_end_para[' Mole_fraction_CH4_PFRC2_end_point_()'][1:].max(), colors = "red", linestyles="--", label="Maximale Betriebsgrenze")
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
plt.vlines(x=[0.0842], ymin = 0, ymax = data_end_para[' Mole_fraction_CH4_PFRC2_end_point_()'][1:].max(), colors = "red", linestyles="--", label="Maximale Betriebsgrenze")
plt.legend()
plt.savefig('Bilder/Parameterstudie_CO2_Massenstrom_Ende.png', dpi=300, bbox_inches='tight')
plt.close("all")


# Testbestimmung CO2 Massenstrom Einlass 
fig, ax1 = plt.subplots(figsize=(7,5))

# --- Erste Achse: Verhältnis H2/CO ---
ax1.plot(co2_run_number[1:], data_end_para['H2/CO'][1:],
         color=color1, label=r'H$_2$/CO', marker='v')
ax1.set_xlabel(r'CO$_2$ Massenstrom am Einlass (kg/s)')
ax1.set_ylabel(r'Verhältnis H$_2$/CO am Ende des Reaktors', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True)

# vertikale Linie
ax1.vlines(x=[0.05455], ymin=0, ymax=2, colors="grey", linestyles="--", label="Komplexere Simulation")
ax1.vlines(x=[0.0842], ymin = 0, ymax = 2, colors = "red", linestyles="--", label="Maximale Betriebsgrenze")

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
plt.vlines(x=[0.0842], ymin = 0, ymax = data_end_para[' Mole_fraction_CH4_PFRC2_end_point_()'][1:].max(), colors = "red", linestyles="--", label="Maximale Betriebsgrenze")
plt.legend()
plt.savefig('Bilder/Parameterstudie_CO2_Temperaturen.png', dpi=300, bbox_inches='tight')
plt.close("all")

#%% ermitteln CH4-Schlupf


x = np.asarray(co2_run_number[1:])
y = np.asarray(data_end_para[' Mole_fraction_CH4_PFRC2_end_point_()'][1:])

# z.B. Farbe nach mittlerer Temperatur
Tmin  = np.asarray(data_end_para[' Surface_temperature_PFRC2_end_point_(K)'][1:])
Tmax  = np.asarray(data_max_para[' Surface_temperature_PFRC2_max_point_(K)'][1:])
Tmean = 0.5*(Tmin + Tmax)

# Segmente für farbige Linie
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
norm = plt.Normalize(Tmean.min(), Tmean.max())

fig, ax = plt.subplots(figsize=(8,5))

cmap_red_yellow_blue = LinearSegmentedColormap.from_list(
    "red_yellow_blue",
    ["blue", "#FFD900F1", "red"]
)


# farbige Linie
lc = LineCollection(segments, cmap=cmap_red_yellow_blue, norm=norm, linewidth=3)
lc.set_array(Tmean)
ax.add_collection(lc)


# Marker-Ebene (auf derselben Farbskala)
sc = ax.scatter(x, y, c=Tmean, cmap=cmap_red_yellow_blue, norm=norm,
                s=50, edgecolor=None, zorder=3)

# Farbskala
cb = plt.colorbar(lc, ax=ax)
cb.set_label('mittlere Temperatur T̄ [K]')

# Layout
ax.set_xlim(x.min(), x.max())
ax.set_ylim(0, y.max()*1.05)
ax.set_xlabel(r'CO$_2$ Massenstrom am Einlass (kg/s)')
ax.set_ylabel(r'Molenbruch CH$_4$ am Ende des Reaktors')
ax.grid(True, linestyle=':')

# vertikale Linien und Legende
ax.vlines([0.05455], ymin=0, ymax=y.max()*1.05, colors="grey", linestyles="--", label="Komplexere Simulation")
ax.vlines([0.0842],  ymin=0, ymax=y.max()*1.05, colors="red",  linestyles="--", label="Maximale Betriebsgrenze")
ax.legend(loc="best")
ax.set_ylim(-0.002)

plt.tight_layout()
plt.savefig('Bilder/Parameterstudie_CO2_CH4_Schlupf_colormap_marker.png', dpi=300, bbox_inches='tight')
# plt.show()

#%% zwei Plots Stoffmenge und Masse
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# Plot 1: Stoffmengen
ax[0].plot(co2_run_number[1:], data_end_para[' Mole_fraction_CO2_PFRC2_end_point_()'][1:], color=color1, label=r'CO$_2$', marker = 's')
ax[0].plot(co2_run_number[1:], data_end_para[' Mole_fraction_CO_PFRC2_end_point_()'][1:], color=color2, label=r'CO', marker = 's')
ax[0].plot(co2_run_number[1:], data_end_para[' Mole_fraction_H2_PFRC2_end_point_()'][1:], color=color4, label=r'H$_2$', marker = 's')
ax[0].plot(co2_run_number[1:], data_end_para[' Mole_fraction_H2O_PFRC2_end_point_()'][1:], color=color5, label=r'H$_2$O', marker = 's')
ax[0].set_xlabel(r'CO$_2$ Massenstrom am Einlass (kg/s)')
ax[0].set_ylabel("Stoffmengenanteil am Reaktorausgang")
ax[0].grid()


# Plot 2: Massenströme
ax[1].plot(co2_run_number[1:], data_end_para['Massentrom CO2'][1:], color=color1, label=r'CO$_2$', marker = 's')
ax[1].plot(co2_run_number[1:], data_end_para['Massentrom CO'][1:], color=color2, label=r'CO', marker = 's')
ax[1].plot(co2_run_number[1:], data_end_para['Massentrom H2'][1:], color=color4, label=r'H$_2$', marker = 's')
ax[1].plot(co2_run_number[1:], data_end_para['Massentrom H2O'][1:], color=color5, label=r'H$_2$O', marker = 's')
ax[1].set_xlabel(r'CO$_2$ Massenstrom am Einlass (kg/s)')
ax[1].set_ylabel("Massenanteil am Reaktorausgang")
ax[1].grid()

# === Gemeinsame Legende ===
handles, labels = [], []
for a in ax:
    h, l = a.get_legend_handles_labels()
    handles += h
    labels += l

from collections import OrderedDict
by_label = OrderedDict(zip(labels, handles))

# Legende UNTER der Grafik, außerhalb des Plotbereichs
leg = fig.legend(by_label.values(), by_label.keys(),
                 loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05), frameon=False)

# keine Layoutanpassung hier!
plt.tight_layout()

# beim Anzeigen wird’s in manchen Umgebungen abgeschnitten:
#plt.show()

# --- aber beim Speichern:
fig.savefig("Bilder/plots_mit_legende.png", bbox_inches='tight', bbox_extra_artists=(leg,))

#%% 3D Plot
from complex_plots import *
import re

# Farbverlauf Rot Gelb Blau


# Verlauf: Rot → Gelb → Blau
cmap_red_yellow_blue = LinearSegmentedColormap.from_list(
    "red_yellow_blue",
    ["blue", "#E9009FF2", "red"]
)

#plot_runs_3d_simple(list(dfs_by_run.values())[1:], cmap=cmap_red_yellow_blue, point_step=5, run_step=1)
