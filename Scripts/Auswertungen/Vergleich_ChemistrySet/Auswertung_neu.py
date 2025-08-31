import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.ticker import FuncFormatter
import seaborn as sb  # type: ignore

import os 
import shutil 
from collections import OrderedDict


from Latex_table import df_to_table

#%% Farben für die Plots definieren
# Diese Farben können angepasst werden, um die Lesbarkeit zu verbessern

color1 = '#1f77b4'
color2 = '#2ca02c'
color3 = "#3b3b3b"
color4 = '#d62728'

color1 = '#4878A8'
color2 = '#7E9680'
color3 = '#B3B3B3'
color4 = '#BC6C25'
color5 = "#960B0B" 
color6 = "#077B1A"

colors = [color1,color2,color3,color4,color5, color6]
#colors = sb.color_palette("muted", 5)

# Set the working directory to the location of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
#%% Funktionen 
def remove_water_and_normalize_topn(series,n):
    # Wasser entfernen
    ohne_wasser = series[~series.index.str.contains("H2O")]
    # Normalisieren
    normiert = ohne_wasser / ohne_wasser.sum()
    # Top 10 auswählen
    top10 = normiert.sort_values(ascending=False).head(n)
    return top10

def get_normalized_composition_strict(df, column_map):
    comp = {key: df.iloc[-1][col] for key, col in column_map.items() if col in df.columns}
    # Optional: Wasser entfernen
    if "H2O" in comp:
        del comp["H2O"]
    comp_series = pd.Series(comp)
    comp_series = comp_series / comp_series.sum()
    return comp_series
#%% Experimentelle Daten 
exp_data = {
    "kein CO2 Exp": [0.599,0.341,0.007,0.048],
    "CO2 Exp": [0.416,0.424,0.0012,0.151]
}

df_exp_data = pd.DataFrame(exp_data, index = ["H2", "CO", "CH4", "CO2"])
#%% Einlesen der Dataframes 
df_co2_aramco_pfr = pd.read_excel('Daten/Aramco_CO2.xlsm', sheet_name='2.soln_no_1_PFRC2', decimal=',')
df_co2_gri_pfr = pd.read_excel('Daten/GRI_CO2.xlsm', sheet_name= '2.soln_no_1_PFRC2', decimal=',')
df_co2_atr_pfr = pd.read_excel('Daten/ATR_CO2.xlsm', sheet_name= '2.soln_no_1_PFRC2', decimal=',')
df_co2_nuig_pfr = pd.read_excel('Daten/NUIG_CO2.xlsm', sheet_name= '2.soln_no_1_PFRC2', decimal=',')
df_co2_smoke_pfr = pd.read_excel('Daten/OpenSmoke_CO2.xlsm', sheet_name= '2.soln_no_1_PFRC2', decimal=',')

df_no_co2_aramco_pfr = pd.read_excel('Daten/Aramco_keinCO2.xlsm', sheet_name='2.soln_no_1_PFRC2', decimal=',')
df_no_co2_gri_pfr = pd.read_excel('Daten/GRI_keinCO2.xlsm', sheet_name= '2.soln_no_1_PFRC2', decimal=',')
df_no_co2_atr_pfr = pd.read_excel('Daten/ATR_keinCO2.xlsm', sheet_name= '2.soln_no_1_PFRC2', decimal=',')
df_no_co2_nuig_pfr = pd.read_excel('Daten/NUIG_keinCO2.xlsm', sheet_name= '2.soln_no_1_PFRC2', decimal=',')
df_no_co2_smoke_pfr = pd.read_excel('Daten/OpenSmoke_keinCO2.xlsm', sheet_name= '2.soln_no_1_PFRC2', decimal=',')
#%% Arrays CO2
dist_pfr_aramco_co2 = df_co2_aramco_pfr['Distance_PFRC2_(m)']
dist_pfr_gri_co2 = df_co2_gri_pfr['Distance_PFRC2_(m)']
dist_pfr_atr_co2 = df_co2_atr_pfr['Distance_PFRC2_(m)']
dist_pfr_nuig_co2 = df_co2_nuig_pfr['Distance_PFRC2_(m)']
dist_pfr_smoke_co2 = df_co2_nuig_pfr['Distance_PFRC2_(m)']

temp_pfr_aramco_co2 = df_co2_aramco_pfr[' Temperature_PFRC2_(K)']
temp_pfr_gri_co2 = df_co2_gri_pfr[' Temperature_PFRC2_(K)']
temp_pfr_atr_co2 = df_co2_atr_pfr[' Temperature_PFRC2_(K)']
temp_pfr_nuig_co2 = df_co2_nuig_pfr[' Temperature_PFRC2_(K)']
temp_pfr_smoke_co2 = df_co2_nuig_pfr[' Temperature_PFRC2_(K)']

x_H2_pfr_aramco_co2 = df_co2_aramco_pfr[' Mole_fraction_H2_PFRC2_()']
x_H2_pfr_gri_co2 = df_co2_gri_pfr[' Mole_fraction_H2_PFRC2_()']
x_H2_pfr_atr_co2 = df_co2_atr_pfr[' Mole_fraction_H2_PFRC2_()']
x_H2_pfr_nuig_co2 = df_co2_nuig_pfr[' Mole_fraction_H2_PFRC2_()']
x_H2_pfr_smoke_co2 = df_co2_nuig_pfr[' Mole_fraction_H2_PFRC2_()']

x_H2O_pfr_aramco_co2 = df_co2_aramco_pfr[' Mole_fraction_H2O_PFRC2_()']
x_H2O_pfr_gri_co2 = df_co2_gri_pfr[' Mole_fraction_H2O_PFRC2_()']
x_H2O_pfr_atr_co2 = df_co2_atr_pfr[' Mole_fraction_H2O_PFRC2_()']
x_H2O_pfr_nuig_co2 = df_co2_nuig_pfr[' Mole_fraction_H2O_PFRC2_()']
x_H2O_pfr_smoke_co2 = df_co2_nuig_pfr[' Mole_fraction_H2O_PFRC2_()']

x_CO_pfr_aramco_co2 = df_co2_aramco_pfr[' Mole_fraction_CO_PFRC2_()']
x_CO_pfr_gri_co2 = df_co2_gri_pfr[' Mole_fraction_CO_PFRC2_()']
x_CO_pfr_atr_co2 = df_co2_atr_pfr[' Mole_fraction_CO_PFRC2_()']
x_CO_pfr_nuig_co2 = df_co2_nuig_pfr[' Mole_fraction_CO_PFRC2_()']
x_CO_pfr_smoke_co2 = df_co2_nuig_pfr[' Mole_fraction_CO_PFRC2_()']

x_CO2_pfr_aramco_co2 = df_co2_aramco_pfr[' Mole_fraction_CO2_PFRC2_()']
x_CO2_pfr_gri_co2 = df_co2_gri_pfr[' Mole_fraction_CO2_PFRC2_()']
x_CO2_pfr_atr_co2 = df_co2_atr_pfr[' Mole_fraction_CO2_PFRC2_()']
x_CO2_pfr_nuig_co2 = df_co2_nuig_pfr[' Mole_fraction_CO2_PFRC2_()']
x_CO2_pfr_smoke_co2 = df_co2_nuig_pfr[' Mole_fraction_CO2_PFRC2_()']

x_CH4_pfr_aramco_co2 = df_co2_aramco_pfr[' Mole_fraction_CH4_PFRC2_()']
x_CH4_pfr_gri_co2 = df_co2_gri_pfr[' Mole_fraction_CH4_PFRC2_()']
x_CH4_pfr_atr_co2 = df_co2_atr_pfr[' Mole_fraction_CH4_PFRC2_()']
x_CH4_pfr_nuig_co2 = df_co2_nuig_pfr[' Mole_fraction_CH4_PFRC2_()']
x_CH4_pfr_smoke_co2 = df_co2_nuig_pfr[' Mole_fraction_CH4_PFRC2_()']

x_unburned_pfr_aramco_co2 = df_co2_aramco_pfr[' Unburned_hydrocarbons_PFRC2_(ppm)']
x_unburned_pfr_gri_co2 = df_co2_gri_pfr[' Unburned_hydrocarbons_PFRC2_(ppm)']
x_unburned_pfr_atr_co2 = df_co2_atr_pfr[' Unburned_hydrocarbons_PFRC2_(ppm)']
x_unburned_pfr_nuig_co2 = df_co2_nuig_pfr[' Unburned_hydrocarbons_PFRC2_(ppm)']
x_unburned_pfr_smoke_co2 = df_co2_nuig_pfr[' Unburned_hydrocarbons_PFRC2_(ppm)']

x_test_pfr_aramco_co2 = df_co2_aramco_pfr[' Mole_fraction_N2_PFRC2_()']
x_test_pfr_gri_co2 = df_co2_gri_pfr[' Mole_fraction_N2_PFRC2_()']
x_test_pfr_atr_co2 = df_co2_atr_pfr[' Mole_fraction_N2_PFRC2_()']
x_test_pfr_nuig_co2 = df_co2_nuig_pfr[' Mole_fraction_N2_PFRC2_()']
x_test_pfr_smoke_co2 = df_co2_nuig_pfr[' Mole_fraction_N2_PFRC2_()']#%% Arrays kein CO2

dist_pfr_aramco_no_co2 = df_no_co2_aramco_pfr['Distance_PFRC2_(m)']
dist_pfr_gri_no_co2 = df_no_co2_gri_pfr['Distance_PFRC2_(m)']
dist_pfr_atr_no_co2 = df_no_co2_atr_pfr['Distance_PFRC2_(m)']
dist_pfr_nuig_no_co2 = df_no_co2_nuig_pfr['Distance_PFRC2_(m)']
dist_pfr_smoke_no_co2 = df_no_co2_nuig_pfr['Distance_PFRC2_(m)']

temp_pfr_aramco_no_co2 = df_no_co2_aramco_pfr[' Temperature_PFRC2_(K)']
temp_pfr_gri_no_co2 = df_no_co2_gri_pfr[' Temperature_PFRC2_(K)']
temp_pfr_atr_no_co2 = df_no_co2_atr_pfr[' Temperature_PFRC2_(K)']
temp_pfr_nuig_no_co2 = df_no_co2_nuig_pfr[' Temperature_PFRC2_(K)']
temp_pfr_smoke_no_co2 = df_no_co2_nuig_pfr[' Temperature_PFRC2_(K)']

x_H2_pfr_aramco_no_co2 = df_no_co2_aramco_pfr[' Mole_fraction_H2_PFRC2_()']
x_H2_pfr_gri_no_co2 = df_no_co2_gri_pfr[' Mole_fraction_H2_PFRC2_()']
x_H2_pfr_atr_no_co2 = df_no_co2_atr_pfr[' Mole_fraction_H2_PFRC2_()']
x_H2_pfr_nuig_no_co2 = df_no_co2_nuig_pfr[' Mole_fraction_H2_PFRC2_()']
x_H2_pfr_smoke_no_co2 = df_no_co2_nuig_pfr[' Mole_fraction_H2_PFRC2_()']

x_H2O_pfr_aramco_no_co2 = df_no_co2_aramco_pfr[' Mole_fraction_H2O_PFRC2_()']
x_H2O_pfr_gri_no_co2 = df_no_co2_gri_pfr[' Mole_fraction_H2O_PFRC2_()']
x_H2O_pfr_atr_no_co2 = df_no_co2_atr_pfr[' Mole_fraction_H2O_PFRC2_()']
x_H2O_pfr_nuig_no_co2 = df_no_co2_nuig_pfr[' Mole_fraction_H2O_PFRC2_()']
x_H2O_pfr_smoke_no_co2 = df_no_co2_nuig_pfr[' Mole_fraction_H2O_PFRC2_()']

x_CO_pfr_aramco_no_co2 = df_no_co2_aramco_pfr[' Mole_fraction_CO_PFRC2_()']
x_CO_pfr_gri_no_co2 = df_no_co2_gri_pfr[' Mole_fraction_CO_PFRC2_()']
x_CO_pfr_atr_no_co2 = df_no_co2_atr_pfr[' Mole_fraction_CO_PFRC2_()']
x_CO_pfr_nuig_no_co2 = df_no_co2_nuig_pfr[' Mole_fraction_CO_PFRC2_()']
x_CO_pfr_smoke_no_co2 = df_no_co2_nuig_pfr[' Mole_fraction_CO_PFRC2_()']

x_CO2_pfr_aramco_no_co2 = df_no_co2_aramco_pfr[' Mole_fraction_CO2_PFRC2_()']
x_CO2_pfr_gri_no_co2 = df_no_co2_gri_pfr[' Mole_fraction_CO2_PFRC2_()']
x_CO2_pfr_atr_no_co2 = df_no_co2_atr_pfr[' Mole_fraction_CO2_PFRC2_()']
x_CO2_pfr_nuig_no_co2 = df_no_co2_nuig_pfr[' Mole_fraction_CO2_PFRC2_()']
x_CO2_pfr_smoke_no_co2 = df_no_co2_nuig_pfr[' Mole_fraction_CO2_PFRC2_()']

x_CH4_pfr_aramco_no_co2 = df_no_co2_aramco_pfr[' Mole_fraction_CH4_PFRC2_()']
x_CH4_pfr_gri_no_co2 = df_no_co2_gri_pfr[' Mole_fraction_CH4_PFRC2_()']
x_CH4_pfr_atr_no_co2 = df_no_co2_atr_pfr[' Mole_fraction_CH4_PFRC2_()']
x_CH4_pfr_nuig_no_co2 = df_no_co2_nuig_pfr[' Mole_fraction_CH4_PFRC2_()']
x_CH4_pfr_smoke_no_co2 = df_no_co2_nuig_pfr[' Mole_fraction_CH4_PFRC2_()']

x_unburned_pfr_aramco_no_co2 = df_no_co2_aramco_pfr[' Unburned_hydrocarbons_PFRC2_(ppm)']
x_unburned_pfr_gri_no_co2 = df_no_co2_gri_pfr[' Unburned_hydrocarbons_PFRC2_(ppm)']
x_unburned_pfr_atr_no_co2 = df_no_co2_atr_pfr[' Unburned_hydrocarbons_PFRC2_(ppm)']
x_unburned_pfr_nuig_no_co2 = df_no_co2_nuig_pfr[' Unburned_hydrocarbons_PFRC2_(ppm)']
x_unburned_pfr_smoke_no_co2 = df_no_co2_nuig_pfr[' Unburned_hydrocarbons_PFRC2_(ppm)']

x_test_pfr_aramco_no_co2 = df_no_co2_aramco_pfr[' Mole_fraction_N2_PFRC2_()']
x_test_pfr_gri_no_co2 = df_no_co2_gri_pfr[' Mole_fraction_N2_PFRC2_()']
x_test_pfr_atr_no_co2 = df_no_co2_atr_pfr[' Mole_fraction_N2_PFRC2_()']
x_test_pfr_nuig_no_co2 = df_no_co2_nuig_pfr[' Mole_fraction_N2_PFRC2_()']
x_test_pfr_smoke_no_co2 = df_no_co2_nuig_pfr[' Mole_fraction_N2_PFRC2_()']#%% Plots 

fig, axs = plt.subplots(2, 2, figsize=(7, 7))

# H2
axs[0, 0].plot(dist_pfr_aramco_no_co2, x_H2_pfr_aramco_no_co2, label='aramco', color=colors[0])
axs[0, 0].plot(dist_pfr_gri_no_co2, x_H2_pfr_gri_no_co2, label='gri', color=colors[1])
axs[0, 0].plot(dist_pfr_atr_no_co2, x_H2_pfr_atr_no_co2, label='atr', color=colors[2])
axs[0, 0].plot(dist_pfr_nuig_no_co2, x_H2_pfr_nuig_no_co2, label='nuig', color=colors[3])
axs[0, 0].plot(dist_pfr_smoke_no_co2, x_H2_pfr_smoke_no_co2, label='OpenSmoke', color=colors[4])
axs[0, 0].set_xlabel("Reaktorlänge (m)")
axs[0, 0].set_ylabel("Massenanteil H₂")
axs[0, 0].set_title("H₂")
axs[0, 0].grid()
axs[0, 0].legend()

# CH4
axs[0, 1].plot(dist_pfr_aramco_no_co2, x_CH4_pfr_aramco_no_co2, label='aramco', color=colors[0])
axs[0, 1].plot(dist_pfr_gri_no_co2, x_CH4_pfr_gri_no_co2, label='gri', color=colors[1])
axs[0, 1].plot(dist_pfr_atr_no_co2, x_CH4_pfr_atr_no_co2, label='atr', color=colors[2])
axs[0, 1].plot(dist_pfr_nuig_no_co2, x_CH4_pfr_nuig_no_co2, label='nuig', color=colors[3])
axs[0, 1].plot(dist_pfr_smoke_no_co2, x_CH4_pfr_smoke_no_co2, label='OpenSmoke', color=colors[4])
axs[0, 1].set_xlabel("Reaktorlänge (m)")
axs[0, 1].set_ylabel("Massenanteil CH₄")
axs[0, 1].set_title("CH₄")
axs[0, 1].grid()
axs[0, 1].legend()

# CO₂
axs[1, 0].plot(dist_pfr_aramco_no_co2, x_CO2_pfr_aramco_no_co2, label='aramco', color=colors[0])
axs[1, 0].plot(dist_pfr_gri_no_co2, x_CO2_pfr_gri_no_co2, label='gri', color=colors[1])
axs[1, 0].plot(dist_pfr_atr_no_co2, x_CO2_pfr_atr_no_co2, label='atr', color=colors[2])
axs[1, 0].plot(dist_pfr_nuig_no_co2, x_CO2_pfr_nuig_no_co2, label='nuig', color=colors[3])
axs[1, 0].plot(dist_pfr_smoke_no_co2, x_CO2_pfr_smoke_no_co2, label='OpenSmoke', color=colors[4])
axs[1, 0].set_xlabel("Reaktorlänge (m)")
axs[1, 0].set_ylabel("Massenanteil CO₂")
axs[1, 0].set_title("CO₂")
axs[1, 0].grid()
axs[1, 0].legend()

# CO
axs[1, 1].plot(dist_pfr_aramco_no_co2, x_CO_pfr_aramco_no_co2, label='aramco', color=colors[0])
axs[1, 1].plot(dist_pfr_gri_no_co2, x_CO_pfr_gri_no_co2, label='gri', color=colors[1])
axs[1, 1].plot(dist_pfr_atr_no_co2, x_CO_pfr_atr_no_co2, label='atr', color=colors[2])
axs[1, 1].plot(dist_pfr_nuig_no_co2, x_CO_pfr_nuig_no_co2, label='nuig', color=colors[3])
axs[1, 1].plot(dist_pfr_smoke_no_co2, x_CO_pfr_smoke_no_co2, label='OpenSmoke', color=colors[4])
axs[1, 1].set_xlabel("Reaktorlänge (m)")
axs[1, 1].set_ylabel("Massenanteil CO")
axs[1, 1].set_title("CO")
axs[1, 1].grid()
axs[1, 1].legend()

# --- gemeinsame Legende erstellen ---
# alle Handles/Labels von allen Achsen einsammeln
handles, labels = [], []
for ax in axs.flat:
    h, l = ax.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)
    ax.legend().remove()   # lokale Legende ausblenden

# duplikate vermeiden (nur einmal je Mechanismus)
by_label = OrderedDict(zip(labels, handles))

# zentrale Legende unten oder oben
fig.legend(by_label.values(), by_label.keys(),
           loc="lower center", ncol=4, frameon=False)

plt.tight_layout(rect=[0,0.05,1,1])  # Platz für Legende oben lassen
plt.savefig("img/H2_CH4_CO_CO2_keinCO2.png", dpi=300)
plt.close()

fig, axs = plt.subplots(2, 2, figsize=(7, 7))

# CO
axs[0, 0].plot(dist_pfr_aramco_co2, x_CO_pfr_aramco_co2, label='aramco', color=colors[0])
axs[0, 0].plot(dist_pfr_gri_co2, x_CO_pfr_gri_co2, label='gri', color=colors[1])
axs[0, 0].plot(dist_pfr_atr_co2, x_CO_pfr_atr_co2, label='atr', color=colors[2])
axs[0, 0].plot(dist_pfr_nuig_co2, x_CO_pfr_nuig_co2, label='nuig', color=colors[3])
axs[0, 0].plot(dist_pfr_smoke_co2, x_CO_pfr_smoke_co2, label='OpenSmoke', color=colors[4])
axs[0, 0].set_xlabel("Reaktorlänge (m)")
axs[0, 0].set_ylabel("Massenanteil CO")
axs[0, 0].set_title("CO")
axs[0, 0].grid()
axs[0, 0].legend()

# CO2
axs[0, 1].plot(dist_pfr_aramco_co2, x_CO2_pfr_aramco_co2, label='aramco', color=colors[0])
axs[0, 1].plot(dist_pfr_gri_co2, x_CO2_pfr_gri_co2, label='gri', color=colors[1])
axs[0, 1].plot(dist_pfr_atr_co2, x_CO2_pfr_atr_co2, label='atr', color=colors[2])
axs[0, 1].plot(dist_pfr_nuig_co2, x_CO2_pfr_nuig_co2, label='nuig', color=colors[3])
axs[0, 1].plot(dist_pfr_smoke_co2, x_CO2_pfr_smoke_co2, label='OpenSmoke', color=colors[4])
axs[0, 1].set_xlabel("Reaktorlänge (m)")
axs[0, 1].set_ylabel("Massenanteil CO₂")
axs[0, 1].set_title("CO₂")
axs[0, 1].grid()
axs[0, 1].legend()

# H2
axs[1, 0].plot(dist_pfr_aramco_co2, x_H2_pfr_aramco_co2, label='aramco', color=colors[0])
axs[1, 0].plot(dist_pfr_gri_co2, x_H2_pfr_gri_co2, label='gri', color=colors[1])
axs[1, 0].plot(dist_pfr_atr_co2, x_H2_pfr_atr_co2, label='atr', color=colors[2])
axs[1, 0].plot(dist_pfr_nuig_co2, x_H2_pfr_nuig_co2, label='nuig', color=colors[3])
axs[1, 0].plot(dist_pfr_smoke_co2, x_H2_pfr_smoke_co2, label='OpenSmoke', color=colors[4])
axs[1, 0].set_xlabel("Reaktorlänge (m)")
axs[1, 0].set_ylabel("Massenanteil H₂")
axs[1, 0].set_title("H₂")
axs[1, 0].grid()
axs[1, 0].legend()

# CH4
axs[1, 1].plot(dist_pfr_aramco_co2, x_CH4_pfr_aramco_co2, label='aramco', color=colors[0])
axs[1, 1].plot(dist_pfr_gri_co2, x_CH4_pfr_gri_co2, label='gri', color=colors[1])
axs[1, 1].plot(dist_pfr_atr_co2, x_CH4_pfr_atr_co2, label='atr', color=colors[2])
axs[1, 1].plot(dist_pfr_nuig_co2, x_CH4_pfr_nuig_co2, label='nuig', color=colors[3])
axs[1, 1].plot(dist_pfr_smoke_co2, x_CH4_pfr_smoke_co2, label='OpenSmoke', color=colors[4])
axs[1, 1].set_xlabel("Reaktorlänge (m)")
axs[1, 1].set_ylabel("Massenanteil CH₄")
axs[1, 1].set_title("CH₄")
axs[1, 1].grid()
axs[1, 1].legend()

# --- gemeinsame Legende erstellen ---
# alle Handles/Labels von allen Achsen einsammeln
handles, labels = [], []
for ax in axs.flat:
    h, l = ax.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)
    ax.legend().remove()   # lokale Legende ausblenden

# duplikate vermeiden (nur einmal je Mechanismus)
by_label = OrderedDict(zip(labels, handles))

# zentrale Legende unten oder oben
fig.legend(by_label.values(), by_label.keys(),
           loc="lower center", ncol=4, frameon=False)

plt.tight_layout(rect=[0,0.05,1,1])  # Platz für Legende oben lassen
plt.savefig("img/H2_CH4_CO_CO2.png", dpi=300)
plt.close()

#%% DF für Datenvergleich 
# Schritt 1: Spalten mit 'mole_fraction' im Namen auswählen
last_row_no_co2 = df_no_co2_gri_pfr.iloc[-1][[col for col in df_no_co2_gri_pfr.columns if "Mole_fraction" in col]]
last_row_co2 = df_co2_gri_pfr.iloc[-1][[col for col in df_co2_gri_pfr.columns if "Mole_fraction" in col]]
# Funktion anwenden
no_co2_normiert = remove_water_and_normalize_topn(last_row_no_co2, 5)
co2_normiert = remove_water_and_normalize_topn(last_row_co2, 5)

species_column_map = {
    "H2":   " Mole_fraction_H2_PFRC2_()",
    "CO":   " Mole_fraction_CO_PFRC2_()",
    "CH4":  " Mole_fraction_CH4_PFRC2_()",
    "CO2":  " Mole_fraction_CO2_PFRC2_()",
    "H2O":  " Mole_fraction_H2O_PFRC2_()",
}

# Zusammensetzungen extrahieren & normalisieren – überall dasselbe Mapping!
gri_co2    = get_normalized_composition_strict(df_co2_gri_pfr, species_column_map)
aramco_co2 = get_normalized_composition_strict(df_co2_aramco_pfr, species_column_map)
nuig_co2   = get_normalized_composition_strict(df_co2_nuig_pfr, species_column_map)
atr_co2    = get_normalized_composition_strict(df_co2_atr_pfr, species_column_map)
smoke_co2    = get_normalized_composition_strict(df_co2_smoke_pfr, species_column_map)

gri_no_co2    = get_normalized_composition_strict(df_no_co2_gri_pfr, species_column_map)
aramco_no_co2 = get_normalized_composition_strict(df_no_co2_aramco_pfr, species_column_map)
nuig_no_co2   = get_normalized_composition_strict(df_no_co2_nuig_pfr, species_column_map)
atr_no_co2    = get_normalized_composition_strict(df_no_co2_atr_pfr, species_column_map)
smoke_no_co2    = get_normalized_composition_strict(df_no_co2_smoke_pfr, species_column_map)

vergleich_species = ["H2", "CO", "CH4", "CO2"]

df_exp_data["GRI_noCO2"]     = gri_no_co2.reindex(vergleich_species)
df_exp_data["GRI_CO2"]       = gri_co2.reindex(vergleich_species)

df_exp_data["ARAMCO_noCO2"]  = aramco_no_co2.reindex(vergleich_species)
df_exp_data["ARAMCO_CO2"]    = aramco_co2.reindex(vergleich_species)

df_exp_data["ATR_noCO2"]     = atr_no_co2.reindex(vergleich_species)
df_exp_data["ATR_CO2"]       = atr_co2.reindex(vergleich_species)

df_exp_data["NUIG_noCO2"]    = nuig_no_co2.reindex(vergleich_species)
df_exp_data["NUIG_CO2"]      = nuig_co2.reindex(vergleich_species)

df_exp_data["Smoke_noCO2"]    = smoke_no_co2.reindex(vergleich_species)
df_exp_data["Smoke_CO2"]      = smoke_co2.reindex(vergleich_species)

df_to_table(df_exp_data, 
            columns = ["kein CO2 Exp", "GRI_noCO2", "ARAMCO_noCO2", "ATR_noCO2", "NUIG_noCO2", "Smoke_noCO2"],
            rounding = [3,3,3,3,3,3],
            latex = True,
            filepath = "Tabellen/Tabelle_Vergleich_keinCO2.tex",
            decimal_sep = ","
            )

df_to_table(df_exp_data, 
            columns = ["CO2 Exp", "GRI_CO2", "ARAMCO_CO2", "ATR_CO2", "NUIG_CO2", "Smoke_CO2"],
            rounding = [3,3,3,3,3,3],
            latex = True,
            filepath = "Tabellen/Tabelle_Vergleich_CO2.tex",
            decimal_sep = ","
            )


# Spalten gruppieren: jeweils "kein CO2" und "CO2" separat plotten
gruppen = [
    ["kein CO2 Exp", "GRI_noCO2", "ARAMCO_noCO2", "ATR_noCO2", "NUIG_noCO2", "Smoke_noCO2"],
    ["CO2 Exp",      "GRI_CO2",   "ARAMCO_CO2",   "ATR_CO2",   "NUIG_CO2", "Smoke_CO2"]
]
titel = ["Vergleich ohne CO₂", "Vergleich mit CO₂"]

k = 10.0            # Skalierungsfaktor
scale_species = ["CH4"]   # nur CH4 wird skaliert

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

for i, group in enumerate(gruppen):
    ax = axes[i]
    df_plot = df_exp_data[group].copy()

    # CH4 skalieren (nur linke Achse)
    if "CH4" in df_plot.index:
        df_plot.loc["CH4"] = df_plot.loc["CH4"] * k

    # Balken zeichnen (linke Achse)
    df_plot.plot(kind="bar", ax=ax, color=colors)
    ax.set_title(titel[i])
    ax.set_xlabel("Spezies")
    ax.set_ylabel(f"Molenbruch (CH₄ ×{int(k)}, alle anderen original)")
    ax.grid(axis='y', linestyle=':')
    ax.legend(loc="upper right")

    # rechte Achse zeigt Originalwerte
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(ax.get_yticks())
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v/k:.3f}"))
    ax2.set_ylabel("Molenbruch (Originalwerte CH₄)")

plt.tight_layout()
plt.savefig("img/vergleich_Experimentaldaten_scaled_CH4.png", dpi=300)
plt.close()

#%% Temperatur und CH4 Plot 
fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

# Ohne CO₂ (links)
ax1 = axs[0]
ax2 = ax1.twinx()  # zweite y-Achse

ax1.plot(dist_pfr_aramco_no_co2, temp_pfr_aramco_no_co2, color=colors[0], label='aramco, Temperatur')
ax1.plot(dist_pfr_gri_no_co2, temp_pfr_gri_no_co2, color=colors[1], label='gri, Temperatur')
ax1.plot(dist_pfr_atr_no_co2, temp_pfr_atr_no_co2, color=colors[2], label='atr, Temperatur')
ax1.plot(dist_pfr_nuig_no_co2, temp_pfr_nuig_no_co2, color=colors[3], label='nuig, Temperatur')
ax1.plot(dist_pfr_smoke_no_co2, temp_pfr_smoke_no_co2, color=colors[4], label='smoke, Temperatur')
ax1.set_ylabel("Temperatur (K)")

ax2.plot(dist_pfr_aramco_no_co2, x_CH4_pfr_aramco_no_co2, color=colors[0], label='aramco, CH4', linestyle = "--")
ax2.plot(dist_pfr_gri_no_co2, x_CH4_pfr_gri_no_co2, color=colors[1], label='gri, CH4', linestyle = "--")
ax2.plot(dist_pfr_atr_no_co2, x_CH4_pfr_atr_no_co2, color=colors[2], label='atr, CH4', linestyle = "--")
ax2.plot(dist_pfr_nuig_no_co2, x_CH4_pfr_nuig_no_co2, color=colors[3], label='nuig, CH4', linestyle = "--")
ax2.plot(dist_pfr_smoke_no_co2, x_CH4_pfr_smoke_no_co2, color=colors[4], label='smoke, CH4', linestyle = "--")
ax2.set_ylabel("CH$_4$ Molenbruch")

ax1.set_title("ohne CO₂")
ax1.set_xlabel("Reaktorlänge (m)")
ax1.grid()
# Legende beider Achsen zusammenführen
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# Mit CO₂ (rechts)
ax3 = axs[1]
ax4 = ax3.twinx()

ax3.plot(dist_pfr_aramco_co2, temp_pfr_aramco_co2, color=colors[0], label='aramco, Temperatur')
ax3.plot(dist_pfr_gri_co2, temp_pfr_gri_co2, color=colors[1], label='gri, Temperatur')
ax3.plot(dist_pfr_atr_co2, temp_pfr_atr_co2, color=colors[2], label='atr, Temperatur')
ax3.plot(dist_pfr_nuig_co2, temp_pfr_nuig_co2, color=colors[3], label='nuig, Temperatur')
ax3.plot(dist_pfr_smoke_co2, temp_pfr_smoke_co2, color=colors[4], label='smoke, Temperatur')
ax3.set_ylabel("Temperatur (K)")

ax4.plot(dist_pfr_aramco_co2, x_CH4_pfr_aramco_co2, color=colors[0], label='aramco, CH4', linestyle = "--")
ax4.plot(dist_pfr_gri_co2, x_CH4_pfr_gri_co2, color=colors[1], label='gri, CH4', linestyle = "--")
ax4.plot(dist_pfr_atr_co2, x_CH4_pfr_atr_co2, color=colors[2], label='atr, CH4', linestyle = "--")
ax4.plot(dist_pfr_nuig_co2, x_CH4_pfr_nuig_co2, color=colors[3], label='nuig, CH4', linestyle = "--")
ax4.plot(dist_pfr_smoke_co2, x_CH4_pfr_smoke_co2, color=colors[4], label='smoke, CH4', linestyle = "--")
ax4.set_ylabel("CH$_4$ Molenbruch")

ax3.set_title("mit CO₂")
ax3.set_xlabel("Reaktorlänge (m)")
ax3.grid()
lines3, labels3 = ax3.get_legend_handles_labels()
lines4, labels4 = ax4.get_legend_handles_labels()
ax3.legend(lines3 + lines4, labels3 + labels4, loc='upper right')

plt.tight_layout()
plt.savefig("img/Temp_CH4.png", dpi = 300)

#%% Kopieren zu LaTeX 
# Relativer Quell- und Zielpfad
src_folder = "img"
dst_folder = os.path.join("..","..","..", "LaTeX", "img_py")  # z. B. ../Bericht/Bilder

# Zielordner erstellen, falls nicht vorhanden
os.makedirs(dst_folder, exist_ok=True)

# Inhalt von img nach ../Bericht/Bilder kopieren
shutil.copytree(src_folder, dst_folder, dirs_exist_ok=True)

plt.close("all")

#plt.plot(dist_pfr_nuig_no_co2, temp_pfr_nuig_no_co2, color=colors[3], label='nuig, Temperatur')
#plt.plot(dist_pfr_smoke_no_co2, temp_pfr_smoke_no_co2, color=colors[4], label='smoke, Temperatur')
plt.plot(dist_pfr_nuig_no_co2, temp_pfr_nuig_no_co2 - temp_pfr_smoke_no_co2, color=colors[3], label='nuig, Temperatur')
plt.legend()
plt.grid()
plt.show()

print(temp_pfr_nuig_no_co2)
print(temp_pfr_smoke_no_co2)

print(temp_pfr_nuig_no_co2 == temp_pfr_smoke_no_co2)