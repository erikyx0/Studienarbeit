import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

import os 

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


# Set the working directory to the location of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

df_aramco_pfr = pd.read_excel('Aramco.xlsm', sheet_name='2.soln_no_1_PFRC2', decimal=',')
df_gri_pfr = pd.read_excel('GRI.xlsm', sheet_name= '2.soln_no_1_PFRC2', decimal=',')
df_atr_pfr = pd.read_excel('ATR.xlsm', sheet_name= '2.soln_no_1_PFRC2', decimal=',')
df_nuig_pfr = pd.read_excel('NUIG.xlsm', sheet_name= '2.soln_no_1_PFRC2', decimal=',')

dist_pfr_aramco = df_aramco_pfr['Distance_PFRC2_(cm)']
dist_pfr_gri = df_gri_pfr['Distance_PFRC2_(cm)']
dist_pfr_atr = df_atr_pfr['Distance_PFRC2_(cm)']
dist_pfr_nuig = df_nuig_pfr['Distance_PFRC2_(cm)']

temp_pfr_aramco = df_aramco_pfr[' Temperature_PFRC2_(K)']
temp_pfr_gri = df_gri_pfr[' Temperature_PFRC2_(K)']
temp_pfr_atr = df_atr_pfr[' Temperature_PFRC2_(K)']
temp_pfr_nuig = df_nuig_pfr[' Temperature_PFRC2_(K)']

x_H2_pfr_aramco = df_aramco_pfr[' Mole_fraction_H2_PFRC2_()']
x_H2_pfr_gri = df_gri_pfr[' Mole_fraction_H2_PFRC2_()']
x_H2_pfr_atr = df_atr_pfr[' Mole_fraction_H2_PFRC2_()']
x_H2_pfr_nuig = df_nuig_pfr[' Mole_fraction_H2_PFRC2_()']

x_H2O_pfr_aramco = df_aramco_pfr[' Mole_fraction_H2O_PFRC2_()']
x_H2O_pfr_gri = df_gri_pfr[' Mole_fraction_H2O_PFRC2_()']
x_H2O_pfr_atr = df_atr_pfr[' Mole_fraction_H2O_PFRC2_()']
x_H2O_pfr_nuig = df_nuig_pfr[' Mole_fraction_H2O_PFRC2_()']

x_CO_pfr_aramco = df_aramco_pfr[' Mole_fraction_CO_PFRC2_()']
x_CO_pfr_gri = df_gri_pfr[' Mole_fraction_CO_PFRC2_()']
x_CO_pfr_atr = df_atr_pfr[' Mole_fraction_CO_PFRC2_()']
x_CO_pfr_nuig = df_nuig_pfr[' Mole_fraction_CO_PFRC2_()']

x_CO2_pfr_aramco = df_aramco_pfr[' Mole_fraction_CO2_PFRC2_()']
x_CO2_pfr_gri = df_gri_pfr[' Mole_fraction_CO2_PFRC2_()']
x_CO2_pfr_atr = df_atr_pfr[' Mole_fraction_CO2_PFRC2_()']
x_CO2_pfr_nuig = df_nuig_pfr[' Mole_fraction_CO2_PFRC2_()']

x_CH4_pfr_aramco = df_aramco_pfr[' Mole_fraction_CH4_PFRC2_()']
x_CH4_pfr_gri = df_gri_pfr[' Mole_fraction_CH4_PFRC2_()']
x_CH4_pfr_atr = df_atr_pfr[' Mole_fraction_CH4_PFRC2_()']
x_CH4_pfr_nuig = df_nuig_pfr[' Mole_fraction_CH4_PFRC2_()']

x_unburned_pfr_aramco = df_aramco_pfr[' Unburned_hydrocarbons_PFRC2_(ppm)']
x_unburned_pfr_gri = df_gri_pfr[' Unburned_hydrocarbons_PFRC2_(ppm)']
x_unburned_pfr_atr = df_atr_pfr[' Unburned_hydrocarbons_PFRC2_(ppm)']
x_unburned_pfr_nuig = df_nuig_pfr[' Unburned_hydrocarbons_PFRC2_(ppm)']

x_test_pfr_aramco = df_aramco_pfr[' Mole_fraction_N2_PFRC2_()']
x_test_pfr_gri = df_gri_pfr[' Mole_fraction_N2_PFRC2_()']
x_test_pfr_atr = df_atr_pfr[' Mole_fraction_N2_PFRC2_()']
x_test_pfr_nuig = df_nuig_pfr[' Mole_fraction_N2_PFRC2_()']

fig, ax1 = plt.subplots(figsize=(10, 6))

# Temperatur auf linker y-Achse
ax1.plot(dist_pfr_aramco, temp_pfr_aramco, label='Aramco_Temp', color=color1)
ax1.plot(dist_pfr_gri, temp_pfr_gri, label='GRI_Temp', color=color2)
ax1.plot(dist_pfr_atr, temp_pfr_atr, label='ATR_Temp', color=color3)
ax1.plot(dist_pfr_nuig, temp_pfr_nuig, label='NUIG_Temp', color=color4)
ax1.set_xlabel('Reaktorlänge [cm]')
ax1.set_ylabel('Temperatur [K]', color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:red')

# CH4-Verläufe auf rechter y-Achse
ax2 = ax1.twinx()
ax2.plot(dist_pfr_aramco, x_CH4_pfr_aramco, label=r'Aramco_CH$_4$', linestyle='--', color=color1)
ax2.plot(dist_pfr_gri, x_CH4_pfr_gri, label=r'GRI_CH$_4$', linestyle='--', color=color2)
ax2.plot(dist_pfr_atr, x_CH4_pfr_atr, label=r'ATR_CH$_4$', linestyle='--', color=color3)
ax2.plot(dist_pfr_nuig, x_CH4_pfr_nuig, label=r'NUIG_CH$_4$', linestyle='--', color=color4)


ax2.set_ylabel(r'x(CH$_4$) [%]', color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')

# Gemeinsame Legende
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.title(r'Temperatur- und CH$_4$-Verläufe im PFR')
plt.grid(True)
plt.tight_layout()
plt.savefig('img/Temp_und_CH4.png', dpi=300)
plt.close()

# Liniendiagramm erstellen (Test)
plt.figure(figsize=(10, 6))
plt.plot(dist_pfr_aramco, x_test_pfr_aramco, label='Aramco', color='blue')
plt.plot(dist_pfr_gri, x_test_pfr_gri, label='GRI', color='green')
plt.plot(dist_pfr_atr, x_test_pfr_atr, label='ATR', color='orange')
plt.plot(dist_pfr_nuig, x_test_pfr_nuig, label='NUIG', color='red')

plt.grid()
plt.legend()
plt.xlabel('Reaktorlänge [cm]')
plt.ylabel('Molefraktion')
plt.title('Molefraktionen im PFR - Aramco')
plt.tight_layout()
plt.grid()
#plt.show()
plt.close()

#%# Plots für die verschiedenen Molefraktionen erstellen
# Plots für H2 und H2O
fig1, axs1 = plt.subplots(1, 2, figsize=(12, 5))

# H2
axs1[0].plot(dist_pfr_aramco, x_H2_pfr_aramco, label='Aramco', color=color1, linewidth=2.5)
axs1[0].plot(dist_pfr_gri, x_H2_pfr_gri, label='GRI', color=color2, linewidth=2.5)
axs1[0].plot(dist_pfr_atr, x_H2_pfr_atr, label='ATR', color=color3, linewidth=2.5)
axs1[0].plot(dist_pfr_nuig, x_H2_pfr_nuig, label='NUIG', color=color4, linewidth=2.5)
axs1[0].set_title('H₂')
axs1[0].set_ylabel('Molefraktion')
axs1[0].set_xlabel('Reaktorlänge [cm]')
axs1[0].grid(True)
axs1[0].legend()

# H2O
axs1[1].plot(dist_pfr_aramco, x_H2O_pfr_aramco, label='Aramco', color=color1, linewidth=2.5)
axs1[1].plot(dist_pfr_gri, x_H2O_pfr_gri, label='GRI', color=color2, linewidth=2.5)
axs1[1].plot(dist_pfr_atr, x_H2O_pfr_atr, label='ATR', color=color3, linewidth=2.5)
axs1[1].plot(dist_pfr_nuig, x_H2O_pfr_nuig, label='NUIG', color=color4, linewidth=2.5)
axs1[1].set_title('H₂O')
axs1[1].set_xlabel('Reaktorlänge [cm]')
axs1[1].grid(True)
axs1[1].legend()

plt.tight_layout()
plt.savefig('img/Plot_H2_H2O.png', dpi=300)
plt.close()

# Plots für CO und CO2
fig2, axs2 = plt.subplots(1, 2, figsize=(12, 5))

# CO
axs2[0].plot(dist_pfr_aramco, x_CO_pfr_aramco, label='Aramco', color=color1, linewidth=2.5)
axs2[0].plot(dist_pfr_gri, x_CO_pfr_gri, label='GRI', color=color2, linewidth=2.5)
axs2[0].plot(dist_pfr_atr, x_CO_pfr_atr, label='ATR', color=color3, linewidth=2.5)
axs2[0].plot(dist_pfr_nuig, x_CO_pfr_nuig, label='NUIG', color=color4, linewidth=2.5)
axs2[0].set_title('CO')
axs2[0].set_ylabel('Molefraktion')
axs2[0].set_xlabel('Reaktorlänge [cm]')
axs2[0].grid(True)
axs2[0].legend()

# CO2
axs2[1].plot(dist_pfr_aramco, x_CO2_pfr_aramco, label='Aramco', color=color1, linewidth=2.5)
axs2[1].plot(dist_pfr_gri, x_CO2_pfr_gri, label='GRI', color=color2, linewidth=2.5)
axs2[1].plot(dist_pfr_atr, x_CO2_pfr_atr, label='ATR', color=color3, linewidth=2.5)
axs2[1].plot(dist_pfr_nuig, x_CO2_pfr_nuig, label='NUIG', color=color4, linewidth=2.5)
axs2[1].set_title('CO₂')
axs2[1].set_xlabel('Reaktorlänge [cm]')
axs2[1].grid(True)
axs2[1].legend()

plt.tight_layout()
plt.savefig('img/Plot_CO_CO2.png', dpi=300)
plt.close()

# Plots für Temperatur und unverbrannte Kohlenwasserstoffe
fig3, axs3 = plt.subplots(1, 2, figsize=(12, 5))
# Temperatur
axs3[0].plot(dist_pfr_aramco, temp_pfr_aramco, label='Aramco', color=color1, linewidth=2.5)
axs3[0].plot(dist_pfr_gri, temp_pfr_gri, label='GRI', color=color2, linewidth=2.5)
axs3[0].plot(dist_pfr_atr, temp_pfr_atr, label='ATR', color=color3, linewidth=2.5)
axs3[0].plot(dist_pfr_nuig, temp_pfr_nuig, label='NUIG', color=color4, linewidth=2.5)
axs3[0].set_title('Temperatur')
axs3[0].set_ylabel('Temperatur [K]')
axs3[0].set_xlabel('Reaktorlänge [cm]')
axs3[0].grid(True)
axs3[0].legend()

# unverbrannte Kohlenwasserstoffe
axs3[1].plot(dist_pfr_aramco, x_unburned_pfr_aramco/1e6, label='Aramco', color=color1, linewidth=2.5)
axs3[1].plot(dist_pfr_gri, x_unburned_pfr_gri/1e6, label='GRI', color=color2, linewidth=2.5)
axs3[1].plot(dist_pfr_atr, x_unburned_pfr_atr/1e6, label='ATR', color=color3, linewidth=2.5)
axs3[1].plot(dist_pfr_nuig, x_unburned_pfr_nuig/1e6, label='NUIG', color=color4, linewidth=2.5)
axs3[1].set_title('unverbrannte Kohlenwasserstoffe')
axs3[1].set_xlabel('Reaktorlänge [cm]')
axs3[1].set_ylabel('Molefraktion ')
axs3[1].grid(True)
axs3[1].legend()


plt.tight_layout()
plt.savefig('img/Temp_kohlenwasserstoffe.png', dpi = 300)
plt.close()

#%% Abgaszusammensetzung

mech_names = ['Atr', 'Gri', 'Aramco', 'Nuig']
spezien = ['H2', 'CO', 'CO2', 'H2O']  # Beliebig erweiterbar

data_dict = {}

for sp in spezien:
    colname = f' Mole_fraction_{sp}_PFRC2_()'
    data_dict[sp] = [
        df_atr_pfr[colname].iloc[-1] * 100,
        df_gri_pfr[colname].iloc[-1] * 100,
        df_aramco_pfr[colname].iloc[-1] * 100,
        df_nuig_pfr[colname].iloc[-1] * 100,
    ]

# Plot: Gruppenbalkendiagramm (für mehrere Spezies)
x = np.arange(len(mech_names))  # Mechanismen auf x-Achse
width = 0.2                    # Balkenbreite
fig, ax = plt.subplots(figsize=(12, 7))

for i, sp in enumerate(spezien):
    ax.bar(x + i*width, data_dict[sp], width, label=sp, color = [color1, color2, color3, color4][i]) # richtige Farben: 

ax.set_xticks(x + width * (len(spezien)-1) / 2)
ax.set_xticklabels(mech_names)
ax.set_ylabel('Molefraktion [%]')
ax.set_title('Speziesanteile in Abgasen')
ax.legend()
ax.grid(axis='y')
plt.tight_layout()
plt.savefig('img/Abgaszusammensetzung_alle.png', dpi=300)
plt.close()