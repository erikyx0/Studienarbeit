import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

import os 

#%% Funktionen
def remove_water_and_normalize_topn(series,n):
    # Wasser entfernen
    ohne_wasser = series[~series.index.str.contains("H2O")]
    # Normalisieren
    normiert = ohne_wasser / ohne_wasser.sum()
    # Top 10 auswählen
    top10 = normiert.sort_values(ascending=False).head(n)
    return top10

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

#%% Dataframes einlesen

df_co2_pfr3 = pd.read_excel('Daten/CO2.xlsm', sheet_name='3.soln_no_1_PFRC3')
df_no_co2_pfr3 = pd.read_excel('Daten/kein CO2.xlsm', sheet_name='3.soln_no_1_PFRC3')

df_co2_pfr9 = pd.read_excel('Daten/CO2.xlsm', sheet_name='11.soln_no_1_PFRC9')
df_no_co2_pfr9 = pd.read_excel('Daten/kein CO2.xlsm', sheet_name='11.soln_no_1_PFRC9')

df_co2_pfr3_stoffmenge = pd.read_excel('Daten/CO2 Stoffmenge.xlsm', sheet_name='3.soln_no_1_PFRC3')
df_no_co2_pfr3_stoffmenge = pd.read_excel('Daten/kein CO2 Stoffmenge.xlsm', sheet_name='3.soln_no_1_PFRC3')

df_co2_pfr9_stoffmenge = pd.read_excel('Daten/CO2 Stoffmenge.xlsm', sheet_name='11.soln_no_1_PFRC9')
df_no_co2_pfr9_stoffmenge = pd.read_excel('Daten/kein CO2 Stoffmenge.xlsm', sheet_name='11.soln_no_1_PFRC9')

#%% Arrays aus Daten extrahieren
# PFR3 
dist_pfr3 = df_co2_pfr3['Distance_PFRC3_(m)']

temp_no_co2_pfr3 = df_no_co2_pfr3[' Surface_temperature_PFRC3_(K)']
temp_co2_pfr3 = df_co2_pfr3[' Surface_temperature_PFRC3_(K)']

x_co2_no_co2_pfr3 = df_no_co2_pfr3[' Mass_fraction_CO2_PFRC3_()']
x_co2_co2_pfr3 = df_co2_pfr3[' Mass_fraction_CO2_PFRC3_()']

x_co_no_co2_pfr3 = df_no_co2_pfr3[' Mass_fraction_CO_PFRC3_()']
x_co_co2_pfr3 = df_co2_pfr3[' Mass_fraction_CO_PFRC3_()']

x_h2o_no_co2_pfr3 = df_no_co2_pfr3[' Mass_fraction_H2O_PFRC3_()']
x_h2o_co2_pfr3 = df_co2_pfr3[' Mass_fraction_H2O_PFRC3_()']

x_h2_no_co2_pfr3 = df_no_co2_pfr3[' Mass_fraction_H2_PFRC3_()']
x_h2_co2_pfr3 = df_co2_pfr3[' Mass_fraction_H2_PFRC3_()']

x_o2_no_co2_pfr3 = df_no_co2_pfr3[' Mass_fraction_O2_PFRC3_()']
x_o2_co2_pfr3 = df_co2_pfr3[' Mass_fraction_O2_PFRC3_()']

x_ch4_no_co2_pfr3 = df_no_co2_pfr3[' Mass_fraction_CH4_PFRC3_()']
x_ch4_co2_pfr3 = df_co2_pfr3[' Mass_fraction_CH4_PFRC3_()']


y_co2_no_co2_pfr3 = df_no_co2_pfr3_stoffmenge[' Mole_fraction_CO2_PFRC3_()']
y_co2_co2_pfr3 = df_co2_pfr3_stoffmenge[' Mole_fraction_CO2_PFRC3_()']

y_co_no_co2_pfr3 = df_no_co2_pfr3_stoffmenge[' Mole_fraction_CO_PFRC3_()']
y_co_co2_pfr3 = df_co2_pfr3_stoffmenge[' Mole_fraction_CO_PFRC3_()']

y_h2o_no_co2_pfr3 = df_no_co2_pfr3_stoffmenge[' Mole_fraction_H2O_PFRC3_()']
y_h2o_co2_pfr3 = df_co2_pfr3_stoffmenge[' Mole_fraction_H2O_PFRC3_()']

y_h2_no_co2_pfr3 = df_no_co2_pfr3_stoffmenge[' Mole_fraction_H2_PFRC3_()']
y_h2_co2_pfr3 = df_co2_pfr3_stoffmenge[' Mole_fraction_H2_PFRC3_()']

y_o2_no_co2_pfr3 = df_no_co2_pfr3_stoffmenge[' Mole_fraction_O2_PFRC3_()']
y_o2_co2_pfr3 = df_co2_pfr3_stoffmenge[' Mole_fraction_O2_PFRC3_()']

y_ch4_no_co2_pfr3 = df_no_co2_pfr3_stoffmenge[' Mole_fraction_CH4_PFRC3_()']
y_ch4_co2_pfr3 = df_co2_pfr3_stoffmenge[' Mole_fraction_CH4_PFRC3_()']

# PFR9 
dist_pfr9 = df_co2_pfr9['Distance_PFRC9_(m)']

temp_no_co2_pfr9 = df_no_co2_pfr9[' Surface_temperature_PFRC9_(K)']
temp_co2_pfr9 = df_co2_pfr9[' Surface_temperature_PFRC9_(K)']

x_co2_no_co2_pfr9 = df_no_co2_pfr9[' Mass_fraction_CO2_PFRC9_()']
x_co2_co2_pfr9 = df_co2_pfr9[' Mass_fraction_CO2_PFRC9_()']

x_co_no_co2_pfr9 = df_no_co2_pfr9[' Mass_fraction_CO_PFRC9_()']
x_co_co2_pfr9 = df_co2_pfr9[' Mass_fraction_CO_PFRC9_()']

x_h2o_no_co2_pfr9 = df_no_co2_pfr9[' Mass_fraction_H2O_PFRC9_()']
x_h2o_co2_pfr9 = df_co2_pfr9[' Mass_fraction_H2O_PFRC9_()']

x_h2_no_co2_pfr9 = df_no_co2_pfr9[' Mass_fraction_H2_PFRC9_()']
x_h2_co2_pfr9 = df_co2_pfr9[' Mass_fraction_H2_PFRC9_()']

#%% Produktgaszusammensetzung

def normalize_dry(row):
    """
    Setzt CH4 = 0 und normiert H2, CO, CO2 auf Volumenbasis (trocken).
    Erwartet eine Series mit den Spalten: 'H2', 'CO', 'CH4', 'CO2'
    """
    h2 = row['H2']
    co = row['CO']
    co2 = row['CO2']
    total = h2 + co + co2

    return pd.Series({
        'H2 (kein CH4)': h2 / total,
        'CO (kein CH4)': co / total,
        'CO2 (kein CH4)': co2 / total,
        'H2/CO (kein CH4)': (h2 / co) if co != 0 else float('inf')
    })

# Letzte Werte extrahieren
data = {
    'H2': [y_h2_no_co2_pfr3.iloc[-1], y_h2_co2_pfr3.iloc[-1]],
    'CO': [y_co_no_co2_pfr3.iloc[-1], y_co_co2_pfr3.iloc[-1]],
    'CH4': [y_ch4_no_co2_pfr3.iloc[-1], y_ch4_co2_pfr3.iloc[-1]],
    'CO2': [y_co2_no_co2_pfr3.iloc[-1], y_co2_co2_pfr3.iloc[-1]],
    'H2/CO': [
        y_h2_no_co2_pfr3.iloc[-1] / y_co_no_co2_pfr3.iloc[-1],
        y_h2_co2_pfr3.iloc[-1] / y_co_co2_pfr3.iloc[-1]
    ]
}

# NICHT transponieren!
df = pd.DataFrame(data, index=['ohne CO2', 'mit CO2'])

# normalize_dry auf jede Zeile anwenden
df_dry = df.apply(normalize_dry, axis=1)

# zusammenführen
df_gesamt = pd.concat([df, df_dry], axis=1)

print(df_gesamt.round(4).T)
#%% Plots 

#%% PFR3
fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

# ACHTUNG: axs enthält nur die linken Achsen → wir erzeugen je eine rechte mit twinx()
twin_axes = [[None, None], [None, None]]

# Plot 1: H2
ax = axs[0, 0]
twin_axes[0][0] = ax.twinx()
ax.plot(dist_pfr3, x_h2_no_co2_pfr3, color='tab:blue', label='H₂ (kein CO₂)')
twin_axes[0][0].plot(dist_pfr3, x_h2_co2_pfr3, color='tab:red', linestyle='--', label='H₂ (mit CO₂)')
ax.set_title('H$_2$')
ax.set_ylabel('kein CO$_2$')
twin_axes[0][0].set_ylabel('mit CO$_2$')
ax.grid()
ax.legend(loc='upper left')
twin_axes[0][0].legend(loc='upper right')

# Plot 2: H2O
ax = axs[0, 1]
twin_axes[0][1] = ax.twinx()
ax.plot(dist_pfr3, x_h2o_no_co2_pfr3, color='tab:green', label='H₂O (kein CO₂)')
twin_axes[0][1].plot(dist_pfr3, x_h2o_co2_pfr3, color='tab:orange', linestyle='--', label='H₂O (mit CO₂)')
ax.set_title('H$_2$O')
ax.legend(loc='upper left')
twin_axes[0][1].legend(loc='upper right')
ax.grid()

# Plot 3: CO2
ax = axs[1, 0]
twin_axes[1][0] = ax.twinx()
ax.plot(dist_pfr3, x_co2_no_co2_pfr3, color='tab:purple', label='CO₂ (kein CO₂)')
twin_axes[1][0].plot(dist_pfr3, x_co2_co2_pfr3, color='tab:pink', linestyle='--', label='CO₂ (mit CO₂)')
ax.set_title('CO$_2$')
ax.set_xlabel('Distance (m)')
ax.set_ylabel('kein CO$_2$')
twin_axes[1][0].set_ylabel('mit CO$_2$')
ax.legend(loc='upper left')
twin_axes[1][0].legend(loc='upper right')
ax.grid()

# Plot 4: CO
ax = axs[1, 1]
twin_axes[1][1] = ax.twinx()
ax.plot(dist_pfr3, x_co_no_co2_pfr3, color='tab:brown', label='CO (kein CO₂)')
twin_axes[1][1].plot(dist_pfr3, x_co_co2_pfr3, color='tab:cyan', linestyle='--', label='CO (mit CO₂)')
ax.set_title('CO')
ax.set_xlabel('Distance (m)')
ax.legend(loc='upper left')
twin_axes[1][1].legend(loc='upper right')
ax.grid()

plt.tight_layout()
#plt.show()
plt.savefig("img/Stoffe_PFR3.jpg", dpi = 300)
plt.close()

#%% PFR9
fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

# ACHTUNG: axs enthält nur die linken Achsen → wir erzeugen je eine rechte mit twinx()
twin_axes = [[None, None], [None, None]]

# Plot 1: H2
ax = axs[0, 0]
twin_axes[0][0] = ax.twinx()
ax.plot(dist_pfr9, x_h2_no_co2_pfr9, color='tab:blue', label='H₂ (kein CO₂)')
twin_axes[0][0].plot(dist_pfr9, x_h2_co2_pfr9, color='tab:red', linestyle='--', label='H₂ (mit CO₂)')
ax.set_title('H$_2$')
ax.set_ylabel('kein CO$_2$')
twin_axes[0][0].set_ylabel('mit CO$_2$')
ax.grid()
ax.legend(loc='upper left')
twin_axes[0][0].legend(loc='upper right')

# Plot 2: H2O
ax = axs[0, 1]
twin_axes[0][1] = ax.twinx()
ax.plot(dist_pfr9, x_h2o_no_co2_pfr9, color='tab:green', label='H₂O (kein CO₂)')
twin_axes[0][1].plot(dist_pfr9, x_h2o_co2_pfr9, color='tab:orange', linestyle='--', label='H₂O (mit CO₂)')
ax.set_title('H$_2$O')
ax.legend(loc='upper left')
twin_axes[0][1].legend(loc='upper right')
ax.grid()

# Plot 3: CO2
ax = axs[1, 0]
twin_axes[1][0] = ax.twinx()
ax.plot(dist_pfr9, x_co2_no_co2_pfr9, color='tab:purple', label='CO₂ (kein CO₂)')
twin_axes[1][0].plot(dist_pfr9, x_co2_co2_pfr9, color='tab:pink', linestyle='--', label='CO₂ (mit CO₂)')
ax.set_title('CO$_2$')
ax.set_xlabel('Distance (m)')
ax.set_ylabel('kein CO$_2$')
twin_axes[1][0].set_ylabel('mit CO$_2$')
ax.legend(loc='upper left')
twin_axes[1][0].legend(loc='upper right')
ax.grid()

# Plot 4: CO
ax = axs[1, 1]
twin_axes[1][1] = ax.twinx()
ax.plot(dist_pfr9, x_co_no_co2_pfr9, color='tab:brown', label='CO (kein CO₂)')
twin_axes[1][1].plot(dist_pfr9, x_co_co2_pfr9, color='tab:cyan', linestyle='--', label='CO (mit CO₂)')
ax.set_title('CO')
ax.set_xlabel('Distance (m)')
ax.legend(loc='upper left')
twin_axes[1][1].legend(loc='upper right')
ax.grid()

plt.tight_layout()
#plt.show()
plt.savefig("img/Stoffe_PFR9.jpg", dpi = 300)
plt.close()

fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Plot für PFR3
axs[0].plot(dist_pfr3, temp_no_co2_pfr3, label="kein CO₂, PFR3")
axs[0].plot(dist_pfr3, temp_co2_pfr3, label="CO₂, PFR3")
axs[0].set_title("Temperaturverlauf in PFR3")
axs[0].set_xlabel("Distanz (m)")
axs[0].set_ylabel("Temperatur (K)")
axs[0].grid()
axs[0].legend()

# Plot für PFR9
axs[1].plot(dist_pfr9, temp_no_co2_pfr9, label="kein CO₂, PFR9")
axs[1].plot(dist_pfr9, temp_co2_pfr9, label="CO₂, PFR9")
axs[1].set_title("Temperaturverlauf in PFR9")
axs[1].set_xlabel("Distanz (m)")
axs[1].grid()
axs[1].legend()

plt.tight_layout()
plt.show()
plt.close()

print(f"Massenstrom kein CO2: {df_no_co2_pfr9[' Exit_mass_flow_rate_PFRC9_(kg/sec)'].iloc[0]} kg/s")
print(f"Massenstrom CO2: {df_co2_pfr9[' Exit_mass_flow_rate_PFRC9_(kg/sec)'].iloc[0]} kg/s")
print(f"Temperatur am Reaktoraustritt (kein CO2): {df_no_co2_pfr9[' Surface_temperature_PFRC9_(K)'].iloc[-1] - 273.15} °C")
print(f"Temperatur am Reaktoraustritt (CO2): {df_co2_pfr9[' Surface_temperature_PFRC9_(K)'].iloc[-1] - 273.15} °C")
print(f"Temperatur in der Mitte von PFR9 (kein CO2): {df_no_co2_pfr9[' Surface_temperature_PFRC9_(K)'].iloc[round(len(df_no_co2_pfr9[' Surface_temperature_PFRC9_(K)'])/2)] - 273.15} °C")
print(f"Temperatur in der Mitte von PFR9 (CO2): {df_co2_pfr9[' Surface_temperature_PFRC9_(K)'].iloc[round(len(df_co2_pfr9[' Surface_temperature_PFRC9_(K)'])/2)] - 273.15} °C")

plt.plot(dist_pfr3, x_ch4_co2_pfr3, label = "CO2")
plt.plot(dist_pfr3, x_ch4_no_co2_pfr3, label = "kein CO2")
plt.grid()
plt.legend()
plt.show()

#%% Abgaszusammensetzungen 
# Schritt 1: Spalten mit 'mole_fraction' im Namen auswählen
last_row_no_co2 = df_no_co2_pfr9_stoffmenge.iloc[-1][[col for col in df_no_co2_pfr9_stoffmenge.columns if "Mole_fraction" in col]]
last_row_co2 = df_co2_pfr9_stoffmenge.iloc[-1][[col for col in df_co2_pfr9_stoffmenge.columns if "Mole_fraction" in col]]
# Funktion anwenden
no_co2_normiert = remove_water_and_normalize_topn(last_row_no_co2, 5)
co2_normiert = remove_water_and_normalize_topn(last_row_co2, 5)
print("CO2:")
print(co2_normiert)
print("kein CO2:")
print(no_co2_normiert)