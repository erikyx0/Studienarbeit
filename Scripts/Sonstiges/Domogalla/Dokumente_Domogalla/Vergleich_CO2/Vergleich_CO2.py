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
print(script_dir)

##% einlesen der Daten 
df_pfr9_ende = pd.read_excel('Parameterstudie_CO2.xlsm', sheet_name='41.PFRC9_end_point_vs_paramete', decimal=",")
df_pfr9_sol1 = pd.read_excel('Parameterstudie_CO2.xlsm', sheet_name='30.PFRC9_soln_no_1', decimal=",")
df_pfr3_sol1_1 = pd.read_excel('Parameterstudie_CO2.xlsm', sheet_name='12.soln_no_1_PFRC3_Run#1', decimal=",")
df_pfr3_sol1_2 = pd.read_excel('Parameterstudie_CO2.xlsm', sheet_name='13.soln_no_1_PFRC3_Run#2', decimal=",")

#%% sequenzierung der Daten 
#pfr3
# Distanz
dist_pfr3_sol1 = df_pfr3_sol1_1['Distance_PFRC3_Run#1_(m)']
dist_pfr3_sol2 = df_pfr3_sol1_2['Distance_PFRC3_Run#2_(m)']

# Temperaturprofile
temp_pfr3_sol1_1 = df_pfr3_sol1_1[' Temperature_PFRC3_Run#1_(K)']
temp_pfr3_sol1_2 = df_pfr3_sol1_2[' Temperature_PFRC3_Run#2_(K)']

# CO2-Profil
x_CO2_pfr3_sol1_1 = df_pfr3_sol1_1[' Mole_fraction_CO2_PFRC3_Run#1_()']
x_CO2_pfr3_sol1_2 = df_pfr3_sol1_2[' Mole_fraction_CO2_PFRC3_Run#2_()']

# CO-Profil
x_CO_pfr3_sol1_1 = df_pfr3_sol1_1[' Mole_fraction_CO_PFRC3_Run#1_()']
x_CO_pfr3_sol1_2 = df_pfr3_sol1_2[' Mole_fraction_CO_PFRC3_Run#2_()']

# H2-Profil
x_H2_pfr3_sol1_1 = df_pfr3_sol1_1[' Mole_fraction_H2_PFRC3_Run#1_()']
x_H2_pfr3_sol1_2 = df_pfr3_sol1_2[' Mole_fraction_H2_PFRC3_Run#2_()']

# H2O-Profil
x_H2O_pfr3_sol1_1 = df_pfr3_sol1_1[' Mole_fraction_H2O_PFRC3_Run#1_()']
x_H2O_pfr3_sol1_2 = df_pfr3_sol1_2[' Mole_fraction_H2O_PFRC3_Run#2_()']

# pfr9
df_pfr9_ende_H2 = df_pfr9_ende[' Mole_fraction_H2_PFRC9_end_point_()']
df_pfr9_ende_H2O = df_pfr9_ende[' Mole_fraction_H2O_PFRC9_end_point_()']
df_pfr9_ende_CO = df_pfr9_ende[' Mole_fraction_CO_PFRC9_end_point_()']
df_pfr9_ende_CO2 = df_pfr9_ende[' Mole_fraction_CO2_PFRC9_end_point_()']

dist_pfr9_sol1 = df_pfr9_sol1['Distance_(m)']

temp_pfr9_sol1_1 = df_pfr9_sol1[' Temperature_PFRC9_Run#1_(K)']
temp_pfr9_sol1_2 = df_pfr9_sol1[' Temperature_PFRC9_Run#2_(K)']

x_CO2_pfr9_sol1_1 = df_pfr9_sol1[' Mole_fraction_CO2_PFRC9_Run#1_()']
x_CO2_pfr9_sol1_2 = df_pfr9_sol1[' Mole_fraction_CO2_PFRC9_Run#2_()']

x_CO_pfr9_sol1_1 = df_pfr9_sol1[' Mole_fraction_CO_PFRC9_Run#1_()']
x_CO_pfr9_sol1_2 = df_pfr9_sol1[' Mole_fraction_CO_PFRC9_Run#2_()']

x_H2_pfr9_sol1_1 = df_pfr9_sol1[' Mole_fraction_H2_PFRC9_Run#1_()']
x_H2_pfr9_sol1_2 = df_pfr9_sol1[' Mole_fraction_H2_PFRC9_Run#2_()']

x_H2O_pfr9_sol1_1 = df_pfr9_sol1[' Mole_fraction_H2O_PFRC9_Run#1_()']
x_H2O_pfr9_sol1_2 = df_pfr9_sol1[' Mole_fraction_H2O_PFRC9_Run#2_()']



#%% Vergleich Endpunkte Plotten
h2 = df_pfr9_ende_H2.values
h2o = df_pfr9_ende_H2O.values
co = df_pfr9_ende_CO.values
co2 = df_pfr9_ende_CO2.values

labels = ['H2', 'H2O', 'CO', 'CO2']
mp1 = [h2[0], h2o[0], co[0], co2[0]]
mp2 = [h2[1], h2o[1], co[1], co2[1]]
diff = np.array(mp2) - np.array(mp1)

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots()

rects1 = ax.bar(x - width, mp1, width, label=r'Simulation ohne CO$_2$', color=color1)
rects2 = ax.bar(x, mp2, width, label=r'Simulation mit CO$_2$', color=color2)
rects3 = ax.bar(x + width, diff, width, label='Differenz', color=color4)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Molenbruch')
ax.legend()
ax.grid( linestyle='dotted', linewidth=1)
plt.tight_layout()

# Prozentuale Änderung berechnen (gegenüber Messpunkt 1)
mp1 = np.array(mp1)
percent_change = np.where(mp1 != 0, 100 * diff / mp1, np.nan)

# Prozentwerte als Text auf die Differenz-Balken schreiben
for xi, yi, pct in zip(x, diff, percent_change):
    if not np.isnan(pct):
        ax.text(
            xi + width,                        # x-Position: Differenz-Balken
            yi if yi >= 0 else yi - 0.01,      # y-Position: Oben/unten am Balken
            f'{pct:+.1f}%',                    # z.B. "+14.2%"
            ha='center',
            va='bottom' if yi >= 0 else 'top',
            fontsize=10,
            color='black'
        )

plt.savefig('img/vergleich_endpunkte.png', dpi = 300)
plt.close()

#%% Plot Nachbrennzone
# Plot zeigt, dass keine Reaktionen mehr stattfinden! 
"""
plt.plot(dist_pfr9_sol1,x_CO2_pfr9_sol1_1, label = r'CO$_2$: kein CO$_2$', color = color1, linestyle = '--')
plt.plot(dist_pfr9_sol1,x_CO2_pfr9_sol1_2, label = r'CO$_2$: CO$_2$', color = color1)

plt.plot(dist_pfr9_sol1,x_CO_pfr9_sol1_1, label = r'CO: kein CO$_2$', color = color2, linestyle = '--')
plt.plot(dist_pfr9_sol1,x_CO_pfr9_sol1_2, label = r'CO: CO$_2$', color = color2)

plt.plot(dist_pfr9_sol1,x_H2_pfr9_sol1_1, label = r'H$_2$: kein CO$_2$', color = color3, linestyle = '--')
plt.plot(dist_pfr9_sol1,x_H2_pfr9_sol1_2, label = r'H$_2$: CO$_2$', color = color3)

plt.plot(dist_pfr9_sol1,x_H2O_pfr9_sol1_1, label = r'CO: kein H$_2$O', color = color4, linestyle = '--')
plt.plot(dist_pfr9_sol1,x_H2O_pfr9_sol1_2, label = r'CO: H$_2$O', color = color4)
"""

plt.plot(dist_pfr9_sol1, temp_pfr9_sol1_1, label='Temp kein CO2', color='red')
plt.plot(dist_pfr9_sol1, temp_pfr9_sol1_2, label='Temp mit CO2', color='red', linestyle = 'dotted')

plt.plot()

plt.grid()
plt.legend(loc='best')
plt.xlabel('Reaktorlänge [m]')
plt.ylabel(r'Molenbruch CO$_2$')
plt.show()
plt.close()

#%% Plot PFR3

plt.plot(dist_pfr3_sol1, x_CO2_pfr3_sol1_1, label=r'CO$_2$: kein CO$_2$', color=color1, linestyle='--')
plt.plot(dist_pfr3_sol1, x_CO2_pfr3_sol1_2, label=r'CO$_2$: CO$_2$', color=color1)
"""
plt.plot(dist_pfr3_sol1, x_CO_pfr3_sol1_1, label=r'CO: kein CO$_2$', color=color2, linestyle='--')
plt.plot(dist_pfr3_sol1, x_CO_pfr3_sol1_2, label=r'CO: CO$_2$', color=color2)

plt.plot(dist_pfr3_sol1, x_H2_pfr3_sol1_1, label=r'H$_2$: kein CO$_2$', color=color3, linestyle='--')
plt.plot(dist_pfr3_sol1, x_H2_pfr3_sol1_2, label=r'H$_2$: CO$_2$', color=color3)

plt.plot(dist_pfr3_sol1, x_H2O_pfr3_sol1_1, label=r'H$_2$O: kein CO$_2$', color=color4, linestyle='--')
plt.plot(dist_pfr3_sol1, x_H2O_pfr3_sol1_2, label=r'H$_2$O: CO$_2$', color=color4)
"""
plt.xlim(0,0.002)  # Schneidet die x-Achse bei 0.002 ab

plt.grid()
plt.legend(loc='best')
plt.xlabel('Reaktorlänge [m]')
plt.ylabel(r'Molenbruch CO$_2$')
plt.show()
plt.close()

