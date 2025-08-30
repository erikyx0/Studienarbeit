import pandas as pd 
import numpy as np 
#from Latex_table import create_latex_table
import os
import matplotlib.pyplot as plt

color1 = '#4878A8'
color2 = '#7E9680'
color3 = '#B3B3B3'
color4 = '#BC6C25'

# Set the working directory to the location of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

#%% Dataframes einlesen

df_co2_1_pfr_Masse = pd.read_excel('Daten/1_CO2_Masse.xlsm', sheet_name='2.soln_no_1_PFRC2')
df_co2_1_pfr_Stoffmenge = pd.read_excel('Daten/1_CO2_Stoffmenge.xlsm', sheet_name='2.soln_no_1_PFRC2')

df_co2_2_pfr_Masse = pd.read_excel('Daten/2_CO2_Masse.xlsm', sheet_name='9.soln_no_1_PFRC4')
df_co2_2_pfr_Stoffmenge = pd.read_excel('Daten/2_CO2_Stoffmenge.xlsm', sheet_name='9.soln_no_1_PFRC4')

df_co2_3_pfr3_Masse = pd.read_excel('Daten/3_CO2_Masse.xlsm', sheet_name='6.soln_no_1_PFRC3')
df_co2_3_pfr3_Stoffmenge = pd.read_excel('Daten/3_CO2_Stoffmenge.xlsm', sheet_name='6.soln_no_1_PFRC3')

df_co2_4_pfr3_Masse = pd.read_excel('Daten/4_CO2_Masse.xlsm', sheet_name='6.soln_no_1_PFRC3')
df_co2_4_pfr3_Stoffmenge = pd.read_excel('Daten/4_CO2_Stoffmenge.xlsm', sheet_name='6.soln_no_1_PFRC3')

#df_co2_5_Masse = pd.read_excel('Daten/5_CO2_Masse.xlsm', sheet_name='2.soln_no_1_PFRC2')
#df_co2_5_Stoffmenge = pd.read_excel('Daten/5_CO2_Stoffmenge.xlsm', sheet_name='2.soln_no_1_PFRC2')

df_co2_6_pfr3_Masse = pd.read_excel('Daten/6_CO2_Masse.xlsm', sheet_name='6.soln_no_1_PFRC3')
df_co2_6_pfr3_Stoffmenge = pd.read_excel('Daten/6_CO2_Stoffmenge.xlsm', sheet_name='6.soln_no_1_PFRC3')

#%% Daten vorbereiten 
x_h2_co2_1 = df_co2_1_pfr_Stoffmenge[" Mole_fraction_H2_PFRC2_()"]
x_h2_co2_2 = df_co2_2_pfr_Stoffmenge[" Mole_fraction_H2_PFRC4_()"]
x_h2_co2_3 = df_co2_3_pfr3_Stoffmenge[" Mole_fraction_H2_PFRC3_()"]
x_h2_co2_4 = df_co2_4_pfr3_Stoffmenge[" Mole_fraction_H2_PFRC3_()"]
x_h2_co2_6 = df_co2_6_pfr3_Stoffmenge[" Mole_fraction_H2_PFRC3_()"]

plt.plot(x_h2_co2_1)
plt.plot(x_h2_co2_2)
plt.plot(x_h2_co2_3)
plt.plot(x_h2_co2_4)
plt.plot(x_h2_co2_6)

plt.show()