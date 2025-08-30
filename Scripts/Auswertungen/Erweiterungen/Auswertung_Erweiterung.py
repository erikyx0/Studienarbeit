import pandas as pd 
import numpy as np 
#from Latex_table import create_latex_table
import os
import matplotlib.pyplot as plt
import re

color1 = '#4878A8'
color2 = '#7E9680'
color3 = '#B3B3B3'
color4 = '#BC6C25'
color5 = "#960B0B"
color6 = "#077B1A"

colors = [color1,color2,color3,color4,color5, color6]

# Set the working directory to the location of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

#%% Funktionen 
def get_normalized_composition_strict(df, column_map):
    comp = {key: df.iloc[-1][col] for key, col in column_map.items() if col in df.columns}
    # Optional: Wasser entfernen
    if "H2O" in comp:
        del comp["H2O"]
    comp_series = pd.Series(comp)
    comp_series = comp_series / comp_series.sum()
    return comp_series

#%% Dataframes einlesen
"""
letzter PFR:
1: PFRC2 
2: PFRC4
3: PFRC8
4: PFRC7
"""

df_co2_1_pfr_Masse = pd.read_excel('Daten/1_CO2_Masse.xlsm', sheet_name='2.soln_no_1_PFRC2')
df_co2_1_pfr_Stoffmenge = pd.read_excel('Daten/1_CO2_Stoffmenge.xlsm', sheet_name='2.soln_no_1_PFRC2')

df_co2_2_pfr_Masse = pd.read_excel('Daten/2_CO2_Masse.xlsm', sheet_name='9.soln_no_1_PFRC4')
df_co2_2_pfr_Stoffmenge = pd.read_excel('Daten/2_CO2_Stoffmenge.xlsm', sheet_name='9.soln_no_1_PFRC4')

df_co2_3_pfr8_Masse = pd.read_excel('Daten/3_CO2_Masse.xlsm', sheet_name='24.soln_no_1_PFRC8')
df_co2_3_pfr8_Stoffmenge = pd.read_excel('Daten/3_CO2_Stoffmenge.xlsm', sheet_name='24.soln_no_1_PFRC8')

df_co2_4_pfr7_Masse = pd.read_excel('Daten/4_CO2_Masse.xlsm', sheet_name='21.soln_no_1_PFRC7')
df_co2_4_pfr7_Stoffmenge = pd.read_excel('Daten/4_CO2_Stoffmenge.xlsm', sheet_name='21.soln_no_1_PFRC7')

#df_co2_5_Masse = pd.read_excel('Daten/5_CO2_Masse.xlsm', sheet_name='2.soln_no_1_PFRC2')
#df_co2_5_Stoffmenge = pd.read_excel('Daten/5_CO2_Stoffmenge.xlsm', sheet_name='2.soln_no_1_PFRC2')

df_co2_6_pfr3_Masse = pd.read_excel('Daten/6_CO2_Masse.xlsm', sheet_name='6.soln_no_1_PFRC3')
df_co2_6_pfr3_Stoffmenge = pd.read_excel('Daten/6_CO2_Stoffmenge.xlsm', sheet_name='6.soln_no_1_PFRC3')

#%% Daten vorbereiten 
x_h2_co2_1 = df_co2_1_pfr_Stoffmenge[" Mole_fraction_H2_PFRC2_()"]
x_h2_co2_2 = df_co2_2_pfr_Stoffmenge[" Mole_fraction_H2_PFRC4_()"]
x_h2_co2_3 = df_co2_3_pfr8_Stoffmenge[" Mole_fraction_H2_PFRC8_()"]
x_h2_co2_4 = df_co2_4_pfr7_Stoffmenge[" Mole_fraction_H2_PFRC7_()"]
x_h2_co2_6 = df_co2_6_pfr3_Stoffmenge[" Mole_fraction_H2_PFRC3_()"]

plt.plot(x_h2_co2_1, label = "1")
plt.plot(x_h2_co2_2, label = "2")
plt.plot(x_h2_co2_3, label = "3")
plt.plot(x_h2_co2_4, label = "4")
plt.plot(x_h2_co2_6, label = "6")
plt.grid()
plt.tight_layout()
plt.legend()
plt.show()

#%% Vergleich experimentaldaten

def clean_columns(df):
    """Führende/trailing Leerzeichen aus Spaltennamen entfernen (deine CSVs haben die oft)."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df

def build_species_map(df, species=("H2","CO","CH4","CO2","H2O")):
    """
    Findet für jede Spezies die passende Mole-Fraction-Spalte mit beliebiger PFRC-ID.
    Falls mehrere PFRC-Treffer existieren, wird die mit der höchsten ID gewählt.
    """
    df = clean_columns(df)
    colmap = {}
    for sp in species:
        pat = re.compile(rf'^Mole_fraction_{re.escape(sp)}_PFRC(\d+)_\(\)$')
        matches = [c for c in df.columns if pat.match(c)]
        if not matches:
            # Fallback: tolerant gegen Varianten ohne PFRC oder leicht andere Schreibweisen
            alt = [c for c in df.columns if c.startswith(f"Mole_fraction_{sp}")]
            if not alt:
                raise KeyError(f"Keine Spalte für Spezies '{sp}' gefunden.")
            colmap[sp] = alt[-1]
        else:
            # Nimm die höchste PFRC-ID (robust falls mehrere vorhanden sind)
            best = max(matches, key=lambda c: int(pat.match(c).group(1)))
            colmap[sp] = best
    return df, {sp: colmap[sp] for sp in species}

def h2_series(df):
    """Bequemer Zugriff nur auf die H2-Spalte (beliebige PFRC-ID)."""
    df_clean, m = build_species_map(df, species=("H2",))
    return df_clean[m["H2"]]

# --- Plot: dynamisch statt hartkodierter PFRC-IDs ---
x_h2_co2_1 = h2_series(df_co2_1_pfr_Stoffmenge)
x_h2_co2_2 = h2_series(df_co2_2_pfr_Stoffmenge)
x_h2_co2_3 = h2_series(df_co2_3_pfr8_Stoffmenge)
x_h2_co2_4 = h2_series(df_co2_4_pfr7_Stoffmenge)
x_h2_co2_6 = h2_series(df_co2_6_pfr3_Stoffmenge)

plt.figure()
plt.plot(x_h2_co2_1, label="1")
plt.plot(x_h2_co2_2, label="2")
plt.plot(x_h2_co2_3, label="3")
plt.plot(x_h2_co2_4, label="4")
plt.plot(x_h2_co2_6, label="6")
plt.grid(True)
plt.legend()
plt.tight_layout()

# --- Vergleich mit Experiment: Species-Mapping pro DF dynamisch bestimmen ---

df1, map1 = build_species_map(df_co2_1_pfr_Stoffmenge)
df2, map2 = build_species_map(df_co2_2_pfr_Stoffmenge)
df3, map3 = build_species_map(df_co2_3_pfr8_Stoffmenge)
df4, map4 = build_species_map(df_co2_4_pfr7_Stoffmenge)
df6, map6 = build_species_map(df_co2_6_pfr3_Stoffmenge)

comp_1 = get_normalized_composition_strict(df1, map1)
comp_2 = get_normalized_composition_strict(df2, map2)
comp_3 = get_normalized_composition_strict(df3, map3)
comp_4 = get_normalized_composition_strict(df4, map4)
comp_6 = get_normalized_composition_strict(df6, map6)

exp_data = {
    "Exp": [0.416,0.424,0.0012,0.151],
    "1 CO2": comp_1.tolist(),
    "2 CO2": comp_2.tolist(),
    "3 CO2": comp_3.tolist(),
    "4 CO2": comp_4.tolist(),
    "6 CO2": comp_6.tolist(),
}

df_exp_data = pd.DataFrame(exp_data, index = ["H2", "CO", "CH4", "CO2"])
print(df_exp_data)

# Säulendiagramm (gruppiert)
ax = df_exp_data.plot(kind="bar", color=colors)
ax.set_xlabel("Spezies")
ax.set_ylabel("Molenbruch")
ax.set_title("Experiment vs. Simulation (CO₂-Fall)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Optional speichern:
# plt.savefig("saeulendiagramm.png", dpi=300)

def tvd_scores(df, exp_col="Exp"):
    # Erwartet df mit Spalten: "Exp", "1 CO2", "2 CO2", ...
    exp = df[exp_col].astype(float)
    sims = df.drop(columns=[exp_col]).astype(float)
    # TVD = 0.5 * L1-Distanz der Verteilungen
    tvd = 0.5 * (sims.sub(exp, axis=0).abs()).sum(axis=0)
    return tvd.sort_values()  # kleiner = besser

scores = tvd_scores(df_exp_data)
print("TVD pro Modell (0=perfekt, 1=schlecht):")
print(scores)
print("\nBestes Modell:", scores.idxmin(), "mit TVD =", float(scores.min()))