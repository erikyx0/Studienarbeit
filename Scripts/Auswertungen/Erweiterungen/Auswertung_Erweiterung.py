import pandas as pd 
import numpy as np 
#from Latex_table import create_latex_table
import os
import matplotlib.pyplot as plt
import re

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

#%% Funktionen 
def get_normalized_composition_strict(df, column_map):
    comp = {key: df.iloc[-1][col] for key, col in column_map.items() if col in df.columns}
    # Optional: Wasser entfernen
    if "H2O" in comp:
        del comp["H2O"]
    comp_series = pd.Series(comp)
    comp_series = comp_series / comp_series.sum()
    return comp_series

#%% Dataframes einlesen Excel (alt)
"""
letzter PFR:
1: PFRC2 
2: PFRC4
3: PFRC8
4: PFRC7
"""

#df_co2_1_pfr_Masse = pd.read_excel('Daten/1_CO2_Masse.xlsm', sheet_name='2.soln_no_1_PFRC2')
#df_co2_1_pfr_Stoffmenge = pd.read_excel('Daten/1_CO2_Stoffmenge.xlsm', sheet_name='2.soln_no_1_PFRC2')

#df_co2_2_pfr_Masse = pd.read_excel('Daten/2_CO2_Masse.xlsm', sheet_name='9.soln_no_1_PFRC4')
#df_co2_2_pfr_Stoffmenge = pd.read_excel('Daten/2_CO2_Stoffmenge.xlsm', sheet_name='9.soln_no_1_PFRC4')

df_co2_3_pfr8_Masse = pd.read_excel('Daten/3_CO2_Masse.xlsm', sheet_name='24.soln_no_1_PFRC8')
df_co2_3_pfr8_Stoffmenge = pd.read_excel('Daten/3_CO2_Stoffmenge.xlsm', sheet_name='24.soln_no_1_PFRC8')

df_co2_4_pfr7_Masse = pd.read_excel('Daten/4_CO2_Masse.xlsm', sheet_name='21.soln_no_1_PFRC7')
df_co2_4_pfr7_Stoffmenge = pd.read_excel('Daten/4_CO2_Stoffmenge.xlsm', sheet_name='21.soln_no_1_PFRC7')

#df_co2_5_Masse = pd.read_excel('Daten/5_CO2_Masse.xlsm', sheet_name='2.soln_no_1_PFRC2')
#df_co2_5_Stoffmenge = pd.read_excel('Daten/5_CO2_Stoffmenge.xlsm', sheet_name='2.soln_no_1_PFRC2')

df_co2_6_pfr3_Masse = pd.read_excel('Daten/6_CO2_Masse.xlsm', sheet_name='6.soln_no_1_PFRC3')
df_co2_6_pfr3_Stoffmenge = pd.read_excel('Daten/6_CO2_Stoffmenge.xlsm', sheet_name='6.soln_no_1_PFRC3')

# ------------------------------------------------------- 

df_no_co2_1_pfr_Masse = pd.read_excel('Daten/1_kein_CO2_Masse.xlsm', sheet_name='2.soln_no_1_PFRC2')
df_no_co2_1_pfr_Stoffmenge = pd.read_excel('Daten/1_kein_CO2_Stoffmenge.xlsm', sheet_name='2.soln_no_1_PFRC2')

df_no_co2_2_pfr_Masse = pd.read_excel('Daten/2_kein_CO2_Masse.xlsm', sheet_name='9.soln_no_1_PFRC4')
df_no_co2_2_pfr_Stoffmenge = pd.read_excel('Daten/2_kein_CO2_Stoffmenge.xlsm', sheet_name='9.soln_no_1_PFRC4')

df_no_co2_3_pfr8_Masse = pd.read_excel('Daten/3_kein_CO2_Masse.xlsm', sheet_name='23.soln_no_1_PFRC8')
df_no_co2_3_pfr8_Stoffmenge = pd.read_excel('Daten/3_kein_CO2_Stoffmenge.xlsm', sheet_name='23.soln_no_1_PFRC8')

df_no_co2_4_pfr7_Masse = pd.read_excel('Daten/4_kein_CO2_Masse.xlsm', sheet_name='21.soln_no_1_PFRC7')
df_no_co2_4_pfr7_Stoffmenge = pd.read_excel('Daten/4_kein_CO2_Stoffmenge.xlsm', sheet_name='21.soln_no_1_PFRC7')

#df_no_co2_5_Masse = pd.read_excel('Daten/5_kein_CO2_Masse.xlsm', sheet_name='2.soln_no_1_PFRC2')
#df_no_co2_5_Stoffmenge = pd.read_excel('Daten/5_kein_CO2_Stoffmenge.xlsm', sheet_name='2.soln_no_1_PFRC2')

df_no_co2_6_pfr3_Masse = pd.read_excel('Daten/6_kein_CO2_Masse.xlsm', sheet_name='6.soln_no_1_PFRC3')
df_no_co2_6_pfr3_Stoffmenge = pd.read_excel('Daten/6_kein_CO2_Stoffmenge.xlsm', sheet_name='6.soln_no_1_PFRC3')

#%% Dataframes einlesen csv

df_co2_1_pfr_Stoffmenge = pd.read_csv('Daten_neu/1/CO2/Stoffmenge/2.soln_no_1_PFRC2.csv')
df_co2_1_pfr_Masse = pd.read_csv('Daten_neu/1/CO2/Masse/2.soln_no_1_PFRC2.csv')

df_co2_2_pfr_Stoffmenge = pd.read_csv('Daten_neu/2/CO2/Stoffmenge/9.soln_no_1_PFRC4.csv')
df_co2_2_pfr_Masse = pd.read_csv('Daten_neu/2/CO2/Masse/9.soln_no_1_PFRC4.csv')

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
#plt.show()

x_h2_no_co2_1 = df_no_co2_1_pfr_Stoffmenge[" Mole_fraction_H2_PFRC2_()"]
x_h2_no_co2_2 = df_no_co2_2_pfr_Stoffmenge[" Mole_fraction_H2_PFRC4_()"]
x_h2_no_co2_3 = df_no_co2_3_pfr8_Stoffmenge[" Mole_fraction_H2_PFRC8_()"]
x_h2_no_co2_4 = df_no_co2_4_pfr7_Stoffmenge[" Mole_fraction_H2_PFRC7_()"]
x_h2_no_co2_6 = df_no_co2_6_pfr3_Stoffmenge[" Mole_fraction_H2_PFRC3_()"]

plt.plot(x_h2_no_co2_1, label = "1")
plt.plot(x_h2_no_co2_2, label = "2")
plt.plot(x_h2_no_co2_3, label = "3")
plt.plot(x_h2_no_co2_4, label = "4")
plt.plot(x_h2_no_co2_6, label = "6")
plt.grid()
plt.tight_layout()
plt.legend()
#plt.show()
plt.close("all")

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

# --- Vergleich mit Experiment: Species-Mapping pro DF dynamisch bestimmen ---
## kein CO2-Fall:
df1, map1 = build_species_map(df_no_co2_1_pfr_Stoffmenge)
df2, map2 = build_species_map(df_no_co2_2_pfr_Stoffmenge)
df3, map3 = build_species_map(df_no_co2_3_pfr8_Stoffmenge)
df4, map4 = build_species_map(df_no_co2_4_pfr7_Stoffmenge)
df6, map6 = build_species_map(df_no_co2_6_pfr3_Stoffmenge)

comp_1 = get_normalized_composition_strict(df1, map1)
comp_2 = get_normalized_composition_strict(df2, map2)
comp_3 = get_normalized_composition_strict(df3, map3)
comp_4 = get_normalized_composition_strict(df4, map4)
comp_6 = get_normalized_composition_strict(df6, map6)

exp_data = {
    "Exp": [0.599,0.341,0.007,0.048],
    "1 CO2": comp_1.tolist(),
    "2 CO2": comp_2.tolist(),
    "3 CO2": comp_3.tolist(),
    "4 CO2": comp_4.tolist(),
    "6 CO2": comp_6.tolist(),
}

df_exp_data_no_CO2 = pd.DataFrame(exp_data, index = ["H2", "CO", "CH4", "CO2"])
#print(df_exp_data)

# Säulendiagramm (gruppiert)
ax = df_exp_data_no_CO2.plot(kind="bar", color=colors)
ax.set_xlabel("Spezies")
ax.set_ylabel("Molenbruch")
ax.set_title("Experiment vs. Simulation (CO₂-Fall)")
plt.xticks(rotation=0)
plt.tight_layout()
#plt.show()


# Optional speichern:
# plt.savefig("saeulendiagramm.png", dpi=300)

def tvd_scores(df, exp_col="Exp"):
    # Erwartet df mit Spalten: "Exp", "1 CO2", "2 CO2", ...
    exp = df[exp_col].astype(float)
    sims = df.drop(columns=[exp_col]).astype(float)
    # TVD = 0.5 * L1-Distanz der Verteilungen
    tvd = 0.5 * (sims.sub(exp, axis=0).abs()).sum(axis=0)
    return tvd.sort_values()  # kleiner = besser

def relative_error_scores(df, exp_col="Exp", eps=1e-6):
    """
    Berechnet mittleren relativen Fehler pro Simulation.
    
    df: DataFrame mit Spalten [exp_col, sim1, sim2, ...]
    exp_col: Name der Spalte mit den Experimentwerten
    eps: kleiner Wert, um Division durch 0 zu vermeiden
    """
    exp = df[exp_col].astype(float).replace(0, eps)   # kleine Werte absichern
    sims = df.drop(columns=[exp_col]).astype(float)

    # relativer Fehler: |sim - exp| / |exp|
    rel_errors = (sims.sub(exp, axis=0).abs().div(exp.abs(), axis=0))

    # Mittelwert über alle Zeilen → Score pro Simulation
    scores = rel_errors.mean(axis=0).sort_values()
    return scores  # kleiner = besser

def mse_scores(df, exp_col="Exp"):
    """
    Berechnet den mittleren quadratischen Fehler (MSE) pro Simulation.

    df: DataFrame mit Spalten [exp_col, sim1, sim2, ...]
    exp_col: Name der Spalte mit den Experimentwerten
    """
    exp = df[exp_col].astype(float)
    sims = df.drop(columns=[exp_col]).astype(float)

    # quadratischer Fehler: (sim - exp)^2
    sq_errors = (sims.sub(exp, axis=0)) ** 2

    # Mittelwert über alle Zeilen → MSE pro Simulation
    mse = sq_errors.mean(axis=0).sort_values()
    return mse  # kleiner = besser

scores = mse_scores(df_exp_data_no_CO2)
print("TVD pro Modell (0=perfekt, 1=schlecht):")
print(scores)
print("\nBestes Modell:", scores.idxmin(), "mit TVD =", float(scores.min()))

df_no_co2 = df_exp_data_no_CO2

## CO2-Fall:
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

df_exp_data_CO2 = pd.DataFrame(exp_data, index = ["H2", "CO", "CH4", "CO2"])
#print(df_exp_data)

# -------------------------------------------------
# Vorbereitung
# -------------------------------------------------
legend_labels = ["Experiment", "Simulation 1", "Simulation 2", "Simulation 3", "Simulation 4", "Simulation 6"]
col_map_in  = ["Exp", "1 CO2", "2 CO2", "3 CO2", "4 CO2", "6 CO2"]
col_map_out = ["Experiment", "Simulation 1", "Simulation 2", "Simulation 3", "Simulation 4", "Simulation 6"]

# robuste Spaltenselektion in fixer Reihenfolge
def select_cols(df):
    return [c for c in col_map_in if c in df.columns]

def rename_cols(df):
    return df.rename(columns=dict(zip(col_map_in, col_map_out)))

# Datenquellen
df_co2 = df_exp_data_CO2.copy()
df_no  = df_no_co2.copy()

# CH4 ×10
for d in (df_no, df_co2):
    if "CH4" in d.index:
        d.loc["CH4"] = d.loc["CH4"] * 10

# Index-Beschriftungen in MathText
species_labels = [r"$\mathrm{H_2}$", r"$\mathrm{CO}$", r"$\mathrm{CH_4}\cdot 10$", r"$\mathrm{CO_2}$"]
for d in (df_no, df_co2):
    d.index = species_labels

# Spaltenreihenfolge + Umbenennung für Legende/Farben
vis_no  = rename_cols(df_no.loc[:, select_cols(df_no)])
vis_co2 = rename_cols(df_co2.loc[:, select_cols(df_co2)])

# -------------------------------------------------
# Plot-Parameter wie Referenz
# -------------------------------------------------
gap = 0.01
bar_width = 0.10

def col_colors(df_like):
    label_to_idx = {name: i for i, name in enumerate(legend_labels)}
    idxs = [label_to_idx.get(c, 0) for c in df_like.columns]
    return [colors[i % len(colors)] for i in idxs]

# -------------------------------------------------
# Plot
# -------------------------------------------------
plt.close("all")
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

plots = [
    (vis_no.astype(float),  "ohne CO₂"),
    (vis_co2.astype(float), "mit CO₂"),
]

for i, (dfp, title) in enumerate(plots):
    ax = axes[i]
    x = np.arange(len(dfp.index))
    n_cols = len(dfp.columns)
    cols   = dfp.columns
    ccols  = col_colors(dfp)

    # Balken manuell zeichnen
    for j, col in enumerate(cols):
        offset = j * (bar_width + gap)
        ax.bar(x + offset, dfp[col].values, width=bar_width, color=ccols[j], label=col)

    # Gruppenzentrierte xticks
    group_span = n_cols * (bar_width + gap) - gap
    centers = x + group_span/2 - (bar_width + gap)/2
    ax.set_xticks(centers)
    ax.set_xticklabels(dfp.index, rotation=0)
    ax.tick_params(axis='x', which='major', labelsize=14)
    ax.tick_params(axis='y', which='major', labelsize=14)

    ax.set_title(title)
    ax.set_xlabel("Spezies")
    if i == 0:
        ax.set_ylabel(r"Molenbruch (CH₄ $\cdot$ 10)")
        ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle="dotted")
    ax.set_axisbelow(True)

# Legende rechts unten wie im Original
axes[1].legend(legend_labels, loc="upper right", fontsize=14)

plt.tight_layout()
plt.savefig("Bilder/Vergleich_Erweiterungen.png", dpi=300)
plt.close("all")

# -------------------------------------------------
# Kennzahlen
# -------------------------------------------------
scores = mse_scores(df_exp_data_CO2)  # Achtung: Funktion liefert MSE, nicht TVD
print("MSE pro Modell (0=perfekt, ↑=schlechter):")
print(scores)
print("\nBestes Modell:", scores.idxmin(), "mit MSE =", float(scores.min()))

#%% Temperaturen extrahieren
temp_co2_1 =  df_co2_1_pfr_Masse[' Temperature_PFRC2_(K)']
temp_co2_2 =  df_co2_2_pfr_Masse[' Temperature_PFRC4_(K)']
temp_co2_3 =  df_co2_3_pfr8_Masse[' Temperature_PFRC8_(K)']
temp_co2_4 =  df_co2_4_pfr7_Masse[' Temperature_PFRC7_(K)']
temp_co2_6 =  df_co2_6_pfr3_Masse[' Temperature_PFRC3_(K)']

temp_no_co2_1 =  df_no_co2_1_pfr_Masse[' Temperature_PFRC2_(K)']
temp_no_co2_2 =  df_no_co2_2_pfr_Masse[' Temperature_PFRC4_(K)']
temp_no_co2_3 =  df_no_co2_3_pfr8_Masse[' Temperature_PFRC8_(K)']
temp_no_co2_4 =  df_no_co2_4_pfr7_Masse[' Temperature_PFRC7_(K)']
temp_no_co2_6 =  df_no_co2_6_pfr3_Masse[' Temperature_PFRC3_(K)']

print(f"temp_co2_1: {temp_co2_1.iloc[-1]-273.15}")
print(f"temp_co2_2: {temp_co2_2.iloc[-1]-273.15}")
print(f"temp_co2_3: {temp_co2_3.iloc[-1]-273.15}")
print(f"temp_co2_4: {temp_co2_4.iloc[-1]-273.15}")
print(f"temp_co2_6: {temp_co2_6.iloc[-1]-273.15}")
print(f"temp_no_co2_1: {temp_no_co2_1.iloc[-1]-273.15}")
print(f"temp_no_co2_2: {temp_no_co2_2.iloc[-1]-273.15}")
print(f"temp_no_co2_3: {temp_no_co2_3.iloc[-1]-273.15}")
print(f"temp_no_co2_4: {temp_no_co2_4.iloc[-1]-273.15}")
print(f"temp_no_co2_6: {temp_no_co2_6.iloc[-1]-273.15}")


df_exp_data_no_CO2.loc["Temperatur Ausgang"] = [1300+273.15,temp_no_co2_1.iloc[-1],temp_no_co2_2.iloc[-1],temp_no_co2_3.iloc[-1],temp_no_co2_4.iloc[-1],temp_no_co2_6.iloc[-1]]
df_exp_data_no_CO2.loc["Temperatur 15"] = [1351.9+273.15,temp_no_co2_1.iloc[round(2/3 * len(temp_no_co2_1))],temp_no_co2_2.iloc[round(2/3 * len(temp_no_co2_2))],temp_no_co2_3.iloc[round(2/3 * len(temp_no_co2_3))],temp_no_co2_4.iloc[round(2/3 * len(temp_no_co2_4))],temp_no_co2_6.iloc[round(2/3 * len(temp_no_co2_6))]]
df_exp_data_no_CO2.loc["Temperatur 11"] = [1407.4+273.15,temp_no_co2_1.iloc[0],temp_no_co2_2.iloc[0],temp_no_co2_3.iloc[0],temp_no_co2_4.iloc[0],temp_no_co2_6.iloc[0]]

df_exp_data_CO2.loc["Temperatur Ausgang"] = [1342+273.15,temp_co2_1.iloc[-1],temp_co2_2.iloc[-1],temp_co2_3.iloc[-1],temp_co2_4.iloc[-1],temp_co2_6.iloc[-1]]
df_exp_data_CO2.loc["Temperatur 15"] = [1371.6+273.15,temp_co2_1.iloc[round(2/3 * len(temp_co2_1))],temp_co2_2.iloc[round(2/3 * len(temp_co2_2))],temp_co2_3.iloc[round(2/3 * len(temp_co2_3))],temp_co2_4.iloc[round(2/3 * len(temp_co2_4))],temp_co2_6.iloc[round(2/3 * len(temp_co2_6))]]
df_exp_data_CO2.loc["Temperatur 11"] = [1411.4+273.15,temp_co2_1.iloc[0],temp_co2_2.iloc[0],temp_co2_3.iloc[0],temp_co2_4.iloc[0],temp_co2_6.iloc[0]]

print(df_exp_data_no_CO2)
print(df_exp_data_CO2)

temp_rows = ["Temperatur Ausgang", "Temperatur 15", "Temperatur 11"]
legend_labels = ["Experiment", "Simulation 1", "Simulation 2", "Simulation 3", "Simulation 4", "Simulation 6"]

# Spaltenauswahl
def select_temp_cols(df):
    candidates = ["Exp", "1 CO2", "2 CO2", "3 CO2", "4 CO2", "6 CO2"]
    return [c for c in candidates if c in df.columns]

# Daten
have_no_co2 = 'df_exp_data_no_CO2' in globals() and all(r in df_exp_data_no_CO2.index for r in temp_rows)
have_co2    = 'df_exp_data_CO2'    in globals() and all(r in df_exp_data_CO2.index    for r in temp_rows)

rename_map = {"Exp": "Experiment",
              "1 CO2": "Simulation 1", "2 CO2": "Simulation 2",
              "3 CO2": "Simulation 3", "4 CO2": "Simulation 4",
              "6 CO2": "Simulation 6"}

vis_temp_no_co2 = None
vis_temp_co2 = None

if have_no_co2:
    cols_no = select_temp_cols(df_exp_data_no_CO2)
    vis_temp_no_co2 = df_exp_data_no_CO2.loc[temp_rows, cols_no].rename(columns=rename_map)

if have_co2:
    cols_co2 = select_temp_cols(df_exp_data_CO2)
    vis_temp_co2 = df_exp_data_CO2.loc[temp_rows, cols_co2].rename(columns=rename_map)

# Figure/Axes
if vis_temp_no_co2 is not None and vis_temp_co2 is not None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
elif vis_temp_co2 is not None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    axes = [ax]
elif vis_temp_no_co2 is not None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    axes = [ax]
else:
    raise RuntimeError("Keine Temperatur-Datenreihen gefunden.")

# Farben
def col_colors(df_like):
    label_to_idx = {name: i for i, name in enumerate(legend_labels)}
    idxs = [label_to_idx.get(c, 0) for c in df_like.columns]
    return [colors[i % len(colors)] for i in idxs]

# Plotparameter
gap = 0.01
bar_width = 0.10

plots = []
if vis_temp_no_co2 is not None:
    plots.append((vis_temp_no_co2.astype(float), "Temperatur ohne CO₂"))
if vis_temp_co2 is not None:
    plots.append((vis_temp_co2.astype(float), "Temperatur mit CO₂"))

# Balken zeichnen
for i, (df_temp, title) in enumerate(plots):
    ax = axes[i]
    x = np.arange(len(df_temp.index))
    n_cols = len(df_temp.columns)
    col_set = col_colors(df_temp)

    for j, col in enumerate(df_temp.columns):
        offset = j * (bar_width + gap)
        ax.bar(
            x + offset,
            df_temp[col].values,
            width=bar_width,
            label=col,
            color=col_set[j]
        )

    group_span = n_cols * (bar_width + gap) - gap
    centers = x + group_span / 2 - (bar_width + gap) / 2
    ax.set_xticks(centers)
    ax.set_xticklabels(df_temp.index, rotation=0)

    ax.set_title(title)
    ax.set_xlabel("Mess-/Positionspunkt", fontsize=12)
    if i == 0:
        ax.set_ylabel("Temperatur [K]")
        ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle="dotted")
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', which='major', labelsize=12)
    ax.tick_params(axis='y', which='major', labelsize=12)

# Legende nur im rechten Plot
"""
axes[-1].legend(
    handles=[plt.Rectangle((0,0),1,1,color=c) for c in col_colors(plots[-1][0])],
    labels=plots[-1][0].columns,
    loc="lower right",
    fontsize=12
)"""
axes[-1].legend(loc="lower right", fontsize=14)

plt.tight_layout()
plt.savefig("Bilder/Vergleich_Temperaturen.png", dpi=300)
# plt.show()

##% Ähnlichkeiten überprüfen
scores = mse_scores(df_exp_data_no_CO2)
print("MRE pro Modell (0=perfekt, 1=schlecht):")
print(scores)
print("\nBestes Modell:", scores.idxmin(), "mit TVD =", float(scores.min()))

scores = mse_scores(df_exp_data_CO2)
print("MRE pro Modell (0=perfekt, 1=schlecht):")
print(scores)
print("\nBestes Modell:", scores.idxmin(), "mit TVD =", float(scores.min()))

"""
                1 & 0.159 & 0.261 \\
                2 & 0.169 & 0.190 \\
                3 & 0.300 & 0.765 \\
                4 & \textbf{0.096} & \textbf{0.167} \\
                6 & 0.494 & 2.426 \\
"""