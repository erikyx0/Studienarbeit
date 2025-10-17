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
color5 = "#6B2D2D"
color6 = "#0B4915"
"""
color1 = "#446A87"  # gedecktes Stahlblau
color2 = "#5C735F"  # dunkles Olivgrün
color3 = "#8C8C8C"  # mittleres Grau
color4 = "#A65E2E"  # warmes Kupferbraun
color5 = "#7B3F3F"  # gedecktes Rotbraun
color6 = "#2F6F4F"  # dunkles Tannengrün
"""

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

# Säulendiagramm (gruppiert)
ax = df_exp_data_CO2.plot(kind="bar", color=colors)
ax.set_xlabel("Spezies")
ax.set_ylabel("Molenbruch")
ax.set_title("Experiment vs. Simulation (CO₂-Fall)")
plt.xticks(rotation=0)
plt.tight_layout()
#plt.show()

# Optional speichern:
# plt.savefig("saeulendiagramm.png", dpi=300)

scores = mse_scores(df_exp_data_CO2)
print("TVD pro Modell (0=perfekt, 1=schlecht):")
print(scores)
print("\nBestes Modell:", scores.idxmin(), "mit TVD =", float(scores.min()))

df_co2 = df_exp_data_CO2

## Plotten beider Fälle zusammen 
vis_no_co2 = df_no_co2.copy()
vis_co2    = df_co2.copy()
for d in (vis_no_co2, vis_co2):
    d.loc["CH4"] *= 10

plt.close("all")

fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

vis_no_co2.plot(kind="bar", ax=axes[0], color=colors, legend=False)
axes[0].set_title("ohne CO₂"); axes[0].set_xlabel("Spezies"); axes[0].set_ylabel("Molenbruch (CH₄ ×10)")
axes[0].tick_params(axis="x", rotation=0)

vis_co2.plot(kind="bar", ax=axes[1], color=colors, legend=False)
axes[1].set_title("mit CO₂"); axes[1].set_xlabel("Spezies")
axes[1].tick_params(axis="x", rotation=0)

handles, _ = axes[0].get_legend_handles_labels()
legend_labels = ["Experiment", "Simulation 1", "Simulation 2", "Simulation 3", "Simulation 4", "Simulation 6"]

species_labels = [r"H$_2$", "CO", r"CH$_4\cdot 10$", r"CO$_2$"]
for d in (df_no_co2, df_co2):
    d.index = species_labels
for a in axes:
    a.tick_params(axis="x", rotation=0)

for a in axes:
    a.set_axisbelow(True)

axes[1].legend(handles, legend_labels)

axes[0].grid(axis="y", linestyle = "dotted")
axes[1].grid(axis="y", linestyle = "dotted")
plt.tight_layout()
plt.savefig("Bilder/Vergleich_Erweiterungen", dpi=300)
plt.close("all")

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

#%% Plot Temperaturen
temp_rows = ["Temperatur Ausgang", "Temperatur 15", "Temperatur 11"]
legend_labels = ["Experiment", "Simulation 1", "Simulation 2", "Simulation 3", "Simulation 4", "Simulation 6"]

# Hilfsfunktion: selektiert vorhandene Spalten robust in richtiger Reihenfolge
def select_temp_cols(df):
    candidates = ["Exp", "1 CO2", "2 CO2", "3 CO2", "4 CO2", "6 CO2"]
    return [c for c in candidates if c in df.columns]

# Daten für NO-CO2 (falls vorhanden)
have_no_co2 = 'df_exp_data_no_CO2' in globals() and all(r in df_exp_data_no_CO2.index for r in temp_rows)
# Daten für CO2
have_co2    = 'df_exp_data_CO2'    in globals() and all(r in df_exp_data_CO2.index    for r in temp_rows)

# Baue die DataFrames in Plot-Form: Index = Temperatur-Zeilen, Columns = Exp/Sim...
dfs = []
if have_no_co2:
    cols_no = select_temp_cols(df_exp_data_no_CO2)
    vis_temp_no_co2 = df_exp_data_no_CO2.loc[temp_rows, cols_no]
    # Spalten-Namen hübsch für Legend:
    rename_map = {"Exp": "Experiment", "1 CO2": "Simulation 1", "2 CO2": "Simulation 2",
                  "3 CO2": "Simulation 3", "4 CO2": "Simulation 4", "6 CO2": "Simulation 6"}
    vis_temp_no_co2 = vis_temp_no_co2.rename(columns=rename_map)
else:
    vis_temp_no_co2 = None

if have_co2:
    cols_co2 = select_temp_cols(df_exp_data_CO2)
    vis_temp_co2 = df_exp_data_CO2.loc[temp_rows, cols_co2]
    rename_map = {"Exp": "Experiment", "1 CO2": "Simulation 1", "2 CO2": "Simulation 2",
                  "3 CO2": "Simulation 3", "4 CO2": "Simulation 4", "6 CO2": "Simulation 6"}
    vis_temp_co2 = vis_temp_co2.rename(columns=rename_map)
else:
    vis_temp_co2 = None

# Plot
if vis_temp_no_co2 is not None and vis_temp_co2 is not None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
elif vis_temp_co2 is not None:
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    axes = [axes]  # vereinheitlichen
elif vis_temp_no_co2 is not None:
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    axes = [axes]
else:
    raise RuntimeError("Keine Temperatur-Datenreihen gefunden. Prüfe, ob die drei Zeilen im/den DataFrame(s) vorhanden sind.")

# Farben (mind. so viele, wie es Spalten gibt)
def col_colors(df_like):
    # mappe auf die Reihenfolge in df_like.columns
    label_to_idx = {name: i for i, name in enumerate(legend_labels)}
    idxs = [label_to_idx.get(c, 0) for c in df_like.columns]
    return [colors[i % len(colors)] for i in idxs]

ax_i = 0
if vis_temp_no_co2 is not None:
    vis_temp_no_co2.plot(kind="bar", ax=axes[ax_i], color=col_colors(vis_temp_no_co2), legend=False)
    axes[ax_i].set_title("Temperatur ohne CO₂")
    axes[ax_i].set_xlabel("Mess-/Positionspunkt")
    axes[ax_i].set_ylabel("Temperatur [K]")
    axes[ax_i].tick_params(axis="x", rotation=0)
    axes[ax_i].set_axisbelow(True)
    axes[ax_i].grid(axis="y", linestyle="dotted")
    ax_i += 1

if vis_temp_co2 is not None:
    vis_temp_co2.plot(kind="bar", ax=axes[ax_i], color=col_colors(vis_temp_co2), legend=False)
    axes[ax_i].set_title("Temperatur mit CO₂")
    axes[ax_i].set_xlabel("Mess-/Positionspunkt")
    if len(axes) == 1:  # falls nur ein Plot, Y-Achse hier beschriften
        axes[ax_i].set_ylabel("Temperatur [K]")
    axes[ax_i].tick_params(axis="x", rotation=0)
    axes[ax_i].set_axisbelow(True)
    axes[ax_i].grid(axis="y", linestyle="dotted")

# Legend aus dem rechten (oder einzigen) Plot holen
handles, labels = axes[-1].get_legend_handles_labels()
if not handles:
    # Falls Legende aus blieb (legend=False), nimm die Spaltennamen und generiere Handles via Dummy-Plot:
    axes[-1].legend(vis_temp_co2.columns if vis_temp_co2 is not None else vis_temp_no_co2.columns,
                    loc="best")
else:
    axes[-1].legend(labels, loc="best")

plt.ylim(1200)

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