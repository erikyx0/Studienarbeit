import numpy as np
import pandas as pd
import unicodedata
import re
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------- Hilfsfunktionen -----------------
def _canon(s: str) -> str:
    """Unicode normalisieren, NBSP entfernen, Whitespace & Nicht-Alnum löschen, lower."""
    s = unicodedata.normalize("NFKC", str(s)).replace("\u00A0", " ")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^0-9a-zA-Z]+", "", s)
    return s.lower()

def _find_col_tolerant(df: pd.DataFrame, run: int, tokens: list[str]) -> str | None:
    """Finde Spalte, deren kanonisierter Name alle tokens enthält."""
    cans = {col: _canon(col) for col in df.columns}
    cands = [orig for orig, can in cans.items() if all(t in can for t in tokens)]
    if not cands:
        return None
    # Heuristik: kürzester kanonisierter Name zuerst
    cands.sort(key=lambda c: len(cans[c]))
    return cands[0]

# ----------------- Hauptfunktion -----------------
def build_heatmap_from_runs(
    dfs_by_run: dict[int, pd.DataFrame],
    *,
    species: str = "H2",
    x_axis: str = "Distance",
    runs: list[int] | range | None = None,
    x_points: int = 300,
    y_labels: dict[int, float | str] | None = None,
    pfrc_tag: str = "PFRC2",
    fill_along_x: bool = False,
):
    """
    Erzeugt eine Heatmap-Matrix (Zeilen = Runs, Spalten = x-Raster) für 'Mole_fraction_<species>'.

    Parameter
    ---------
    dfs_by_run : dict[int, DataFrame]
        Pro Run ein DataFrame mit den originalen (Excel-)Spalten.
    species : str
        Z. B. 'H2', 'CO', 'CO2', 'CH4'.
    x_axis : str
        Basisname der x-Achse. Beispiele: 'Distance', 'Axial_Position', 'Residence_time'.
        Die Funktion sucht tolerant nach einer Spalte, die Tokens [x_axis, pfrc_tag, run] enthält.
    runs : list|range|None
        Welche Runs verwenden. Default: alle Schlüssel aus dfs_by_run sortiert.
    x_points : int
        Anzahl Stützstellen fürs gemeinsame x-Raster.
    y_labels : dict[int, float|str]|None
        Optional: Zeilenbeschriftungen anstelle von Run-Nummern (z. B. Parameter je Run).
    pfrc_tag : str
        Tag, das in den Spaltennamen steckt (z. B. 'PFRC2').
    fill_along_x : bool
        True → NaN entlang der x-Achse interpolieren (vorsichtig benutzen).

    Returns
    -------
    heatmap_df : pd.DataFrame
        Zeilen = (optionale) y_labels oder Runs, Spalten = x-Raster (floats), Werte = Mole fraction species
    x_grid : np.ndarray
        Verwendetes x-Raster.
    row_index : pd.Index
        Zeilenindex (Runs oder y-Labels).
    """
    if runs is None:
        runs = sorted(dfs_by_run.keys())
    else:
        runs = list(runs)

    # 1) Pro Run x/y tolerant suchen und sammeln
    curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    missing = []
    for run in runs:
        df = dfs_by_run.get(run)
        if df is None:
            missing.append((run, "DF fehlt"))
            continue

        # Tokens für x-Spalte und species-Spalte
        x_tokens = [_canon(x_axis), _canon(pfrc_tag), f"run{run}"]
        y_tokens = ["mole", "fraction", _canon(species), _canon(pfrc_tag), f"run{run}"]

        x_col = _find_col_tolerant(df, run, x_tokens)
        y_col = _find_col_tolerant(df, run, y_tokens)

        if x_col is None or y_col is None:
            missing.append((run, f"x={x_col}, y={y_col}"))
            continue

        x = pd.to_numeric(df[x_col], errors="coerce").to_numpy()
        y = pd.to_numeric(df[y_col], errors="coerce").to_numpy()
        m = ~np.isnan(x) & ~np.isnan(y)
        x, y = x[m], y[m]
        if x.size < 2:
            missing.append((run, "zu wenige Punkte"))
            continue

        # sicherstellen: x monoton für Interpolation
        order = np.argsort(x)
        curves[run] = (x[order], y[order])

    if missing:
        for run, info in missing:
            print(f"[Hinweis] Run {run}: übersprungen ({info})")
    if not curves:
        raise RuntimeError("Keine verwertbaren Kurven gefunden – prüfe Spaltennamen/Tokens.")

    # 2) gemeinsames x-Raster bestimmen
    xmin = min(x.min() for x, _ in curves.values())
    xmax = max(x.max() for x, _ in curves.values())
    x_grid = np.linspace(xmin, xmax, int(x_points))

    # 3) Interpolation auf Raster (außerhalb → NaN)
    rows = []
    row_keys = []
    for run in runs:
        if run not in curves:
            continue
        x, y = curves[run]
        yg = np.interp(x_grid, x, y, left=np.nan, right=np.nan)
        rows.append(yg)
        row_keys.append(y_labels.get(run, run) if y_labels else run)

    heatmap = np.vstack(rows)  # shape: (n_runs, len(x_grid))
    heatmap_df = pd.DataFrame(heatmap, index=row_keys, columns=x_grid)

    # Optional Lücken entlang x schließen
    if fill_along_x:
        heatmap_df = heatmap_df.interpolate(axis=1, limit_direction="both")

    # Reihenfolge/Index
    try:
        heatmap_df = heatmap_df.sort_index()
    except Exception:
        pass

    return heatmap_df, x_grid, heatmap_df.index


# ----------------- Beispiel-Nutzung & Plot -----------------
# Annahme: dfs_by_run = {1: df1, 2: df2, ..., 21: df21} existiert bereits.

# 1) Heatmap mit Distance (x) und Run (y), Wert = H2
# heatmap_df, x_grid, y_index = build_heatmap_from_runs(
#     dfs_by_run,
#     species="H2",
#     x_axis="Distance",
#     runs=range(1,22),
#     x_points=300,
#     y_labels=None,       # oder z.B. {1: 900, 2: 950, ...} für Parameter als y-Achse
#     pfrc_tag="PFRC2",
#     fill_along_x=False
# )

# 2) Plotten
# plt.figure(figsize=(12, 6))
# sns.heatmap(
#     heatmap_df,
#     cmap="viridis",
#     cbar_kws={"label": f"Mole fraction { 'H₂' if 'H2'=='H2' else species }"}
# )
# plt.xlabel("Distance (m)")
# plt.ylabel("Run #")
# plt.title(f"Heatmap: Mole fraction {species} über {x_axis} und Runs")
# plt.tight_layout()
# plt.show()
