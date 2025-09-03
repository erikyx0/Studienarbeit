import re
import unicodedata
import pandas as pd

def canonical(s: str) -> str:
    """Spaltennamen robust normalisieren (Whitespaces, Sonderzeichen, Groß/Klein)."""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("\u00A0", " ").strip()  # NBSP entfernen
    s = re.sub(r"\s+", "", s)             # alle Whitespaces raus
    s = re.sub(r"[^A-Za-z0-9]+", "", s)   # nur alphanumerisch
    return s.lower()

def find_header_row(df: pd.DataFrame, run: int) -> int:
    """Falls Header nicht in Zeile 0: Suche eine Zeile mit typischen Tokens."""
    target_tokens = [
        f"distance_pfrc2_run#{run}_",
        f"mole_fraction_h2_pfrc2_run#{run}_"
    ]
    target_tokens = [canonical(t) for t in target_tokens]

    for r in range(min(10, len(df))):
        row_vals = [canonical(v) for v in df.iloc[r].tolist()]
        score = sum(any(tok in val for val in row_vals) for tok in target_tokens)
        if score >= 1:
            return r
    return 0

def get_col_by_canonical(df: pd.DataFrame, target_name: str) -> str | None:
    """Gib den echten Spaltennamen zurück, der kanonisch zu target_name passt."""
    target_can = canonical(target_name)
    mapping = {canonical(c): c for c in df.columns}
    return mapping.get(target_can)

# in excel_loader.py
def load_runs(file: str, runs=range(1,22), mode="long", species="H2"):
    """
    mode='long'  → Distance_m, <species>, Run  (schlank für Plots/Heatmap)  [aktuell]
    mode='raw'   → alle Originalspalten je Sheet + Run
    """
    import pandas as pd
    data_parts = []

    for i in runs:
        sheetname = f"{i+8}.soln_no_1_PFRC2_Run#{i}"
        # (… dein bestehender Header-Fix etc. bleibt gleich …)
        df_raw = pd.read_excel(file, sheet_name=sheetname, header=None, engine="openpyxl")
        hdr_row = find_header_row(df_raw, i)
        df = pd.read_excel(file, sheet_name=sheetname, header=hdr_row, engine="openpyxl")

        if mode == "raw":
            df["Run"] = i
            data_parts.append(df)
            continue

        # mode == 'long'
        dist_name = f"Distance_PFRC2_Run#{i}_(m)"
        spec_name = f"Mole_fraction_{species}_PFRC2_Run#{i}_()"
        dist_col = get_col_by_canonical(df, dist_name)
        spec_col = get_col_by_canonical(df, spec_name)
        if not dist_col or not spec_col:
            print(f"[Warnung] Run {i}: Spalten nicht gefunden in Sheet '{sheetname}'.")
            continue

        d = pd.DataFrame({
            "Distance_m": df[dist_col],
            species: df[spec_col],
            "Run": i
        }).dropna()
        data_parts.append(d)

    if not data_parts:
        raise RuntimeError("Keine Daten gefunden – prüfe Sheet-/Spaltennamen.")

    out = pd.concat(data_parts, ignore_index=True)
    return out
