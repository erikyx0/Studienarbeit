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

def load_runs(file: str, runs=range(1,22)) -> pd.DataFrame:
    """Lade alle gewünschten Runs in ein einheitliches DataFrame [Distance, H2, Run]."""
    data_long = []

    for i in runs:
        sheetname = f"{i+8}.soln_no_1_PFRC2_Run#{i}"

        # Erstmal roh laden, Header suchen
        df_raw = pd.read_excel(file, sheet_name=sheetname, header=None, engine="openpyxl")
        hdr_row = find_header_row(df_raw, i)
        df = pd.read_excel(file, sheet_name=sheetname, header=hdr_row, engine="openpyxl")

        dist_name = f"Distance_PFRC2_Run#{i}_(m)"
        h2_name   = f"Mole_fraction_H2_PFRC2_Run#{i}_()"

        dist_col = get_col_by_canonical(df, dist_name)
        h2_col   = get_col_by_canonical(df, h2_name)

        if not dist_col or not h2_col:
            print(f"[Warnung] Run {i}: Spalten nicht gefunden in Sheet '{sheetname}'.")
            print("  Beispiel-Spalten:", [repr(c) for c in list(df.columns)[:8]])
            continue

        d = pd.DataFrame({
            "Distance_m": df[dist_col],
            "H2": df[h2_col],
            "Run": i
        }).dropna()

        data_long.append(d)

    if not data_long:
        raise RuntimeError("Keine Daten gefunden – prüfe Sheet-/Spaltennamen.")

    return pd.concat(data_long, ignore_index=True)
