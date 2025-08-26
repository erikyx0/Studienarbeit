import pandas as pd

def df_to_table(df, columns, rounding, *,
                delimiter=" & ",
                decimal_sep=".",
                latex=True,
                filepath=None,
                encoding="utf-8"):
    """
    Erstellt LaTeX- oder CSV-Tabellen.

    Parameters
    ----------
    df : pandas.DataFrame
        Eingabedaten.
    columns : list[str]
        Spaltenauswahl in der gewünschten Reihenfolge.
    rounding : list
        Formatangaben je Spalte.
    delimiter : str, optional
        Spaltentrennzeichen (für CSV oder LaTeX). Default " & ".
    decimal_sep : str, optional
        Dezimaltrennzeichen ("." oder ","). Default ".".
    latex : bool, optional
        True → LaTeX-Tabellen, False → CSV. Default True.
    filepath : str or Path, optional
        Falls angegeben, wird das Ergebnis in diese Datei geschrieben.
    encoding : str, optional
        Zeichencodierung beim Schreiben in Datei. Default "utf-8".

    Returns
    -------
    str
        Der generierte Tabellen-String (wird auch zurückgegeben).
    """

    if len(columns) != len(rounding):
        raise ValueError("columns und rounding müssen gleich lang sein.")

    # Kopf
    header  = delimiter.join(columns)
    lines   = []
    if latex:
        header += r" \\ \hline"
        colspec = " | ".join("c" for _ in columns)
        lines   = [rf"\begin{{tabular}}{{{colspec}}}", r"\hline", header]
        tail    = [r"\hline", r"\end{tabular}"]
    else:
        lines, tail = [header], []

    # Hilfsfunktionen ----------------------------------------------
    def fmt(value, spec):
        """Formatiert value gemäss spec (siehe Docstring)."""
        if pd.isnull(value):
            return ""
        if spec == "raw":
            return str(value)

        # --- numerische Formate ---
        if isinstance(spec, int):
            s = f"{value:.{spec}f}"
        elif isinstance(spec, str):
            s = f"{value:{spec}}"
        elif isinstance(spec, tuple):
            kind, num = spec
            if kind == "e":           # wissenschaftliche Notation
                s = f"{value:.{num}e}"
            elif kind == "f":         # Festpunkt
                s = f"{value:.{num}f}"
            elif kind == "sig":       # signifikante Stellen
                s = f"{value:.{num}g}"
            else:
                raise ValueError(f"Unbekanntes Format-Tuple: {spec}")
        else:
            raise TypeError("rounding-Eintrag muss int, str, tuple oder 'raw' sein")

        # Dezimalpunkt ersetzen?
        return s.replace(".", ",") if decimal_sep == "," else s

    # Zeilen generieren ---------------------------------------------
    for _, row in df.iterrows():
        cells = [fmt(row[col], spec) for col, spec in zip(columns, rounding)]
        line  = delimiter.join(cells) + (r" \\" if latex else "")
        lines.append(line)

    lines.extend(tail)
    output = "\n".join(lines)

    # Falls filepath angegeben → speichern
    if filepath is not None:
        with open(filepath, "w", encoding=encoding) as f:
            f.write(output)

    return output
