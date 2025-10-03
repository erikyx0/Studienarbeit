# pfr_psrchain.py
# Plug-Flow-Reaktor als Kette von PSRs (IdealGasConstPressureReactor) – Cantera 3.x
# Lizenz: MIT

from __future__ import annotations
from typing import List, Optional, Union, Dict, Sequence
import numpy as np
import cantera as ct
import timeit

try:
    import pandas as pd
except ImportError:
    pd = None  # DataFrame-Export optional


class PlugFlowReactorPSRChain:
    """
    PFR-Approximation als N PSRs in Serie (konstanter Querschnitt, fester Massenstrom).
    Jedes Segment i hat Volumen V_i = A * dz_i. Die lokale mittlere Verweilzeit
    tau_i ~ dz_i / u_i mit u_i = mdot / (rho_i * A).

    Annahmen:
      - 1D Plug-Flow (axiale Diskretisierung über PSR-Kette)
      - Konstanter Querschnitt A [m^2]
      - Vorgegebener Massenstrom mdot [kg/s]
      - Stationärer Betrieb (Netz wird auf steady state gelöst)
      - Optionale Wärmeverluste je Segment (U_i, A_wall_i) gegen T_env

    Parameter
    ---------
    mechanism : str
        Mechanismusdatei, z. B. "gri30.yaml".
    T0, P0 : float
        Eintrittszustand [K], [Pa].
    composition : str | Dict[str, float]
        Zuströms-Zusammensetzung (Cantera-String oder dict).
    mdot : float
        Massenstrom [kg/s].
    area : float
        Querschnittsfläche A [m^2].
    length : float
        Reaktorlänge L [m].
    n_segments : int
        Anzahl Segmente (gleich lange Segmente), falls segment_lengths nicht gegeben.
    segment_lengths : Sequence[float] | None
        Individuelle Segmentlängen (Summe = length). Überschreibt n_segments.
    U : float | Sequence[float] | None
        Gesamtwärmeübergang [W/(m^2 K)], skalar oder je Segment. None = adiabatisch.
    A_wall : float | Sequence[float] | None
        Wärmeübertragerfläche [m^2], skalar oder je Segment. Muss gesetzt sein, wenn U gesetzt ist.
    T_env : float
        Umgebungstemperatur [K] für Wärmeverlust.
    species_to_record : list[str] | None
        Liste Spezies, die gespeichert werden.
    loglevel : int
        Cantera-Loglevel (0 = ruhig).
    """

    def __init__(
        self,
        mechanism: str,
        T0: float,
        P0: float,
        composition: Union[str, Dict[str, float]],
        mdot: float,
        area: float,
        length: float,
        n_segments: int = 200,
        segment_lengths: Optional[Sequence[float]] = None,
        U: Optional[Union[float, Sequence[float]]] = None,
        A_wall: Optional[Union[float, Sequence[float]]] = None,
        T_env: float = 300.0,
        species_to_record: Optional[List[str]] = None,
        loglevel: int = 0,
    ) -> None:
        # Eingaben
        self.mechanism = mechanism
        self.T0 = float(T0)
        self.P0 = float(P0)
        self.composition = composition
        self.mdot = float(mdot)
        self.area = float(area)
        self.length = float(length)
        self.T_env = float(T_env)
        self.loglevel = int(loglevel)
        self.species_to_record = species_to_record or [
            "CH4", "O2", "CO2", "H2O", "CO", "H2", "OH"
        ]

        # Diskretisierung
        if segment_lengths is not None:
            dz = np.asarray(segment_lengths, dtype=float)
            if dz.ndim != 1 or dz.size < 1:
                raise ValueError("segment_lengths muss 1D und nicht leer sein.")
            if not np.isclose(dz.sum(), self.length, rtol=1e-10, atol=1e-12):
                raise ValueError("Summe(segment_lengths) muss == length sein.")
        else:
            if n_segments < 1:
                raise ValueError("n_segments muss ≥ 1 sein.")
            dz = np.full(int(n_segments), self.length / int(n_segments), dtype=float)

        self.dz = dz
        self.n = dz.size
        self.z = np.concatenate([[0.0], np.cumsum(dz)])

        # Wärmeparameter prüfen/aufbereiten
        self.U = None
        self.Aw = None
        if (U is None) ^ (A_wall is None):
            raise ValueError("U und A_wall entweder beide None (adiabat) oder beide gesetzt.")
        if U is not None:
            self.U = np.full(self.n, float(U), dtype=float) if np.isscalar(U) else np.asarray(U, dtype=float)
            self.Aw = np.full(self.n, float(A_wall), dtype=float) if np.isscalar(A_wall) else np.asarray(A_wall, dtype=float)
            if self.U.size != self.n or self.Aw.size != self.n:
                raise ValueError("U und A_wall müssen skalar oder Länge n_segments haben.")

        # Ergebniscontainer
        self._T = np.zeros(self.n + 1)
        self._P = np.zeros(self.n + 1)
        self._tau = np.zeros(self.n + 1)
        self._X: Dict[str, np.ndarray] = {s: np.zeros(self.n + 1) for s in self.species_to_record}

        # Cantera-Objekte (intern)
        self._reactors: List[ct.IdealGasConstPressureReactor] = []
        self._net: Optional[ct.ReactorNet] = None

    # ------------------------ Hauptlauf ------------------------

    def run(self) -> None:
        import cantera as ct
        import numpy as np

        # 1) Startzustand als "laufendes" Inlet
        gas_in = ct.Solution(self.mechanism, loglevel=self.loglevel)
        gas_in.TPX = self.T0, self.P0, self.composition

        # Ergebnisse an z=0
        self._T[0] = gas_in.T
        self._P[0] = gas_in.P
        for s in self._X:
            if s in gas_in.species_names:
                self._X[s][0] = gas_in[s].X[0]

        tau_cum = 0.0

        # 2) Marching über Segmente (jedes Segment: eigenes kleines Netz)
        for i in range(self.n):
            # Fixes Inlet-Reservoir mit *gefrorenem* Zustand = Outlet des vorigen Segments
            inlet_res = ct.Reservoir(gas_in, name=f"inlet_seg_{i}")

            # Downstream-Reservoir (nur zum Abführen; Zustand egal, aber stabil)
            gas_out = ct.Solution(self.mechanism, loglevel=self.loglevel)
            gas_out.TPX = gas_in.T, gas_in.P, gas_in.X
            outlet_res = ct.Reservoir(gas_out, name=f"outlet_seg_{i}")

            # PSR für Segment i
            gseg = ct.Solution(self.mechanism, loglevel=self.loglevel)
            gseg.TPX = gas_in.T, gas_in.P, gas_in.X  # **warm start**
            r = ct.IdealGasConstPressureReactor(gseg, energy="on", name=f"psr_{i}")
            r.volume = self.area * self.dz[i]

            # Ein-/Auslass (fixer mdot)
            ct.MassFlowController(inlet_res, r, mdot=self.mdot)
            ct.MassFlowController(r, outlet_res, mdot=self.mdot)

            # Optional: Wärmeverlust an Umgebung
            if self.U is not None:
                env_g = ct.Solution(self.mechanism, loglevel=self.loglevel)
                env_g.TPX = self.T_env, self.P0, self.composition
                env = ct.Reservoir(env_g, name=f"env_{i}")
                ct.Wall(r, env, A=float(self.Aw[i]), U=float(self.U[i]))

            # Kleines Netz nur mit DIESEM Reaktor lösen
            net = ct.ReactorNet([r])
            # Toleranzen (etwas relaxter ist oft deutlich schneller)
            net.rtol = 1e-6
            net.atol = 1e-12
            net.advance_to_steady_state()

            # Profil schreiben
            self._T[i + 1] = r.T
            self._P[i + 1] = r.thermo.P

            rho = r.thermo.density
            u = self.mdot / (rho * self.area)
            if u <= 0:
                raise RuntimeError("Nichtpositive Geschwindigkeit u – prüfe mdot/area/State.")
            tau_cum += self.dz[i] / u
            self._tau[i + 1] = tau_cum

            for s in self._X:
                if s in r.thermo.species_names:
                    self._X[s][i + 1] = r.thermo[s].X[0]

            # 3) Update des "laufenden" Inlet-Zustands für das nächste Segment
            gas_in = ct.Solution(self.mechanism, loglevel=self.loglevel)
            gas_in.TPX = r.T, r.thermo.P, r.thermo.X

    # ------------------------ Zugriff/Export ------------------------

    @property
    def z_profile(self) -> np.ndarray:
        return self.z

    @property
    def T_profile(self) -> np.ndarray:
        return self._T

    @property
    def P_profile(self) -> np.ndarray:
        return self._P

    @property
    def tau_profile(self) -> np.ndarray:
        """Kumulative Verweilzeit entlang z [s]."""
        return self._tau

    def X_profile(self, species: str) -> np.ndarray:
        if species not in self._X:
            raise KeyError(f"Spezies '{species}' wurde nicht geloggt. "
                           f"Passe species_to_record beim Konstruktor an.")
        return self._X[species]

    def as_dataframe(self):
        if pd is None:
            raise RuntimeError("pandas ist nicht installiert (pip install pandas).")
        data = {"z_m": self.z, "T_K": self._T, "P_Pa": self._P, "tau_s": self._tau}
        for s, arr in self._X.items():
            data[f"X_{s}"] = arr
        return pd.DataFrame(data)

    # ------------------------ Utilities ------------------------

    def save_csv(self, path: str) -> None:
        """Speichere Profile als CSV (erfordert pandas)."""
        df = self.as_dataframe()
        df.to_csv(path, index=False)

class NonReactiveMixer:
    """
    Stationärer Mischer für ideale Gasgemische (kein Reaktionsumsatz).

    Annahmen:
      - Alle Ströme sind Gasphasen und verwenden DENSELBEN Mechanismus.
      - Keine Reaktion, nur Mischung + (optional) Wärmestrom Qdot.
      - Druck des Outlets wird vorgegeben (P_out). Standard: P_out = P des ersten Stroms.

    API (Kurz):
      m = NonReactiveMixer(mechanism="gri30.yaml", P_out=None)
      m.add_stream(mdot, T, P, composition)  # mehrfach aufrufen
      m.run(Qdot=0.0)                        # W (positiv: in den Mischer, negativ: Verlust)
      # Zugriff:
      m.mdot_out, m.T_out, m.P_out, m.X_out (dict), m.Y_out (dict), m.h_out

    Details:
      - Y_out = mdot-gewichtete Massfraktionen
      - h_out = (Σ mdot_i * h_i + Qdot) / mdot_tot
      - T_out: Lösung von h_mix(T, P_out, X_out) = h_out via Bisektion
    """

    def __init__(self, mechanism: str, P_out: Optional[float] = None, loglevel: int = 0) -> None:
        self.mechanism = mechanism
        self.P_out_user = P_out
        self.loglevel = int(loglevel)
        self._streams: List[Tuple[float, float, float, Union[str, Dict[str, float]]]] = []
        # Ergebnisse
        self._ran = False
        self.mdot_out: float = 0.0
        self.T_out: float = np.nan
        self.P_out: float = np.nan
        self.h_out: float = np.nan
        self._X_out_vec = None  # numpy vector in Mechanismus-Reihenfolge
        self._species = None    # Liste der Speziesnamen in Mechanismus-Reihenfolge

    # -------------------------- Input --------------------------

    def add_stream(self, mdot: float, T: float, P: float, composition: Union[str, Dict[str, float]]) -> None:
        """Einen Zuström hinzufügen: mdot [kg/s], T [K], P [Pa], composition (Cantera-String oder dict)."""
        if mdot <= 0.0:
            raise ValueError("mdot muss > 0 sein.")
        self._streams.append((float(mdot), float(T), float(P), composition))
        self._ran = False

    # -------------------------- Solve --------------------------

    def run(self, Qdot: float = 0.0) -> None:
        """
        Rechnung ausführen.
        Qdot [W]: Wärmestrom in den Mischer (positiv: heizen; negativ: Wärmeverlust).
        """
        if not self._streams:
            raise RuntimeError("Keine Zuströme. add_stream(...) zuerst aufrufen.")

        # Cantera-Objekt für Eigenschaften
        g = ct.Solution(self.mechanism, loglevel=self.loglevel)
        ns = g.n_species
        self._species = list(g.species_names)

        # Gemischbildung: mdot-gewichtete Y_out + Energiebilanz
        mdot_tot = 0.0
        Y_mix = np.zeros(ns)
        Hin = 0.0  # Σ mdot_i * h_i

        # Wir wählen P_out: falls nicht gesetzt, nehme P des ersten Stroms
        P_out = self.P_out_user if self.P_out_user is not None else self._streams[0][2]

        for mdot, T, P, X in self._streams:
            mdot_tot += mdot
            g.TPX = T, P, X
            # Y_i als Vektor
            Y_i = g.Y  # np.array (Massfraktionen)
            Y_mix += mdot * Y_i
            Hin += mdot * g.enthalpy_mass  # J/kg * kg/s = W

        Y_mix /= mdot_tot
        # Sicherheits-Normalisierung
        Y_mix = np.clip(Y_mix, 0.0, 1.0)
        s = Y_mix.sum()
        if s <= 0:
            raise RuntimeError("Zusammensetzung degeneriert (Summe Y <= 0).")
        Y_mix /= s

        # Ziel-Enthalpie (massenspezifisch) aus Energiebilanz
        h_target = (Hin + Qdot) / mdot_tot  # J/kg

        # Aus Y -> X für das Ziel-Gemisch (bei P_out, T guess) benötigt Cantera T
        # Wir lösen T so, dass h(T, P_out, Y_mix) = h_target.
        # Robuste Bisektion zwischen T_lo..T_hi.
        T_lo, T_hi = self._temperature_bracket(h_target, P_out, Y_mix, g)

        T_out = self._solve_T_for_enthalpy(h_target, P_out, Y_mix, g, T_lo, T_hi)

        # Endzustand setzen + Speichern
        g.TPY = T_out, P_out, Y_mix
        self.mdot_out = mdot_tot
        self.T_out = T_out
        self.P_out = P_out
        self.h_out = g.enthalpy_mass
        self._X_out_vec = g.X.copy()
        self._ran = True

    # -------------------------- Output --------------------------

    def X_out(self) -> Dict[str, float]:
        """Molenbrüche als dict {Spezies: X}."""
        self._assert_ran()
        return {sp: float(x) for sp, x in zip(self._species, self._X_out_vec)}

    def Y_out(self) -> Dict[str, float]:
        """Massfraktionen als dict {Spezies: Y}."""
        self._assert_ran()
        g = ct.Solution(self.mechanism, loglevel=self.loglevel)
        g.TPX = self.T_out, self.P_out, self.X_out()
        return {sp: float(y) for sp, y in zip(self._species, g.Y)}

    def as_dataframe(self):
        """Kompakte tabellarische Übersicht des Outlet-Zustands (erfordert pandas)."""
        self._assert_ran()
        if pd is None:
            raise RuntimeError("pandas ist nicht installiert (pip install pandas).")
        g = ct.Solution(self.mechanism, loglevel=self.loglevel)
        g.TPX = self.T_out, self.P_out, self.X_out()
        data = {
            "property": ["mdot_out_kg_s", "T_out_K", "P_out_Pa", "h_out_J_kg", "cp_out_J_kgK", "rho_out_kg_m3"],
            "value":    [self.mdot_out, self.T_out, self.P_out, self.h_out, g.cp_mass, g.density],
        }
        df = pd.DataFrame(data)
        return df

    # -------------------------- Interna --------------------------

    def _temperature_bracket(self, h_target: float, P_out: float, Y: np.ndarray, g: ct.Solution) -> Tuple[float, float]:
        """
        Suche einen Temperaturbereich [T_lo, T_hi], in dem h(T) - h_target das Vorzeichen wechselt.
        Start mit grober Spanne um die Inlet-Temperaturen.
        """
        # Heuristik: min/max Inlet-T ± Puffer
        T_in = [T for _, T, _, _ in self._streams]
        T_lo = max(50.0, min(T_in) - 200.0)
        T_hi = max(T_in) + 2000.0

        def h_minus(T):
            g.TPY = T, P_out, Y
            return g.enthalpy_mass - h_target

        f_lo = h_minus(T_lo)
        f_hi = h_minus(T_hi)
        # Falls kein Vorzeichenwechsel: erweitern
        expand = 0
        while f_lo * f_hi > 0 and expand < 10:
            T_lo = max(20.0, 0.5 * T_lo)
            T_hi = T_hi * 1.5
            f_lo = h_minus(T_lo)
            f_hi = h_minus(T_hi)
            expand += 1

        if f_lo * f_hi > 0:
            # Immer noch kein Wechsel -> nimm den näheren Rand (degenerierter Fall)
            # Trotzdem liefern wir ein Intervall zurück; Root-Suche klippt dann implizit.
            pass
        return T_lo, T_hi

    def _solve_T_for_enthalpy(
        self, h_target: float, P_out: float, Y: np.ndarray, g: ct.Solution, T_lo: float, T_hi: float
    ) -> float:
        """Bisektion für h(T)-h_target = 0 auf [T_lo, T_hi]."""
        def f(T):
            g.TPY = T, P_out, Y
            return g.enthalpy_mass - h_target

        f_lo = f(T_lo)
        f_hi = f(T_hi)
        # Wenn kein Vorzeichenwechsel: projiziere auf den Rand mit kleinerem |f|
        if f_lo * f_hi > 0:
            return T_lo if abs(f_lo) < abs(f_hi) else T_hi

        for _ in range(80):
            T_mid = 0.5 * (T_lo + T_hi)
            f_mid = f(T_mid)
            if abs(f_mid) < 1e-3:  # ~1 J/kg Toleranz
                return T_mid
            if f_lo * f_mid <= 0:
                T_hi, f_hi = T_mid, f_mid
            else:
                T_lo, f_lo = T_mid, f_mid
        return 0.5 * (T_lo + T_hi)

    def _assert_ran(self):
        if not self._ran:
            raise RuntimeError("Mixer noch nicht gelaufen. Rufe .run() zuerst auf.")

class CSTR:
    """
    Einfacher, stationärer/instationärer CSTR (const-p) für ideale Gasgemische.

    Annahmen:
      - Ein Gas-Inlet (mdot_in, T_in, P_in, composition)
      - Ein Outlet mit gleichem Massenstrom (stationär)
      - Druckführung über Downstream-Reservoir (≈ P_set)
      - Keine mechanische Wellen-/Rührerarbeit (nur Wärme)

    Wärmeführung (exakt EINE Variante):
      1) Konvektiv: U [W/m²K], A_wall [m²], T_env [K]
      2) Direkt:    Qdot_target [W]  (positiv: Heizleistung in den Reaktor;
                                      negativ: Wärmeverlust aus dem Reaktor)

    Parameter
    ---------
    mechanism : str            z. B. "gri30.yaml"
    V : float                  Reaktorvolumen [m³]
    P_set : float              Soll-Druck (Pa), via Downstream-Reservoir
    mdot_in : float            Massenstrom Inlet [kg/s]
    T_in : float               Inlettemperatur [K]
    P_in : float               Inletdruck [Pa] (meist = P_set)
    composition_in : str|dict  Inlet-Zusammensetzung
    species_to_record : list[str]  Spezies, die geloggt werden
    U, A_wall, T_env : Wärmeübergang (konvektiv) – exklusiv zu Qdot_target
    Qdot_target : float|None   Ziel-Wärmestrom (W) – exklusiv zu U/A_wall
    rtol, atol : float         CVODE-Toleranzen
    loglevel : int             Cantera-Loglevel
    """

    def __init__(
        self,
        mechanism: str,
        V: float,
        P_set: float,
        mdot_in: float,
        T_in: float,
        P_in: float,
        composition_in: Union[str, Dict[str, float]],
        *,
        species_to_record: Optional[list[str]] = None,
        # Wärmeführung (Variante A)
        U: Optional[float] = None,
        A_wall: Optional[float] = None,
        T_env: Optional[float] = None,
        # Wärmeführung (Variante B)
        Qdot_target: Optional[float] = None,
        # Numerik
        rtol: float = 1e-6,
        atol: float = 1e-12,
        loglevel: int = 0,
    ) -> None:
        self.mechanism = mechanism
        self.V = float(V)
        self.P_set = float(P_set)
        self.mdot_in = float(mdot_in)
        self.T_in = float(T_in)
        self.P_in = float(P_in)
        self.composition_in = composition_in
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.loglevel = int(loglevel)

        self.species_to_record = species_to_record or ["CH4", "O2", "CO2", "H2O", "CO", "H2", "OH"]

        # Wärmeführung – Exklusivität prüfen
        have_conv = (U is not None) or (A_wall is not None) or (T_env is not None)
        have_qdot = (Qdot_target is not None)
        if have_conv and have_qdot:
            raise ValueError("Entweder konvektiv (U/A_wall[/T_env]) ODER Qdot_target angeben – nicht beides.")

        if have_conv:
            if (U is None) or (A_wall is None):
                raise ValueError("Für konvektive Wärmeführung bitte U und A_wall angeben.")
            if T_env is None:
                T_env = 300.0
            self.mode = "convective"
            self.U = float(U)
            self.Aw = float(A_wall)
            self.Tenv = float(T_env)
            self.Qdot_target = None
        elif have_qdot:
            self.mode = "qdot"
            self.Qdot_target = float(Qdot_target)
            self.U = self.Aw = None
            self.Tenv = 300.0  # Startwert, wird intern gesucht
        else:
            self.mode = "adiabatic"
            self.U = self.Aw = None
            self.Tenv = None
            self.Qdot_target = None

        # Ergebniscontainer
        self._ran = False
        self.T: float = np.nan
        self.P: float = np.nan
        self.tau: float = np.nan
        self._X: Dict[str, float] = {}
        self._g_last = None  # letzter Reactor-Gaszustand

    # -------------------------- Steady State --------------------------

    def run_steady(self) -> None:
        """Stationären Zustand berechnen (advance_to_steady_state)."""
        inlet_g = ct.Solution(self.mechanism, loglevel=self.loglevel)
        inlet_g.TPX = self.T_in, self.P_in, self.composition_in
        upstream = ct.Reservoir(inlet_g, name="upstream")

        # Downstream bei P_set
        out_g = ct.Solution(self.mechanism, loglevel=self.loglevel)
        out_g.TPX = self.T_in, self.P_set, self.composition_in
        downstream = ct.Reservoir(out_g, name="downstream")

        # Reaktor initialisieren (warm start mit Inlet)
        g0 = ct.Solution(self.mechanism, loglevel=self.loglevel)
        g0.TPX = self.T_in, self.P_set, self.composition_in
        r = ct.IdealGasConstPressureReactor(g0, energy="on", name="cstr")
        r.volume = self.V

        # Ein- und Auslass MFC (gleicher mdot → stationär Massenbilanz erfüllt)
        ct.MassFlowController(upstream, r, mdot=self.mdot_in)
        ct.MassFlowController(r, downstream, mdot=self.mdot_in)

        # Wärmeführung
        env = None
        if self.mode == "convective":
            genv = ct.Solution(self.mechanism, loglevel=self.loglevel)
            genv.TPX = self.Tenv, self.P_set, self.composition_in
            env = ct.Reservoir(genv, name="environment")
            ct.Wall(r, env, A=self.Aw, U=self.U)
        elif self.mode == "qdot":
            # Wir implementieren konstantes Qdot, indem wir T_env einer "starken" Wall so wählen,
            # dass Qdot_actual ≈ Qdot_target (Bisection).
            r, env = self._tune_env_for_Qdot(r, upstream)

        net = ct.ReactorNet([r])
        net.rtol, net.atol = self.rtol, self.atol
        net.advance_to_steady_state()

        # Ergebnisse
        self.T = r.T
        self.P = r.thermo.P
        rho = r.thermo.density
        self.tau = self.V * rho / self.mdot_in  # s
        self._X = {s: r.thermo[s].X[0] for s in self.species_to_record if s in r.thermo.species_names}
        self._g_last = r.thermo
        self._ran = True

    # -------------------------- Transient --------------------------

    def run_transient(self, t_end: float, dt_save: float = 1e-3):
        """
        Zeitintegration (z. B. Anfahrvorgang). Gibt (t, T, P, dict(species->array)) zurück.
        """
        inlet_g = ct.Solution(self.mechanism, loglevel=self.loglevel)
        inlet_g.TPX = self.T_in, self.P_in, self.composition_in
        upstream = ct.Reservoir(inlet_g)

        out_g = ct.Solution(self.mechanism, loglevel=self.loglevel)
        out_g.TPX = self.T_in, self.P_set, self.composition_in
        downstream = ct.Reservoir(out_g)

        g0 = ct.Solution(self.mechanism, loglevel=self.loglevel)
        g0.TPX = self.T_in, self.P_set, self.composition_in
        r = ct.IdealGasConstPressureReactor(g0, energy="on")
        r.volume = self.V

        ct.MassFlowController(upstream, r, mdot=self.mdot_in)
        ct.MassFlowController(r, downstream, mdot=self.mdot_in)

        env = None
        if self.mode == "convective":
            genv = ct.Solution(self.mechanism, loglevel=self.loglevel)
            genv.TPX = (self.Tenv if self.Tenv is not None else 300.0), self.P_set, self.composition_in
            env = ct.Reservoir(genv)
            ct.Wall(r, env, A=self.Aw, U=self.U)
        elif self.mode == "qdot":
            # Für transient konstantes Qdot: einfache PI-artige Nachführung von T_env
            env = self._env_for_transient_Qdot(r, upstream, Kp=5.0, UA=1e5)

        net = ct.ReactorNet([r])
        net.rtol, net.atol = self.rtol, self.atol

        ts, Ts, Ps = [], [], []
        species_traces = {s: [] for s in self.species_to_record if s in r.thermo.species_names}

        t = 0.0
        while t < t_end:
            t_next = min(t + dt_save, t_end)
            net.advance(t_next)
            t = t_next
            ts.append(t)
            Ts.append(r.T)
            Ps.append(r.thermo.P)
            for s in species_traces.keys():
                species_traces[s].append(r.thermo[s].X[0])

        # letzte Zustände für Properties:
        self.T = Ts[-1]
        self.P = Ps[-1]
        rho = r.thermo.density
        self.tau = self.V * rho / self.mdot_in
        self._X = {s: species_traces[s][-1] for s in species_traces.keys()}
        self._g_last = r.thermo
        self._ran = True

        return (np.array(ts), np.array(Ts), np.array(Ps),
                {s: np.array(vals) for s, vals in species_traces.items()})

    # -------------------------- Output --------------------------

    def X_out(self) -> Dict[str, float]:
        self._assert_ran()
        return dict(self._X)

    def Y_out(self) -> Dict[str, float]:
        self._assert_ran()
        g = ct.Solution(self.mechanism, loglevel=self.loglevel)
        g.TPX = self.T, self.P, {k: self._X.get(k, 0.0) for k in g.species_names}
        return {sp: float(y) for sp, y in zip(g.species_names, g.Y)}

    def summary(self) -> str:
        self._assert_ran()
        return (f"CSTR steady-state:\n"
                f"  T = {self.T:.2f} K,  P = {self.P:.0f} Pa\n"
                f"  tau = {self.tau:.4f} s,  mdot = {self.mdot_in:.4f} kg/s\n"
                f"  Top species X: " +
                ", ".join(f"{k}={v:.4e}" for k, v in list(self._X.items())[:6]))

    def as_dataframe(self):
        self._assert_ran()
        if pd is None:
            raise RuntimeError("pandas ist nicht installiert (pip install pandas).")
        data = {
            "property": ["T_K", "P_Pa", "tau_s", "mdot_in_kg_s"],
            "value": [self.T, self.P, self.tau, self.mdot_in],
        }
        df = pd.DataFrame(data)
        return df

    # -------------------------- Interna --------------------------

    def _tune_env_for_Qdot(self, r: ct.IdealGasConstPressureReactor, upstream: ct.Reservoir):
        """
        Stellt für steady-state ein Ziel-Qdot ein, indem T_env einer starken Wall per
        Bisektion gesucht wird. Qdot_actual = mdot*(h_out - h_in).
        """
        UA = 1e5  # starke Kopplung, damit T_env direkt wirkt
        T_lo, T_hi = 80.0, 3000.0
        tol_W = max(5.0, 1e-5 * abs(self.Qdot_target))

        def solve_with_Tenv(Tenv: float) -> Tuple[float, ct.Reservoir]:
            genv = ct.Solution(self.mechanism, loglevel=self.loglevel)
            genv.TPX = Tenv, self.P_set, self.composition_in
            env = ct.Reservoir(genv)
            ct.Wall(r, env, A=1.0, U=UA)
            net = ct.ReactorNet([r])
            net.rtol, net.atol = self.rtol, self.atol
            net.advance_to_steady_state()
            h_in = upstream.thermo.enthalpy_mass
            h_out = r.thermo.enthalpy_mass
            Qdot_actual = self.mdot_in * (h_out - h_in)
            return Qdot_actual, env

        # initiale Versuche
        Q_lo, env_lo = solve_with_Tenv(T_lo)
        # Wall-Objekte hängen nun; für neuen Test re-initialisieren wir Reaktorzustand:
        g_reset = ct.Solution(self.mechanism, loglevel=self.loglevel); g_reset.TPX = self.T_in, self.P_set, self.composition_in
        r = ct.IdealGasConstPressureReactor(g_reset, energy="on"); r.volume = self.V
        ct.MassFlowController(upstream, r, mdot=self.mdot_in)
        out_g = ct.Solution(self.mechanism, loglevel=self.loglevel); out_g.TPX = self.T_in, self.P_set, self.composition_in
        downstream = ct.Reservoir(out_g); ct.MassFlowController(r, downstream, mdot=self.mdot_in)

        Q_hi, env_hi = solve_with_Tenv(T_hi)
        # reset again for bisection loop
        g_reset = ct.Solution(self.mechanism, loglevel=self.loglevel); g_reset.TPX = self.T_in, self.P_set, self.composition_in
        r = ct.IdealGasConstPressureReactor(g_reset, energy="on"); r.volume = self.V
        ct.MassFlowController(upstream, r, mdot=self.mdot_in)
        out_g = ct.Solution(self.mechanism, loglevel=self.loglevel); out_g.TPX = self.T_in, self.P_set, self.composition_in
        downstream = ct.Reservoir(out_g); ct.MassFlowController(r, downstream, mdot=self.mdot_in)

        # Prüfen, ob Ziel in [Q_lo, Q_hi]
        if not (min(Q_lo, Q_hi) <= self.Qdot_target <= max(Q_lo, Q_hi)):
            # Clip auf näheren Rand
            return (r, env_lo) if abs(Q_lo - self.Qdot_target) < abs(Q_hi - self.Qdot_target) else (r, env_hi)

        env_best = None
        for _ in range(40):
            T_mid = 0.5 * (T_lo + T_hi)
            # reset Reaktorzustand vor jedem Test
            g_reset = ct.Solution(self.mechanism, loglevel=self.loglevel); g_reset.TPX = self.T_in, self.P_set, self.composition_in
            r = ct.IdealGasConstPressureReactor(g_reset, energy="on"); r.volume = self.V
            ct.MassFlowController(upstream, r, mdot=self.mdot_in)
            out_g = ct.Solution(self.mechanism, loglevel=self.loglevel); out_g.TPX = self.T_in, self.P_set, self.composition_in
            downstream = ct.Reservoir(out_g); ct.MassFlowController(r, downstream, mdot=self.mdot_in)

            Q_mid, env_mid = solve_with_Tenv(T_mid)
            if abs(Q_mid - self.Qdot_target) <= tol_W:
                env_best = env_mid
                break
            if (Q_lo - self.Qdot_target) * (Q_mid - self.Qdot_target) <= 0:
                T_hi, Q_hi, env_hi = T_mid, Q_mid, env_mid
            else:
                T_lo, Q_lo, env_lo = T_mid, Q_mid, env_mid
            env_best = env_mid

        return r, env_best

    def _env_for_transient_Qdot(self, r: ct.IdealGasConstPressureReactor, upstream: ct.Reservoir, Kp: float = 5.0, UA: float = 1e5):
        """
        Erzeugt ein environment-Reservoir, dessen Temperatur wir für transiente
        Läufe in run_transient() *inkrementell* nachführen könnten (einfacher P-Regler).
        Hier nur Erstellung – die eigentliche Regelung müsstest du im Zeitloop implementieren,
        falls wirklich nötig.
        """
        genv = ct.Solution(self.mechanism, loglevel=self.loglevel)
        genv.TPX = 300.0, self.P_set, self.composition_in
        env = ct.Reservoir(genv)
        ct.Wall(r, env, A=1.0, U=UA)
        # (Optional: in deinem Zeitloop könntest du env.thermo.T anpassen, um Qdot näher an Ziel zu bringen.)
        return env

    def _assert_ran(self):
        if not self._ran:
            raise RuntimeError("Reaktor wurde noch nicht berechnet. Rufe run_steady() oder run_transient() zuerst auf.")


"""
# ------------------------ Demo PFR ------------------------
if __name__ == "__main__":
    # Beispiel: stöchiometrische CH4-Luft bei 1 atm, 1000 K
    mech = "gri30.yaml"
    P0 = ct.one_atm
    T0 = 1000.0
    phi = 1.0
    O2_stoich = 2.0
    O2 = O2_stoich / phi
    N2 = 3.76 * O2
    X0 = f"CH4:1, O2:{O2}, N2:{N2}"

    pfr = PlugFlowReactorPSRChain(
        mechanism=mech,
        T0=T0, P0=P0, composition=X0,
        mdot=0.1,       # kg/s
        area=0.01,      # m^2
        length=1.0,     # m
        n_segments=300, # fein genug für glattes Profil
        # Wärmeverlust-Beispiel (deaktiviert/adiabat):
        # U=50.0, A_wall=0.5, T_env=300.0,
        species_to_record=["CH4","O2","CO2","H2O","CO","H2","OH"],
    )
    pfr.run()
    print(f"Outlet T = {pfr.T_profile[-1]:.1f} K @ z={pfr.z_profile[-1]:.3f} m; "
          f"tau = {pfr.tau_profile[-1]*1e3:.2f} ms")

    # Optional CSV
    try:
        pfr.save_csv("pfr_profile.csv")
        print("CSV gespeichert: pfr_profile.csv")
    except Exception as e:
        print(f"CSV nicht gespeichert: {e}")
"""


"""
# --------------------- Demo Mixer ---------------------
if __name__ == "__main__":
    # Zwei Ströme CH4-Luft (unterschiedlich heiß), adiabatisch gemischt
    start_time = timeit.default_timer()
    mech = "gri30.yaml"
    P = ct.one_atm
    m = NonReactiveMixer(mechanism=mech, P_out=P)

    # Strom 1: warm
    phi = 1.0
    O2 = 2.0/phi; N2 = 3.76*O2
    X = f"CH4:1, O2:{O2}, N2:{N2}"
    m.add_stream(mdot=0.08, T=1200.0, P=P, composition=X)

    # Strom 2: kalt Luftüberschuss
    X2 = "O2:1, N2:3.76"
    m.add_stream(mdot=0.02, T=300.0, P=P, composition=X2)

    # adiabatisch
    m.run(Qdot=0.0)
    print(f"Outlet: T={m.T_out:.2f} K, P={m.P_out:.0f} Pa, mdot={m.mdot_out:.3f} kg/s, h={m.h_out:.1f} J/kg")
    print("X_out (Top 6):", dict(list(m.X_out().items())[:6]))
    stop_time = timeit.default_timer()
    print(f"Rechenzeit: {stop_time - start_time:.3f} s")
"""

# ----------------------- Demo CSTR -----------------------
if __name__ == "__main__":
    mech = "gri30.yaml"
    P = ct.one_atm

    # stöchiometrische CH4-Luft am Inlet
    phi = 1.0
    O2 = 2.0/phi
    N2 = 3.76*O2
    X_in = f"CH4:1, O2:{O2}, N2:{N2}"

    # A) Adiabatischer CSTR (steady)
    cstr = CSTR(
        mechanism=mech,
        V=1e-3,            # 1 Liter
        P_set=P,
        mdot_in=0.05,      # kg/s
        T_in=1000.0,       # K
        P_in=P,
        composition_in=X_in,
    )
    cstr.run_steady()
    print(cstr.summary())

    # B) Konvektiv gekühlt
    cstrU = CSTR(
        mechanism=mech, V=1e-3, P_set=P,
        mdot_in=0.05, T_in=1000.0, P_in=P, composition_in=X_in,
        U=50.0, A_wall=0.2, T_env=300.0
    )
    cstrU.run_steady()
    print("Konvektiv:", cstrU.summary())

    # C) Ziel-Qdot = -10 kW (Wärmeverlust)
    cstrQ = CSTR(
        mechanism=mech, V=1e-3, P_set=P,
        mdot_in=0.05, T_in=1000.0, P_in=P, composition_in=X_in,
        Qdot_target=-10_000.0
    )
    cstrQ.run_steady()
    print("Qdot:", cstrQ.summary())