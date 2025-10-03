# pfr_psrchain.py
# Plug-Flow-Reaktor als Kette von PSRs (IdealGasConstPressureReactor) – Cantera 3.x
# Lizenz: MIT

from __future__ import annotations
from typing import List, Optional, Union, Dict, Sequence
import numpy as np
import cantera as ct

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
        """Kette aufbauen, auf stationär lösen, Profile einsammeln."""
        # Inlet-Zustand
        gas_in = ct.Solution(self.mechanism, loglevel=self.loglevel)
        gas_in.TPX = self.T0, self.P0, self.composition

        upstream = ct.Reservoir(gas_in, name="inlet")

        # Downstream-Reservoir (zur Druckankopplung)
        gas_out = ct.Solution(self.mechanism, loglevel=self.loglevel)
        gas_out.TPX = self.T0, self.P0, self.composition
        downstream = ct.Reservoir(gas_out, name="outlet")

        # Startwerte (z=0) loggen
        self._T[0] = gas_in.T
        self._P[0] = gas_in.P
        for s in self._X.keys():
            if s in gas_in.species_names:
                self._X[s][0] = gas_in[s].X[0]

        reactors: List[ct.IdealGasConstPressureReactor] = []
        mfc_in: List[ct.MassFlowController] = []
        walls = []

        prev_reactor = None
        for i in range(self.n):
            # Segment i – eigener Gaszustand (Kopieren des aktuellen Inlet-Zustands)
            gas_i = ct.Solution(self.mechanism, loglevel=self.loglevel)
            gas_i.TPX = gas_in.T, gas_in.P, gas_in.X

            r = ct.IdealGasConstPressureReactor(gas_i, energy="on", name=f"psr_{i}")
            r.volume = self.area * self.dz[i]
            reactors.append(r)

            # Einlass: entweder Inlet-Reservoir oder vorheriger PSR
            src = upstream if prev_reactor is None else prev_reactor
            mfc_in.append(ct.MassFlowController(src, r, mdot=self.mdot))

            # Optionale Wärmeverluste
            if self.U is not None:
                env_gas = ct.Solution(self.mechanism, loglevel=self.loglevel)
                env_gas.TPX = self.T_env, self.P0, self.composition
                env = ct.Reservoir(env_gas, name=f"env_{i}")
                walls.append(ct.Wall(r, env, A=float(self.Aw[i]), U=float(self.U[i])))

            prev_reactor = r

            # Update "gas_in" für das nächste Segment: zu Beginn identisch,
            # nach Lösung des Netzes wird unten der reale stationäre Zustand gelesen.

        # Auslass vom letzten Reaktor
        ct.MassFlowController(reactors[-1], downstream, mdot=self.mdot)

        # Netz lösen (stationär)
        net = ct.ReactorNet(reactors)
        # Optional: Toleranzen
        # net.rtol = 1e-9
        # net.atol = 1e-15
        net.advance_to_steady_state()

        # Profile einsammeln + Verweilzeiten berechnen
        tau_cum = 0.0
        for i, r in enumerate(reactors, start=1):
            self._T[i] = r.T
            self._P[i] = r.thermo.P

            rho_i = r.thermo.density
            u_i = self.mdot / (rho_i * self.area)  # m/s
            if u_i <= 0.0:
                raise RuntimeError("Nichtpositive Geschwindigkeit u_i – prüfe mdot/area/State.")
            dt_i = self.dz[i - 1] / u_i
            tau_cum += dt_i
            self._tau[i] = tau_cum

            for s in self._X.keys():
                if s in r.thermo.species_names:
                    self._X[s][i] = r.thermo[s].X[0]

        # Finalisieren
        self._reactors = reactors
        self._net = net

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


# ------------------------ Demo ------------------------
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
