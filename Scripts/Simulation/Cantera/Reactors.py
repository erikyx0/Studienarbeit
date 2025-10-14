# base_reactor.py
from __future__ import annotations
from typing import Callable, Dict, Iterable, List, Optional, Tuple
import cantera as ct
import math
import copy
import pandas as pd


class BaseReactor:
    """
    Universelle Basisklasse für Cantera-Reaktoren.
    - Verwaltet ct.Solution (Gaszustand) und den konkreten ct.Reactor (oder Subklassen).
    - Bietet robuste Umrechnungen zwischen Molen- und Massenanteilen.
    - Enthält ein flexibles Mess-/Sensor-System inkl. History-Logging.

    Ableitung:
        - Überschreibe _build_reactor() für die konkrete Reaktor-Art (z.B. IdealGasReactor).
        - Optional: überschreibe setup_io() um Inlets/Outlets zu setzen.
        - Nutze integrate() / step() für Zeitintegration.

    Messungen:
        - add_sensor("T", lambda self: self.gas.T)
        - measure() -> Dict[str, float]
        - log_measurement(t) fügt self.history einen Snapshot (dict) hinzu.
    """

    def __init__(
        self,
        gas: ct.Solution,
        name: str = "BaseReactor",
        *,
        T: Optional[float] = None,
        P: Optional[float] = None,
        X: Optional[Dict[str, float]] = None,
        Y: Optional[Dict[str, float]] = None,
        state_is_copy: bool = True,
    ) -> None:
        """
        Args:
            gas: Cantera Solution-Objekt (zus. Kinetik/EOS).
            name: Anzeigename.
            T, P: Startzustand (K) und (Pa). Entweder (T,P,X) oder (T,P,Y) setzen.
            X, Y: Startzusammensetzung als dict (Species -> Anteil).
            state_is_copy: True -> intern mit deepcopy(gas) arbeiten (empfohlen).
        """
        self.name = name
        self.gas = copy.deepcopy(gas) if state_is_copy else gas
        self._species = list(self.gas.species_names)

        # Sensoren: name -> callable(self) -> float | dict
        self._sensors: Dict[str, Callable[[BaseReactor], object]] = {}
        self.history: List[Dict[str, object]] = []

        # Reaktorobjekte (Cantera)
        self.reactor: Optional[ct.ReactorBase] = None
        self.network: Optional[ct.ReactorNet] = None

        # initialer Zustand
        if T is not None and P is not None:
            if X is not None:
                self.set_state_TPX(T=T, P=P, X=X)
            elif Y is not None:
                self.set_state_TPY(T=T, P=P, Y=Y)
            else:
                # Nur T,P → Zusammensetzung aus aktuellem Gas übernehmen
                self.gas.TP = T, P

        # Sensoren mit sinnvollen Standardgrößen
        self._register_default_sensors()

        # Konkreten Cantera-Reaktor bauen (in Subklassen überschreiben)
        self._build_reactor()

    # ---------- Hooks für Ableitungen ----------

    def _build_reactor(self) -> None:
        """
        Von Subklassen zu überschreiben: z.B. ct.IdealGasReactor(self.gas, energy='on')
        """
        if isinstance(self.gas, ct.Solution):
            # Default: ein einfacher IdealGasReactor (energieaktiv)
            self.reactor = ct.IdealGasReactor(self.gas, name=self.name, energy='on')
        else:
            raise ValueError("gas muss ein ct.Solution sein.")

        self.network = ct.ReactorNet([self.reactor])

    def setup_io(self) -> None:
        """
        Optional in Subklassen überschreiben, um Reservoirs, Ventile, MFCs etc. zu setzen.
        """
        pass

    # ---------- State-API ----------

    def set_state_TPX(self, T: float, P: float, X: Dict[str, float]) -> None:
        """Setzt Zustand aus T [K], P [Pa], Molenanteilen X (dict)."""
        X_vec = self._dict_to_fraction_vector(X, basis="X")
        self.gas.TPX = T, P, X_vec

    def set_state_TPY(self, T: float, P: float, Y: Dict[str, float]) -> None:
        """Setzt Zustand aus T [K], P [Pa], Massenanteilen Y (dict)."""
        Y_vec = self._dict_to_fraction_vector(Y, basis="Y")
        self.gas.TPY = T, P, Y_vec

    def set_composition_X(self, X: Dict[str, float]) -> None:
        """Nur Zusammensetzung (Molenanteile) ändern, T/P bleiben."""
        X_vec = self._dict_to_fraction_vector(X, basis="X")
        self.gas.X = X_vec

    def set_composition_Y(self, Y: Dict[str, float]) -> None:
        """Nur Zusammensetzung (Massenanteile) ändern, T/P bleiben."""
        Y_vec = self._dict_to_fraction_vector(Y, basis="Y")
        self.gas.Y = Y_vec

    def snapshot_state(self) -> Dict[str, object]:
        """State-Snapshot (praktisch für Logging/Debug)."""
        return {
            "name": self.name,
            "T": self.gas.T,
            "P": self.gas.P,
            "rho": self.gas.density,
            "h": self.gas.enthalpy_mass,
            "u": self.gas.int_energy_mass,
            "cp": self.gas.cp_mass,
            "cv": self.gas.cv_mass,
            "X": self.get_X_dict(),
            "Y": self.get_Y_dict(),
        }

    # ---------- Integration ----------

    def step(self, dt: float) -> float:
        """
        Integrations-Einzelschritt: advance by dt (s). Gibt neue Netzwerkzeit zurück.
        """
        if self.network is None:
            self.network = ct.ReactorNet([self.reactor])
        t0 = self.network.time
        self.network.advance(t0 + dt)
        return self.network.time

    def integrate(self, t_end: float, *, dt_log: Optional[float] = None) -> None:
        """
        Integriere bis t_end (s). Optional: dt_log → Messpunkte in self.history.
        """
        if self.network is None:
            self.network = ct.ReactorNet([self.reactor])

        if dt_log is None or dt_log <= 0:
            self.network.advance(t_end)
            return

        t = self.network.time
        while t < t_end - 1e-15:
            t_next = min(t + dt_log, t_end)
            self.network.advance(t_next)
            t = self.network.time
            self.log_measurement(t)

    # ---------- Mess-/Sensor-System ----------

    def _register_default_sensors(self) -> None:
        """Standard-Sensoren für typische Größen."""
        self.add_sensor("t", lambda self: self.network.time if self.network else 0.0)
        self.add_sensor("T", lambda self: self.gas.T)
        self.add_sensor("P", lambda self: self.gas.P)
        self.add_sensor("rho", lambda self: self.gas.density)
        self.add_sensor("h_mass", lambda self: self.gas.enthalpy_mass)
        self.add_sensor("u_mass", lambda self: self.gas.int_energy_mass)
        self.add_sensor("mmw", lambda self: self.gas.mean_molecular_weight)
        # Zusammensetzungen als verschachtelte Dicts
        self.add_sensor("X", lambda self: self.get_X_dict())
        self.add_sensor("Y", lambda self: self.get_Y_dict())

    def add_sensor(self, name: str, func: Callable[['BaseReactor'], object]) -> None:
        """
        Registriert einen Sensor.
        func: callable(self) -> skalar oder dict (wird in measure() gemergt)
        """
        self._sensors[name] = func

    def measure(self) -> Dict[str, object]:
        """Führt alle Sensoren aus und mapt (verschachtelte) Ergebnisse in ein Dict."""
        out: Dict[str, object] = {}
        for k, f in self._sensors.items():
            val = f(self)
            if isinstance(val, dict):
                # verschachtelt mit Präfix k ablegen
                for subk, subv in val.items():
                    out[f"{k}.{subk}"] = subv
            else:
                out[k] = val
        return out

    def log_measurement(self, t: Optional[float] = None) -> Dict[str, object]:
        """Nimmt Messung auf, ergänzt optional explizite Zeit t."""
        data = self.measure()
        if t is not None:
            data["t"] = t
        self.history.append(data)
        return data

    def history_as_dataframe(self) -> pd.DataFrame:
        """Wandelt History in einen DataFrame (breit) um."""
        if not self.history:
            return pd.DataFrame()
        # flache keys (bereits per measure() organisiert)
        return pd.DataFrame(self.history)

    # ---------- Zusammensetzungen / Umrechnungen ----------

    def get_X_dict(self, species: Optional[Iterable[str]] = None) -> Dict[str, float]:
        """Aktuelle Molenanteile als Dict (nur spezifizierte Species, sonst alle)."""
        if species is None:
            species = self._species
        X = self.gas.X  # ndarray
        return {sp: float(X[self.gas.species_index(sp)]) for sp in species}

    def get_Y_dict(self, species: Optional[Iterable[str]] = None) -> Dict[str, float]:
        """Aktuelle Massenanteile als Dict (nur spezifizierte Species, sonst alle)."""
        if species is None:
            species = self._species
        Y = self.gas.Y
        return {sp: float(Y[self.gas.species_index(sp)]) for sp in species}

    def mole_to_mass(self, X: Dict[str, float]) -> Dict[str, float]:
        """
        Konvertiert Molenanteile → Massenanteile (Dictionary).
        """
        Xn = self._normalize_composition(X)
        # mittlere molare Masse:
        mmw = sum(Xn[sp] * self.gas.molecular_weights[self.gas.species_index(sp)]
                  for sp in Xn)
        Y = {
            sp: (Xn[sp] * self.gas.molecular_weights[self.gas.species_index(sp)]) / mmw
            for sp in Xn
        }
        # numerisch säubern
        return self._clean_and_normalize(Y)

    def mass_to_mole(self, Y: Dict[str, float]) -> Dict[str, float]:
        """
        Konvertiert Massenanteile → Molenanteile (Dictionary).
        """
        Yn = self._normalize_composition(Y)
        denom = sum(
            Yn[sp] / self.gas.molecular_weights[self.gas.species_index(sp)]
            for sp in Yn
        )
        X = {
            sp: (Yn[sp] / self.gas.molecular_weights[self.gas.species_index(sp)]) / denom
            for sp in Yn
        }
        return self._clean_and_normalize(X)

    def mean_mw_from_X(self, X: Dict[str, float]) -> float:
        """Mittlere molare Masse aus Molenanteilen (kg/kmol)."""
        Xn = self._normalize_composition(X)
        return sum(Xn[sp] * self.gas.molecular_weights[self.gas.species_index(sp)]
                   for sp in Xn)

    def mean_mw_from_Y(self, Y: Dict[str, float]) -> float:
        """Mittlere molare Masse aus Massenanteilen (kg/kmol)."""
        Yn = self._normalize_composition(Y)
        denom = sum(Yn[sp] / self.gas.molecular_weights[self.gas.species_index(sp)]
                    for sp in Yn)
        return 1.0 / denom

    # ---------- Helfer ----------

    def _dict_to_fraction_vector(self, comp: Dict[str, float], *, basis: str) -> str:
        """
        Wandelt ein (u. U. unvollständiges) dict in den Cantera-String
        'H2:0.1, O2:0.2, N2:0.7' für gas.X oder gas.Y um.
        - Fehlende Species werden implizit 0 gesetzt.
        - normiert automatisch auf Summe=1.
        """
        comp_n = self._normalize_composition(comp)
        # filtern auf bekannte Species, Warnung bei Unbekannten könnte man ergänzen
        pairs = [f"{sp}:{comp_n.get(sp, 0.0):.16g}" for sp in self._species]
        return ", ".join(pairs)

    @staticmethod
    def _normalize_composition(comp: Dict[str, float]) -> Dict[str, float]:
        """Normiert dict auf Summe=1, entfernt NaN/negativ (setz
