# Reactors.py
from __future__ import annotations
from typing import Callable, Dict, Iterable, List, Optional
import math
import cantera as ct
import pandas as pd


# ========== Versionsrobuste Helper ==========
def safe_clone(gas: ct.Solution) -> ct.Solution:
    """
    Versucht ct.Solution robust zu klonen (für unterschiedliche Cantera-Versionen).
    """
    # 1) Bevorzugt: clone()
    try:
        return gas.clone()  # neuere Versionen
    except Exception:
        pass

    # 2) Aus .source rekonstruieren (falls vorhanden)
    src = getattr(gas, "source", None)
    if src:
        try:
            return ct.Solution(src)
        except Exception:
            pass

    # 3) Aus input_name (+ evtl. Phasenname) rekonstruieren
    inp = getattr(gas, "input_name", None)
    ph = getattr(gas, "name", None)
    if inp:
        try:
            return ct.Solution(inp, ph) if ph else ct.Solution(inp)
        except Exception:
            pass

    raise RuntimeError(
        "safe_clone: Konnte ct.Solution nicht klonen. "
        "Workaround: state_is_copy=False verwenden oder Cantera aktualisieren."
    )


def _set_mfc_mdot(mfc, mdot: float) -> None:
    """Versionssicher Massenstrom für MassFlowController setzen."""
    mdot = float(mdot)
    # Property?
    if hasattr(mfc, "mass_flow_rate"):
        try:
            mfc.mass_flow_rate = mdot
            return
        except Exception:
            pass
    # Setter-Methoden?
    for meth in ("set_mass_flow_rate", "setMassFlowRate"):
        if hasattr(mfc, meth):
            getattr(mfc, meth)(mdot)
            return
    raise AttributeError("MassFlowController: mdot-Setter nicht gefunden.")


def _get_mfc_mdot(mfc) -> float:
    """Versionssicher Massenstrom (Best-Effort) aus MassFlowController lesen."""
    if mfc is None:
        return 0.0
    if hasattr(mfc, "mass_flow_rate"):
        try:
            return float(mfc.mass_flow_rate)
        except Exception:
            pass
    for attr in ("mdot", "m_dot", "mDot"):
        if hasattr(mfc, attr):
            try:
                return float(getattr(mfc, attr))
            except Exception:
                pass
    return float("nan")


def _set_valve_coeff(valve, C: float) -> None:
    """Versionssicher Ventilkoeffizient (ct.Valve) setzen."""
    C = float(C)
    for attr in ("valve_coeff", "K", "coeff"):
        if hasattr(valve, attr):
            try:
                setattr(valve, attr, C)
                return
            except Exception:
                pass
    for meth in ("set_valve_coeff", "setValveCoeff"):
        if hasattr(valve, meth):
            getattr(valve, meth)(C)
            return
    raise AttributeError("Valve: Ventilkoeffizient konnte nicht gesetzt werden.")


def _set_wall_heat_transfer_coeff(wall, UA: float) -> None:
    """Versionssicher den Wärmeübergangskoeffizienten an der Wall setzen (UA bei A=1)."""
    UA = float(UA)
    # Property?
    for attr in ("heat_transfer_coeff", "U", "h"):
        if hasattr(wall, attr):
            try:
                setattr(wall, attr, UA)
                return
            except Exception:
                pass
    # Setter?
    for meth in ("set_heat_transfer_coeff", "setHeatTransferCoeff"):
        if hasattr(wall, meth):
            getattr(wall, meth)(UA)
            return
    raise AttributeError("Wall: heat_transfer_coeff konnte nicht gesetzt werden.")


# ========== Basisklasse ==========
class BaseReactor:
    """
    Universelle Basisklasse für Cantera-Reaktoren.
    - Verwaltet ct.Solution (Gas) und ct.Reactor(+ReactorNet).
    - Sichere X↔Y-Umrechnung und State-Setter (TPX/TPY).
    - Flexibles Sensor-/Logging-System.

    Ableitung:
        - _build_reactor() überschreiben für konkrete Reaktortypen.
        - setup_io() optional für Inlets/Outlets.
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
        self.name = name
        self.gas = safe_clone(gas) if state_is_copy else gas
        self._species = list(self.gas.species_names)

        self.reactor: Optional[ct.ReactorBase] = None
        self.network: Optional[ct.ReactorNet] = None

        # Sensorik
        self._sensors: Dict[str, Callable[[BaseReactor], object]] = {}
        self.history: List[Dict[str, object]] = []

        # Initialzustand
        if T is not None and P is not None:
            if X is not None:
                self.set_state_TPX(T, P, X)
            elif Y is not None:
                self.set_state_TPY(T, P, Y)
            else:
                self.gas.TP = T, P

        # Default-Sensoren
        self._register_default_sensors()

        # Konkreten Reaktor aufsetzen
        self._build_reactor()

    # ---- Hooks ----
    def _build_reactor(self) -> None:
        """Default: IdealGasReactor mit Energie an."""
        self.reactor = ct.IdealGasReactor(self.gas, name=self.name, energy="on")
        self.network = ct.ReactorNet([self.reactor])

    def setup_io(self) -> None:
        """Für Inlet/Outlet in Subklassen überschreiben."""
        pass

    # ---- State API ----
    def set_state_TPX(self, T: float, P: float, X: Dict[str, float]) -> None:
        self.gas.TPX = T, P, self._dict_to_fraction_vector(X, basis="X")

    def set_state_TPY(self, T: float, P: float, Y: Dict[str, float]) -> None:
        self.gas.TPY = T, P, self._dict_to_fraction_vector(Y, basis="Y")

    def set_composition_X(self, X: Dict[str, float]) -> None:
        self.gas.X = self._dict_to_fraction_vector(X, basis="X")

    def set_composition_Y(self, Y: Dict[str, float]) -> None:
        self.gas.Y = self._dict_to_fraction_vector(Y, basis="Y")

    # ---- Integration ----
    def step(self, dt: float) -> float:
        if self.network is None:
            self.network = ct.ReactorNet([self.reactor])
        t0 = self.network.time
        self.network.advance(t0 + dt)
        return self.network.time

    def integrate(self, t_end: float, *, dt_log: Optional[float] = None) -> None:
        if self.network is None:
            self.network = ct.ReactorNet([self.reactor])

        if not dt_log or dt_log <= 0:
            self.network.advance(t_end)
            return

        t = self.network.time
        while t < t_end - 1e-15:
            t_next = min(t + dt_log, t_end)
            self.network.advance(t_next)
            t = self.network.time
            self.log_measurement(t)

    # ---- Sensorik & Logging ----
    def _register_default_sensors(self) -> None:
        self.add_sensor("t", lambda self: (self.network.time if self.network else 0.0))
        self.add_sensor("T", lambda self: self.gas.T)
        self.add_sensor("P", lambda self: self.gas.P)
        self.add_sensor("rho", lambda self: self.gas.density)
        self.add_sensor("mmw", lambda self: self.gas.mean_molecular_weight)
        self.add_sensor("X", lambda self: self.get_X_dict())
        self.add_sensor("Y", lambda self: self.get_Y_dict())

    def add_sensor(self, name: str, func: Callable[["BaseReactor"], object]) -> None:
        self._sensors[name] = func

    def measure(self) -> Dict[str, object]:
        out: Dict[str, object] = {}
        for k, f in self._sensors.items():
            v = f(self)
            if isinstance(v, dict):
                for subk, subv in v.items():
                    out[f"{k}.{subk}"] = subv
            else:
                out[k] = v
        return out

    def log_measurement(self, t: Optional[float] = None) -> Dict[str, object]:
        data = self.measure()
        if t is not None:
            data["t"] = t
        self.history.append(data)
        return data

    def history_as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.history) if self.history else pd.DataFrame()

    # ---- Zusammensetzungen ----
    def get_X_dict(self, species: Optional[Iterable[str]] = None) -> Dict[str, float]:
        if species is None:
            species = self._species
        X = self.gas.X
        return {sp: float(X[self.gas.species_index(sp)]) for sp in species}

    def get_Y_dict(self, species: Optional[Iterable[str]] = None) -> Dict[str, float]:
        if species is None:
            species = self._species
        Y = self.gas.Y
        return {sp: float(Y[self.gas.species_index(sp)]) for sp in species}

    def mole_to_mass(self, X: Dict[str, float]) -> Dict[str, float]:
        Xn = self._normalize_composition(X)
        mmw = sum(Xn[s] * self.gas.molecular_weights[self.gas.species_index(s)] for s in Xn)
        Y = {
            s: (Xn[s] * self.gas.molecular_weights[self.gas.species_index(s)]) / mmw
            for s in Xn
        }
        return self._clean_and_normalize(Y)

    def mass_to_mole(self, Y: Dict[str, float]) -> Dict[str, float]:
        Yn = self._normalize_composition(Y)
        denom = sum(Yn[s] / self.gas.molecular_weights[self.gas.species_index(s)] for s in Yn)
        X = {s: (Yn[s] / self.gas.molecular_weights[self.gas.species_index(s)]) / denom for s in Yn}
        return self._clean_and_normalize(X)

    def mean_mw_from_X(self, X: Dict[str, float]) -> float:
        Xn = self._normalize_composition(X)
        return sum(Xn[s] * self.gas.molecular_weights[self.gas.species_index(s)] for s in Xn)

    def mean_mw_from_Y(self, Y: Dict[str, float]) -> float:
        Yn = self._normalize_composition(Y)
        denom = sum(Yn[s] / self.gas.molecular_weights[self.gas.species_index(s)] for s in Yn)
        return 1.0 / denom

    # ---- Helfer ----
    def _dict_to_fraction_vector(self, comp: Dict[str, float], *, basis: str) -> str:
        comp_n = self._normalize_composition(comp)
        pairs = [f"{sp}:{comp_n.get(sp, 0.0):.16g}" for sp in self._species]
        return ", ".join(pairs)

    @staticmethod
    def _normalize_composition(comp: Dict[str, float]) -> Dict[str, float]:
        clean = {}
        for k, v in comp.items():
            if v is None:
                continue
            vv = float(v)
            if math.isnan(vv):
                continue
            clean[k] = max(vv, 0.0)
        s = sum(clean.values())
        if s <= 0:
            return {}
        return {k: v / s for k, v in clean.items()}

    @staticmethod
    def _clean_and_normalize(comp: Dict[str, float]) -> Dict[str, float]:
        tiny = 1e-16
        comp2 = {k: (0.0 if abs(v) < tiny else float(v)) for k, v in comp.items()}
        return BaseReactor._normalize_composition(comp2)


# ========== PSR (Perfectly Stirred Reactor) ==========
class PSR(BaseReactor):
    """
    Perfectly Stirred Reactor (CSTR/PSR).
    - isobar (ConstPressureReactor) oder isochor (IdealGasReactor)
    - adiabatisch ODER nicht-adiabatisch via Wall: Qdot = UA*(T_env - T)
    - Inlet (Reservoir + MassFlowController), Outlet (Reservoir + Valve)
    - Komfort: set_tau(tau) -> setzt mdot anhand momentaner Dichte
    """

    def __init__(
        self,
        gas: ct.Solution,
        name: str = "PSR",
        *,
        T: Optional[float] = None,
        P: Optional[float] = None,
        X: Optional[Dict[str, float]] = None,
        Y: Optional[Dict[str, float]] = None,
        volume: float = 1.0e-3,
        constant_pressure: bool = True,
        energy_enabled: bool = True,
        UA: float = 0.0,
        T_env: float = 300.0,
        state_is_copy: bool = True,
    ) -> None:
        self.volume = float(volume)
        self.constant_pressure = bool(constant_pressure)
        self.energy_enabled = bool(energy_enabled)
        self.UA = float(UA)
        self.T_env = float(T_env)

        # IO-Objekte
        self._inlet_res: Optional[ct.Reservoir] = None
        self._outlet_res: Optional[ct.Reservoir] = None
        self._mfc_in: Optional[ct.MassFlowController] = None
        self._valve_out: Optional[ct.Valve] = None
        self._C_out: float = 1e-5

        # Umgebung / Wall
        self._env_gas: Optional[ct.Solution] = None
        self._env_res: Optional[ct.Reservoir] = None
        self._wall: Optional[ct.Wall] = None

        super().__init__(gas, name=name, T=T, P=P, X=X, Y=Y, state_is_copy=state_is_copy)

        # Volumen setzen
        self.reactor.volume = self.volume

        # Zusätzliche Sensoren
        self.add_sensor("tau", lambda self: self.current_tau())
        self.add_sensor("mdot_in", lambda self: _get_mfc_mdot(self._mfc_in))
        self.add_sensor("Qdot", lambda self: self.current_Qdot())
        self.add_sensor("T_env", lambda self: self.T_env)

    def _build_reactor(self) -> None:
        """Konkreten Reaktor und optionale Wärmeverluste aufbauen."""
        if self.constant_pressure:
            self.reactor = ct.ConstPressureReactor(
                self.gas, name=self.name, energy=("on" if self.energy_enabled else "off")
            )
        else:
            self.reactor = ct.IdealGasReactor(
                self.gas, name=self.name, energy=("on" if self.energy_enabled else "off")
            )
        self.network = ct.ReactorNet([self.reactor])

        # Nicht-adiabat?
        if self.UA > 0.0:
            self._env_gas = safe_clone(self.gas)
            self._env_gas.TP = self.T_env, self.gas.P
            self._env_res = ct.Reservoir(self._env_gas, name=f"{self.name}_env")

            # A=1 → 'UA' direkt als heat_transfer_coeff setzen
            self._wall = ct.Wall(left=self._env_res, right=self.reactor, A=1.0)
            _set_wall_heat_transfer_coeff(self._wall, self.UA)
            # starre Wand
            try:
                self._wall.expansion_rate_coeff = 0.0
            except Exception:
                pass

    # ---------- Public API ----------
    def set_inlet(
        self,
        *,
        T: float,
        P: float,
        X: Optional[Dict[str, float]] = None,
        Y: Optional[Dict[str, float]] = None,
        mdot: Optional[float] = None,
    ) -> None:
        """Inlet-Reservoir + MassFlowController konfigurieren; mdot optional setzen."""
        inlet_gas = safe_clone(self.gas)
        if X is not None:
            inlet_gas.TPX = T, P, self._dict_to_fraction_vector(X, basis="X")
        elif Y is not None:
            inlet_gas.TPY = T, P, self._dict_to_fraction_vector(Y, basis="Y")
        else:
            inlet_gas.TP = T, P

        # Reservoir anlegen oder Zustand aktualisieren
        if self._inlet_res is None:
            self._inlet_res = ct.Reservoir(inlet_gas, name=f"{self.name}_in")
        else:
            # Bestehendes Reservoir updaten
            if X is not None:
                self._inlet_res.thermo.TPX = T, P, self._dict_to_fraction_vector(X, basis="X")
            elif Y is not None:
                self._inlet_res.thermo.TPY = T, P, self._dict_to_fraction_vector(Y, basis="Y")
            else:
                self._inlet_res.thermo.TP = T, P

        # Massenstromregler
        if self._mfc_in is None:
            self._mfc_in = ct.MassFlowController(self._inlet_res, self.reactor,
                                                 mdot=(mdot if mdot is not None else 0.0))
        elif mdot is not None:
            _set_mfc_mdot(self._mfc_in, mdot)

        # Outlet-Reservoir + Ventil, damit Abfluss/Druckreferenz existiert
        if self._outlet_res is None:
            self._outlet_res = ct.Reservoir(safe_clone(self.gas), name=f"{self.name}_out")
        if self._valve_out is None:
            self._valve_out = ct.Valve(self.reactor, self._outlet_res)
            _set_valve_coeff(self._valve_out, self._C_out)

    def set_outlet_valve_coeff(self, C_out: float) -> None:
        """Ventilkoeffizient des Outlets anpassen (Einfluss auf Abfluss/Druck)."""
        self._C_out = float(C_out)
        if self._valve_out is not None:
            _set_valve_coeff(self._valve_out, self._C_out)

    def set_mdot(self, mdot: float) -> None:
        if self._mfc_in is None:
            raise RuntimeError("Kein Inlet konfiguriert. set_inlet(...) zuerst aufrufen.")
        _set_mfc_mdot(self._mfc_in, mdot)

    def set_tau(self, tau: float) -> None:
        """Setzt mdot so, dass momentan tau ≈ rho*V/mdot gilt (statischer Set)."""
        rho = self.gas.density
        mdot = rho * self.reactor.volume / float(tau)
        self.set_mdot(mdot)

    def set_heat_loss(self, UA: float, T_env: Optional[float] = None) -> None:
        """UA (W/K) und optional T_env (K) anpassen. UA<=0 → adiabatisch."""
        self.UA = float(UA)
        if T_env is not None:
            self.T_env = float(T_env)

        if self.UA <= 0.0:
            if self._wall is not None:
                _set_wall_heat_transfer_coeff(self._wall, 0.0)
            return

        # Umgebung/Wall ggf. anlegen
        if self._env_res is None:
            self._env_gas = safe_clone(self.gas)
            self._env_gas.TP = self.T_env, self.gas.P
            self._env_res = ct.Reservoir(self._env_gas, name=f"{self.name}_env")
        if self._wall is None:
            self._wall = ct.Wall(left=self._env_res, right=self.reactor, A=1.0)
            try:
                self._wall.expansion_rate_coeff = 0.0
            except Exception:
                pass

        _set_wall_heat_transfer_coeff(self._wall, self.UA)
        if self._env_gas is not None:
            self._env_gas.TP = self.T_env, self.gas.P

    # ---------- Sensors / Convenience ----------
    def current_tau(self) -> float:
        mdot = _get_mfc_mdot(self._mfc_in)
        if mdot <= 0 or math.isnan(mdot):
            return float("inf")
        return self.gas.density * self.reactor.volume / mdot

    def current_Qdot(self) -> float:
        if self.UA <= 0.0:
            return 0.0
        # positiv → Wärmefluss IN den Reaktor
        return self.UA * (self.T_env - self.gas.T)


# ========== Minimalbeispiel ==========
if __name__ == "__main__":
    gas = ct.Solution("gri30.yaml")

    # PSR anlegen (nicht-adiabat)
    psr = PSR(
        gas,
        name="psr_demo",
        T=1000.0,
        P=ct.one_atm,
        X={"CH4": 1, "O2": 2, "N2": 7.52},
        volume=2.0e-3,
        constant_pressure=True,
        energy_enabled=True,
        UA=50.0,      # W/K (UA=0 → adiabatisch)
        T_env=800.0,  # K
        state_is_copy=True,
    )

    # Inlet setzen: vorgeheizter Feed und Massenstrom
    psr.set_inlet(
        T=900.0,
        P=ct.one_atm,
        X={"CH4": 1, "O2": 2, "N2": 7.52},
        mdot=0.02,  # kg/s
    )

    # (optional) Tau-Set statt mdot:
    # psr.set_tau(0.15)

    # integrieren und loggen
    psr.integrate(0.05, dt_log=0.001)

    df = psr.history_as_dataframe()
    print(df.filter(["t", "T", "P", "tau", "mdot_in", "Qdot"]).head())
