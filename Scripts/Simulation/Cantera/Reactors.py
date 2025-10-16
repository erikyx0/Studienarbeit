# base_reactor.py
from __future__ import annotations
from typing import Callable, Dict, Iterable, List, Optional
import math
import cantera as ct
import pandas as pd


# ========= Versionsrobuste Helper =========
def safe_clone(gas: ct.Solution) -> ct.Solution:
    """Robustes Klonen eines ct.Solution über verschiedene Cantera-Versionen."""
    try:
        return gas.clone()  # neuere Versionen
    except Exception:
        pass
    src = getattr(gas, "source", None)
    if src:
        try:
            return ct.Solution(src)
        except Exception:
            pass
    inp = getattr(gas, "input_name", None)
    ph = getattr(gas, "name", None)
    if inp:
        try:
            return ct.Solution(inp, ph) if ph else ct.Solution(inp)
        except Exception:
            pass
    raise RuntimeError(
        "safe_clone: ct.Solution konnte nicht geklont werden. "
        "Nutze state_is_copy=False oder aktualisiere Cantera."
    )


def _set_wall_heat_transfer_coeff(wall, UA: float) -> None:
    """Versionssicher den Wärmeübergangskoeffizienten der Wall setzen (A=1 → 'UA')."""
    UA = float(UA)
    for attr in ("heat_transfer_coeff", "U", "h"):
        if hasattr(wall, attr):
            try:
                setattr(wall, attr, UA)
                return
            except Exception:
                pass
    for meth in ("set_heat_transfer_coeff", "setHeatTransferCoeff"):
        if hasattr(wall, meth):
            getattr(wall, meth)(UA)
            return
    raise AttributeError("Wall: heat_transfer_coeff konnte nicht gesetzt werden.")


# ========= Basisklasse =========
class BaseReactor:
    """
    Universelle Basisklasse für Cantera-Reaktoren.

    Kernfeatures
    ------------
    - Saubere State-API: set_state_TPX/TPY, set_composition_X/Y, snapshot_state
    - Umrechnungen: mole_to_mass, mass_to_mole, mean_mw_from_X/Y
    - Sensorik & Logging: add_sensor, measure, log_measurement, history_as_dataframe
    - Integration: integrate/step mit vorgelagertem Heat-Update
    - Zentrales Wärmemanagement (für alle abgeleiteten Reaktoren):
        * ADIABATIC:     Q = 0
        * UA:            Q = UA * (T_env - T)
        * Q (fixed):     Q = konst.; intern T_env = T + Q/UA_hint
        * TWALL:         feste Wandtemperatur via großes UA
        * CALLBACK:      Q = f(self) (beliebige Vorgabe), intern über T_env abgebildet

    Ableitung
    ---------
    - Überschreibe _build_reactor(), um den konkreten Reaktor zu erzeugen (z.B. PSR/PFR).
    - setup_io() optional für Inlets/Outlets.
    """

    # ---- Konstruktion ----
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
        energy_enabled: bool = True,  # wird im Default-_build_reactor genutzt
    ) -> None:
        self.name = name
        self.gas = safe_clone(gas) if state_is_copy else gas
        self._species = list(self.gas.species_names)

        # Reaktorobjekte
        self.reactor: Optional[ct.ReactorBase] = None
        self.network: Optional[ct.ReactorNet] = None
        self.energy_enabled = bool(energy_enabled)

        # Sensorik & History
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

        # Wärmemanagement (zentral)
        self._env_gas: Optional[ct.Solution] = None
        self._env_res: Optional[ct.Reservoir] = None
        self._wall: Optional[ct.Wall] = None

        self.heat_mode: str = "ADIABATIC"  # "ADIABATIC","UA","Q","TWALL","CALLBACK"
        self.UA: float = 0.0               # W/K (für UA/TWALL)
        self.T_env: float = 300.0          # K   (UA/TWALL)
        self._Qdot_set: float = 0.0        # W   (für Q)
        self._UA_Qmode: float = 1e6        # W/K (intern, Q/Twall/Callback)
        self._heat_callback: Optional[Callable[[BaseReactor], float]] = None

        # Default-Sensoren registrieren
        self._register_default_sensors()

        # Konkreten Reaktor erstellen (Subklassen überschreiben i.d.R. diese Methode)
        self._build_reactor()

    # ---- Hooks für Ableitungen ----
    def _build_reactor(self) -> None:
        """Default: IdealGasReactor (energy on/off). Subklassen überschreiben dies."""
        self.reactor = ct.IdealGasReactor(
            self.gas, name=self.name, energy=("on" if self.energy_enabled else "off")
        )
        self.network = ct.ReactorNet([self.reactor])

    def setup_io(self) -> None:
        """Optional von Subklassen: Inlets/Outlets/Controller einrichten."""
        pass

    # ---- State-API ----
    def set_state_TPX(self, T: float, P: float, X: Dict[str, float]) -> None:
        self.gas.TPX = T, P, self._dict_to_fraction_vector(X, basis="X")

    def set_state_TPY(self, T: float, P: float, Y: Dict[str, float]) -> None:
        self.gas.TPY = T, P, self._dict_to_fraction_vector(Y, basis="Y")

    def set_composition_X(self, X: Dict[str, float]) -> None:
        self.gas.X = self._dict_to_fraction_vector(X, basis="X")

    def set_composition_Y(self, Y: Dict[str, float]) -> None:
        self.gas.Y = self._dict_to_fraction_vector(Y, basis="Y")

    def snapshot_state(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "T": self.gas.T,
            "P": self.gas.P,
            "rho": self.gas.density,
            "h_mass": self.gas.enthalpy_mass,
            "u_mass": self.gas.int_energy_mass,
            "cp_mass": self.gas.cp_mass,
            "cv_mass": self.gas.cv_mass,
            "mmw": self.gas.mean_molecular_weight,
            "X": self.get_X_dict(),
            "Y": self.get_Y_dict(),
            "Qdot": self._current_Qdot(),
            "T_env": self.T_env,
            "heat_mode": self.heat_mode,
        }

    # ---- Integration ----
    def step(self, dt: float) -> float:
        if self.network is None:
            self.network = ct.ReactorNet([self.reactor])
        t0 = self.network.time
        self._update_heat_before_step()
        self.network.advance(t0 + dt)
        return self.network.time

    def integrate(self, t_end: float, *, dt_log: Optional[float] = None, ctrl_dt: Optional[float] = None) -> None:
        """
        Integriere bis t_end [s]. Vor jedem Schritt werden die Heat-Grenzbedingungen
        (zentrale Wärme-Policy) aktualisiert. dt_log: Mess-Logging-Takt.
        ctrl_dt: interne Kontrollschrittweite (Standard: dt_log oder 1e-3 s).
        """
        if self.network is None:
            self.network = ct.ReactorNet([self.reactor])

        if ctrl_dt is None or ctrl_dt <= 0:
            ctrl_dt = dt_log if (dt_log and dt_log > 0) else 1e-3

        t = self.network.time
        while t < t_end - 1e-15:
            t_next = min(t + ctrl_dt, t_end)
            self._update_heat_before_step()
            self.network.advance(t_next)
            t = self.network.time
            if dt_log and dt_log > 0:
                # Log auf ctrl_dt-Takt ist fein genug; dt_log dient als Anzeige-/Exporttakt
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
        self.add_sensor("Qdot", lambda self: self._current_Qdot())
        self.add_sensor("T_env", lambda self: self.T_env)
        self.add_sensor("heat_mode", lambda self: self.heat_mode)

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
        Y = {s: (Xn[s] * self.gas.molecular_weights[self.gas.species_index(s)]) / mmw for s in Xn}
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

    # ---- Wärme-API (zentral, für alle Reaktortypen) ----
    def set_adiabatic(self) -> None:
        """Adiabater Betrieb: Q=0."""
        self.heat_mode = "ADIABATIC"
        if self._wall is not None:
            _set_wall_heat_transfer_coeff(self._wall, 0.0)

    def set_heat_UA(self, UA: float, T_env: float) -> None:
        """Linearer Wärmeübergang: Q = UA * (T_env - T)."""
        self.heat_mode = "UA"
        self.UA = float(UA)
        self.T_env = float(T_env)
        if self.UA <= 0.0:
            self.set_adiabatic()
            return
        self._ensure_env_wall()
        _set_wall_heat_transfer_coeff(self._wall, self.UA)
        if self._env_gas is not None:
            self._env_gas.TP = self.T_env, self.gas.P

    def set_fixed_heat(self, Qdot_W: float, UA_hint: float = 1e6) -> None:
        """
        Fester Wärmestrom in den Reaktor (W). Vorzeichen:
        +Q heizt den Reaktor, -Q entzieht Wärme (Verlust).
        Intern wird T_env dynamisch so gesetzt, dass Q ≈ UA_hint*(T_env - T) = Qdot_W.
        """
        self.heat_mode = "Q"
        self._Qdot_set = float(Qdot_W)
        self._UA_Qmode = float(UA_hint)
        self._ensure_env_wall()
        _set_wall_heat_transfer_coeff(self._wall, self._UA_Qmode)
        self.T_env = self.gas.T + self._Qdot_set / max(self._UA_Qmode, 1e-12)
        if self._env_gas is not None:
            self._env_gas.TP = self.T_env, self.gas.P

    def set_Twall(self, T_wall_K: float, UA: float = 1e6) -> None:
        """
        Feste Wandtemperatur (≈ Dirichlet-Bedingung). Großes UA koppelt stark an T_wall.
        """
        self.heat_mode = "TWALL"
        self.T_env = float(T_wall_K)
        self.UA = float(UA)
        self._ensure_env_wall()
        _set_wall_heat_transfer_coeff(self._wall, self.UA)
        if self._env_gas is not None:
            self._env_gas.TP = self.T_env, self.gas.P

    def set_heat_callback(self, func: Callable[["BaseReactor"], float], UA_hint: float = 1e6) -> None:
        """
        Benutzerdefinierte Wärmevorgabe: func(self)->Q (W). Vorzeichen wie oben.
        Intern wird T_env so gesetzt, dass Q ≈ UA_hint*(T_env - T).
        """
        self.heat_mode = "CALLBACK"
        self._heat_callback = func
        self._UA_Qmode = float(UA_hint)
        self._ensure_env_wall()
        _set_wall_heat_transfer_coeff(self._wall, self._UA_Qmode)

    # ---- interne Wärme-Helper ----
    def _ensure_env_wall(self) -> None:
        """Stellt Umgebung (Reservoir) + Wall (A=1) bereit."""
        if self._env_res is None:
            g = safe_clone(self.gas)
            g.TP = self.T_env, self.gas.P
            self._env_gas = g
            self._env_res = ct.Reservoir(g, name=f"{self.name}_env")
        if self._wall is None:
            self._wall = ct.Wall(left=self._env_res, right=self.reactor, A=1.0)
            try:
                self._wall.expansion_rate_coeff = 0.0
            except Exception:
                pass

    def _update_heat_before_step(self) -> None:
        """Wird vor jedem Advance aufgerufen, setzt T_env/UA gemäß heat_mode."""
        mode = self.heat_mode
        if mode == "ADIABATIC":
            if self._wall is not None:
                _set_wall_heat_transfer_coeff(self._wall, 0.0)
            return

        if mode == "UA" or mode == "TWALL":
            self._ensure_env_wall()
            _set_wall_heat_transfer_coeff(self._wall, self.UA)
            if self._env_gas is not None:
                self._env_gas.TP = self.T_env, self.gas.P
            return

        if mode == "Q":
            self._ensure_env_wall()
            _set_wall_heat_transfer_coeff(self._wall, self._UA_Qmode)
            self.T_env = self.gas.T + self._Qdot_set / max(self._UA_Qmode, 1e-12)
            if self._env_gas is not None:
                self._env_gas.TP = self.T_env, self.gas.P
            return

        if mode == "CALLBACK":
            self._ensure_env_wall()
            _set_wall_heat_transfer_coeff(self._wall, self._UA_Qmode)
            Q = float(self._heat_callback(self)) if self._heat_callback else 0.0
            self.T_env = self.gas.T + Q / max(self._UA_Qmode, 1e-12)
            if self._env_gas is not None:
                self._env_gas.TP = self.T_env, self.gas.P
            return

    def _current_Qdot(self) -> float:
        """Aktuelle Wärmestrom-Sicht: bei Q/TWALL/CALLBACK das Soll, bei UA der Istwert."""
        if self.heat_mode == "ADIABATIC":
            return 0.0
        if self.heat_mode == "UA":
            return float(self.UA) * (self.T_env - self.gas.T)
        if self.heat_mode == "Q":
            return float(self._Qdot_set)
        if self.heat_mode == "TWALL":
            return float(self.UA) * (self.T_env - self.gas.T)
        if self.heat_mode == "CALLBACK":
            return float(self._heat_callback(self)) if self._heat_callback else 0.0
        return 0.0

    # ---- Utils ----
    def _dict_to_fraction_vector(self, comp: Dict[str, float], *, basis: str) -> str:
        """Dict → 'A:0.1, B:0.2, ...' (vollständige Speciesliste; normiert)."""
        comp_n = self._normalize_composition(comp)
        return ", ".join(f"{sp}:{comp_n.get(sp, 0.0):.16g}" for sp in self._species)

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

# ------- Versionsrobuste Helper für MFC/Valve -------
def _set_mfc_mdot(mfc, mdot: float) -> None:
    mdot = float(mdot)
    if hasattr(mfc, "mass_flow_rate"):
        try:
            mfc.mass_flow_rate = mdot
            return
        except Exception:
            pass
    for meth in ("set_mass_flow_rate", "setMassFlowRate"):
        if hasattr(mfc, meth):
            getattr(mfc, meth)(mdot)
            return
    raise AttributeError("MassFlowController: mdot-Setter nicht gefunden.")

def _get_mfc_mdot(mfc) -> float:
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
# ----------------- PSR -----------------
class PSR(BaseReactor):
    """
    Perfectly Stirred Reactor (CSTR/PSR) auf Cantera-Basis.

    - isobar (ConstPressureReactor) oder isochor (IdealGasReactor)
    - Inlet: Reservoir + MassFlowController (konstanter mdot)
    - Outlet: Reservoir + Valve (Druckreferenz, einfacher Abfluss)
    - Komfort: set_mdot(), set_tau()
    - Sensoren: tau, mdot_in
    - Wärmeführung kommt vollständig aus BaseReactor:
        set_adiabatic(), set_heat_UA(), set_fixed_heat(), set_Twall(), set_heat_callback()
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
        state_is_copy: bool = True,
    ) -> None:
        self.volume = float(volume)
        self.constant_pressure = bool(constant_pressure)

        # IO-Objekte
        self._inlet_res: Optional[ct.Reservoir] = None
        self._outlet_res: Optional[ct.Reservoir] = None
        self._mfc_in: Optional[ct.MassFlowController] = None
        self._valve_out: Optional[ct.Valve] = None
        self._C_out: float = 1e-5  # Default-Ventilkoeffizient

        super().__init__(
            gas, name=name, T=T, P=P, X=X, Y=Y,
            state_is_copy=state_is_copy, energy_enabled=energy_enabled
        )

        # Volumen setzen (nachdem der Reaktor erzeugt wurde)
        self.reactor.volume = self.volume

        # Zusätzliche Sensoren
        self.add_sensor("tau", lambda self: self.current_tau())
        self.add_sensor("mdot_in", lambda self: _get_mfc_mdot(self._mfc_in))

    # ---- Reaktoraufbau ----
    def _build_reactor(self) -> None:
        if self.constant_pressure:
            self.reactor = ct.ConstPressureReactor(
                self.gas, name=self.name,
                energy=("on" if self.energy_enabled else "off")
            )
        else:
            self.reactor = ct.IdealGasReactor(
                self.gas, name=self.name,
                energy=("on" if self.energy_enabled else "off")
            )
        self.network = ct.ReactorNet([self.reactor])

    # ---- IO-Setup ----
    def set_inlet(
        self,
        *,
        T: float,
        P: float,
        X: Optional[Dict[str, float]] = None,
        Y: Optional[Dict[str, float]] = None,
        mdot: Optional[float] = None,
    ) -> None:
        """Inlet-Reservoir + MassFlowController aufsetzen/aktualisieren."""
        inlet_gas = safe_clone(self.gas)
        if X is not None:
            inlet_gas.TPX = T, P, self._dict_to_fraction_vector(X, basis="X")
        elif Y is not None:
            inlet_gas.TPY = T, P, self._dict_to_fraction_vector(Y, basis="Y")
        else:
            inlet_gas.TP = T, P

        if self._inlet_res is None:
            self._inlet_res = ct.Reservoir(inlet_gas, name=f"{self.name}_in")
        else:
            # Zustand des bestehenden Reservoirs aktualisieren
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

        # Outlet-Reservoir + Ventil
        if self._outlet_res is None:
            self._outlet_res = ct.Reservoir(safe_clone(self.gas), name=f"{self.name}_out")
        if self._valve_out is None:
            self._valve_out = ct.Valve(self.reactor, self._outlet_res)
            _set_valve_coeff(self._valve_out, self._C_out)

    # ---- Bedien-API ----
    def set_outlet_valve_coeff(self, C_out: float) -> None:
        """Ventilkoeffizient des Outlets setzen (Einfluss auf Abfluss/Druck)."""
        self._C_out = float(C_out)
        if self._valve_out is not None:
            _set_valve_coeff(self._valve_out, self._C_out)

    def set_mdot(self, mdot: float) -> None:
        """Inlet-Massenstrom setzen [kg/s]."""
        if self._mfc_in is None:
            raise RuntimeError("Kein Inlet konfiguriert. set_inlet(...) zuerst aufrufen.")
        _set_mfc_mdot(self._mfc_in, mdot)

    def set_tau(self, tau: float) -> None:
        """
        mdot so setzen, dass (momentan) tau ≈ rho*V/mdot gilt.
        Hinweis: ρ ändert sich dynamisch → dies ist ein statischer Set auf Basis des aktuellen Zustands.
        """
        rho = self.gas.density
        mdot = rho * self.reactor.volume / float(tau)
        self.set_mdot(mdot)

    # ---- Convenience/Sensoren ----
    def current_tau(self) -> float:
        mdot = _get_mfc_mdot(self._mfc_in)
        if mdot <= 0 or math.isnan(mdot):
            return float("inf")
        return self.gas.density * self.reactor.volume / mdot


# ----------------- CSTR -----------------
class CSTR(BaseReactor):
    """
    Continuous Stirred-Tank Reactor (PSR/CSTR) auf Cantera-Basis.

    Features
    --------
    - isobar (ConstPressureReactor) oder isochor (IdealGasReactor)
    - Inlet: Reservoir + MassFlowController (konstanter mdot)
    - Outlet: Reservoir + Valve (Druckreferenz, einfacher Abfluss)
    - Komfort: set_mdot(), set_tau()
    - Stationär lösen: advance_to_steady_state() + robuster Fallback
    - Sensoren: tau, mdot_in

    Wärmeführung (aus BaseReactor):
      set_adiabatic(), set_heat_UA(UA, T_env), set_fixed_heat(Qdot[, UA_hint]),
      set_Twall(T_wall[, UA]), set_heat_callback(func[, UA_hint])
    """

    def __init__(
        self,
        gas: ct.Solution,
        name: str = "CSTR",
        *,
        T: Optional[float] = None,
        P: Optional[float] = None,
        X: Optional[Dict[str, float]] = None,
        Y: Optional[Dict[str, float]] = None,
        volume: float = 1.0e-3,
        constant_pressure: bool = True,
        energy_enabled: bool = True,
        state_is_copy: bool = True,
    ) -> None:
        self.volume = float(volume)
        self.constant_pressure = bool(constant_pressure)

        # IO-Objekte
        self._inlet_res: Optional[ct.Reservoir] = None
        self._outlet_res: Optional[ct.Reservoir] = None
        self._mfc_in: Optional[ct.MassFlowController] = None
        self._valve_out: Optional[ct.Valve] = None
        self._C_out: float = 1e-5  # Default Ventilkoeffizient

        super().__init__(
            gas, name=name, T=T, P=P, X=X, Y=Y,
            state_is_copy=state_is_copy, energy_enabled=energy_enabled
        )

        # Volumen setzen (nachdem der Reaktor erzeugt wurde)
        self.reactor.volume = self.volume

        # Zusätzliche Sensoren
        self.add_sensor("tau", lambda self: self.current_tau())
        self.add_sensor("mdot_in", lambda self: _get_mfc_mdot(self._mfc_in))

    # ---- Reaktoraufbau ----
    def _build_reactor(self) -> None:
        if self.constant_pressure:
            self.reactor = ct.ConstPressureReactor(
                self.gas, name=self.name,
                energy=("on" if self.energy_enabled else "off")
            )
        else:
            self.reactor = ct.IdealGasReactor(
                self.gas, name=self.name,
                energy=("on" if self.energy_enabled else "off")
            )
        self.network = ct.ReactorNet([self.reactor])

    # ---- IO-Setup ----
    def set_inlet(
        self,
        *,
        T: float,
        P: float,
        X: Optional[Dict[str, float]] = None,
        Y: Optional[Dict[str, float]] = None,
        mdot: Optional[float] = None,
    ) -> None:
        """Inlet-Reservoir + MassFlowController aufsetzen/aktualisieren."""
        inlet_gas = safe_clone(self.gas)
        if X is not None:
            inlet_gas.TPX = T, P, self._dict_to_fraction_vector(X, basis="X")
        elif Y is not None:
            inlet_gas.TPY = T, P, self._dict_to_fraction_vector(Y, basis="Y")
        else:
            inlet_gas.TP = T, P

        if self._inlet_res is None:
            self._inlet_res = ct.Reservoir(inlet_gas, name=f"{self.name}_in")
        else:
            # Zustand des bestehenden Reservoirs aktualisieren
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

        # Outlet-Reservoir + Ventil
        if self._outlet_res is None:
            self._outlet_res = ct.Reservoir(safe_clone(self.gas), name=f"{self.name}_out")
        if self._valve_out is None:
            self._valve_out = ct.Valve(self.reactor, self._outlet_res)
            _set_valve_coeff(self._valve_out, self._C_out)

    def set_outlet_valve_coeff(self, C_out: float) -> None:
        """Ventilkoeffizient des Outlets setzen (Einfluss auf Abfluss/Druck)."""
        self._C_out = float(C_out)
        if self._valve_out is not None:
            _set_valve_coeff(self._valve_out, self._C_out)

    # ---- Bedien-API ----
    def set_mdot(self, mdot: float) -> None:
        """Inlet-Massenstrom setzen [kg/s]."""
        if self._mfc_in is None:
            raise RuntimeError("Kein Inlet konfiguriert. set_inlet(...) zuerst aufrufen.")
        _set_mfc_mdot(self._mfc_in, mdot)

    def set_tau(self, tau: float) -> None:
        """
        mdot so setzen, dass (momentan) tau ≈ rho*V/mdot gilt.
        Hinweis: ρ ändert sich dynamisch → dies ist ein statischer Set auf Basis des aktuellen Zustands.
        """
        rho = self.gas.density
        mdot = rho * self.reactor.volume / float(tau)
        self.set_mdot(mdot)

    # ---- Convenience/Sensoren ----
    def current_tau(self) -> float:
        mdot = _get_mfc_mdot(self._mfc_in)
        if mdot <= 0 or math.isnan(mdot):
            return float("inf")
        return self.gas.density * self.reactor.volume / mdot

    # ---- Stationär lösen ----
    def solve_steady(
        self,
        *,
        max_time: float = 10.0,
        ctrl_dt: float = 1e-3,
        tol_T: float = 1e-8,
        tol_X: float = 1e-10,
        after_step: Optional[Callable[["CSTR"], None]] = None,
    ) -> Dict[str, object]:
        """
        Versucht zuerst `advance_to_steady_state()`. Fällt zurück auf
        kurze Integrationsschritte, bis dT/dt und dX/dt praktisch 0 sind.
        Gibt einen Snapshot (dict) des stationären Zustands zurück.
        """
        # 1) Direkter Versuch (wenn verfügbar)
        try:
            self.network.advance_to_steady_state()
            return self.snapshot_state()
        except Exception:
            pass

        # 2) Fallback: kurze Schritte, Konvergenz prüfen
        if self.network is None:
            self.network = ct.ReactorNet([self.reactor])

        t0 = self.network.time
        last_T = self.gas.T
        last_X = self.gas.X.copy()

        while self.network.time - t0 < max_time:
            t_next = self.network.time + ctrl_dt
            # Wärme-Randbedingungen aus BaseReactor aktualisieren:
            self._update_heat_before_step()
            self.network.advance(t_next)

            # Optionale Hook
            if after_step is not None:
                after_step(self)

            dT = abs(self.gas.T - last_T)
            dX = float(abs(self.gas.X - last_X).max())

            last_T = self.gas.T
            last_X = self.gas.X.copy()

            if dT < tol_T and dX < tol_X:
                break

        return self.snapshot_state()


# ---------- Minimalbeispiel ----------
if __name__ == "__main__":
    gas = ct.Solution("gri30.yaml")
    r = CSTR(
        gas,
        name="cstr_demo",
        T=1000.0, P=ct.one_atm,
        X={"CH4": 1, "O2": 2, "N2": 7.52},
        volume=2e-3,
        constant_pressure=True,
        energy_enabled=True,
    )

    # Inlet & Betriebsweise
    r.set_inlet(T=900.0, P=ct.one_atm, X={"CH4": 1, "O2": 2, "N2": 7.52}, mdot=0.02)
    # Wärmeführung (kommt aus BaseReactor)
    # r.set_adiabatic()
    r.set_heat_UA(UA=80.0, T_env=800.0)
    # r.set_fixed_heat(Qdot_W=-5000.0)   # fester Wärmeverlust 5 kW

    # Stationär lösen
    steady = r.solve_steady(max_time=2.0, ctrl_dt=5e-4)
    print({k: steady[k] for k in ("T", "P", "Qdot", "heat_mode")})