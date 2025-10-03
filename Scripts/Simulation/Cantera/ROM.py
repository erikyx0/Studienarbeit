# -*- coding: utf-8 -*-
# PSR -> PFR (FlowReactor) Benchmark für zwei Cases
# Kompatibel mit Cantera 3.x; PressureController ist versionsrobust aufgebaut.
# Ausgaben: CSV je Case + Konsolen-Zusammenfassung (inkl. runtime)

import time
import csv
import sys
import math
import cantera as ct

# -------- Hilfsfunktionen --------
def kgph_to_kgs(x): return x/3600.0
def C_to_K(tC): return tC + 273.15

def make_stream_reservoir(mechanism, T, P, X, mdot):
    gas = ct.Solution(mechanism); gas.TPX = T, P, X
    return ct.Reservoir(gas), mdot

# -------- PSR (const-p) mit optionaler Wärmeabfuhr --------
def run_psr_with_Qdot(
    mechanism, inlets, P_in, V_reactor_m3,
    Q_wall_kW=0.0, T_env=300.0, Kpc=1e-5, t_final_factor=15.0
):
    # Umgebung (Massenauslass + Wärmesenke)
    gas_env = ct.Solution(mechanism); gas_env.TPX = T_env, P_in, "N2:1.0"
    env = ct.Reservoir(gas_env)

    # Startgas im PSR
    gas0 = ct.Solution(mechanism); gas0.TPX = T_env, P_in, "N2:1.0"
    r = ct.IdealGasConstPressureReactor(gas0, volume=V_reactor_m3, energy='on')

    # Inlets anbinden + mdot_total sammeln
    mdot_total = 0.0
    for res, mdot in inlets:
        ct.MassFlowController(res, r, mdot=mdot)
        mdot_total += mdot
    if mdot_total <= 0:
        raise ValueError("PSR: keine/negative Inlet-Massenströme.")

    # **WICHTIG: Outlet als MassFlowController, NICHT PressureController**
    # Gleicher Massenstrom wie Inlet-Summe => stationäre Masse im Reaktor.
    ct.MassFlowController(r, env, mdot=mdot_total)

    # Wärmeabfuhr via Wall (Q ≈ Q_wall_kW)
    wall = ct.Wall(r, env)
    wall.area = 1.0
    wall.expansion_rate_coeff = 0.0
    wall.heat_transfer_coeff = 0.0  # im Integrationsloop geregelt

    net = ct.ReactorNet([r])

    # Aufenthaltszeit grob schätzen für Integrationshorizont
    rho_guess = max(1e-6, r.thermo.density)
    tau_guess = max(1e-4, rho_guess * V_reactor_m3 / mdot_total)
    t, t_end = 0.0, 15.0 * tau_guess

    Q_target = Q_wall_kW * 1000.0
    regulate_heat = abs(Q_target) > 0.0

    while t < t_end:
        if regulate_heat:
            dT = max(1e-6, abs(r.T - T_env))
            wall.heat_transfer_coeff = abs(Q_target) / dT
        t = net.advance(t + 0.2 * tau_guess)

    # PSR-RT berechnen (aus aktuellem Zustand)
    rho_psr = r.thermo.density
    Vdot = mdot_total / rho_psr
    tau_psr = V_reactor_m3 / max(1e-12, Vdot)

    return r, tau_psr, mdot_total

# -------- PFR als FlowReactor (homogen, adiabatisch) --------
def run_pfr_flowreactor_to_csv(
    mechanism, inlet_gas_state, P_in, mdot, tau_pfr,
    A_m2=1.0e-2, porosity=1.0, output_filename="pfr_output.csv",
    log_every=200
):
    """
    Homogener FlowReactor (keine Oberfläche). Länge so gewählt, dass tau = L / u.
    u = mdot / (rho * A * ε). rho wird fortlaufend aktualisiert, daher nutzen wir
    den Inletzustand zur Geometrieableitung und integrieren bis L.
    """
    # Inlet-Gas vorbereiten
    gas = ct.Solution(mechanism)
    gas.TPX = inlet_gas_state['T'], P_in, inlet_gas_state['Xdict']

    # Geometrie/Strömung am Inlet
    rho_in = gas.density
    u_in = mdot / (rho_in * A_m2 * porosity)
    L = max(1e-6, u_in * tau_pfr)  # Ziel-Länge

    # FlowReactor (homogen)
    r = ct.FlowReactor(gas)
    r.area = A_m2
    r.surface_area_to_volume_ratio = 0.0  # keine Oberfläche
    r.mass_flow_rate = mdot
    r.energy_enabled = True  # adiabatisch (keine Wand -> keine Q̇-Schnittstelle)

    sim = ct.ReactorNet([r])

    # Output-Header
    names = gas.species_names
    data = []
    n = 0

    # Startlog
    # print(f"PFR start: x=0 mm, T={r.T:.2f} K, p={r.thermo.P/1e5:.2f} bar")

    while sim.distance < L:
        sim.step()  # integrator macht kleine Schritte nach Bedarf
        if (n % log_every) == 0:
            # print(f"x={sim.distance*1e3:8.3f} mm  T={r.T:8.2f} K")
            pass
        n += 1

        data.append(
            [sim.distance*1e3, r.T-273.15, r.thermo.P/ct.one_atm] + list(r.thermo.X)
        )

    # CSV schreiben
    with open(output_filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(['Distance (mm)', 'T (C)', 'P (atm)'] + names)
        w.writerows(data)

    return {
        'L_m': float(L),
        'u_in_m_s': float(u_in),
        'A_m2': float(A_m2),
        'porosity': float(porosity),
        'rows': len(data),
        'outfile': output_filename
    }

# -------- Ein Case komplett rechnen --------
def run_case(mechanism, case, P_bar=30.0, A_m2=1.0e-2, porosity=1.0,
             pfr_tau_override=None, csv_basename="case"):
    """
    Rechnet PSR (mit Q̇) -> PFR (FlowReactor) für ein Case.
    Gibt Dict mit runtime, PSR/PFR-Kernwerten & Dateinamen zurück.
    """
    P_in = P_bar * 1e5
    t0 = time.perf_counter()

    # Zusammensetzungen
    X_natgas = case.get('X_natgas', "CH4:0.96, C2H6:0.03, N2:0.01")
    X_o2, X_co2, X_steam = "O2:1.0", "CO2:1.0", "H2O:1.0"

    # Inlets
    inlets = []
    if case['NG_mdot_kgph']  > 0: inlets.append(make_stream_reservoir(mechanism, C_to_K(case['NG_T_C']),  P_in, X_natgas, kgph_to_kgs(case['NG_mdot_kgph'])))
    if case['CO2_mdot_kgph'] > 0: inlets.append(make_stream_reservoir(mechanism, C_to_K(case['CO2_T_C']), P_in, X_co2,   kgph_to_kgs(case['CO2_mdot_kgph'])))
    if case['O2_mdot_kgph']  > 0: inlets.append(make_stream_reservoir(mechanism, C_to_K(case['O2_T_C']),  P_in, X_o2,    kgph_to_kgs(case['O2_mdot_kgph'])))
    if case['H2O_mdot_kgph'] > 0: inlets.append(make_stream_reservoir(mechanism, C_to_K(case['H2O_T_C']), P_in, X_steam, kgph_to_kgs(case['H2O_mdot_kgph'])))

    # PSR
    V_m3 = case['V_L'] / 1000.0
    r_psr, tau_psr, mdot_total = run_psr_with_Qdot(
        mechanism, inlets, P_in, V_m3, Q_wall_kW=case['Q_kW']
    )

    thermo = r_psr.thermo
    names = thermo.species_names
    Xdict_psr = {sp: float(x) for sp, x in zip(names, thermo.X)}
    T_psr = r_psr.T

    # PFR-Verweilzeit
    tau_pfr = pfr_tau_override if pfr_tau_override is not None else 2.0 * tau_psr

    # PFR (FlowReactor, homogen)
    csv_name = f"{csv_basename}.csv"
    pfr_meta = run_pfr_flowreactor_to_csv(
        mechanism,
        inlet_gas_state={'T': T_psr, 'Xdict': Xdict_psr},
        P_in=P_in, mdot=mdot_total, tau_pfr=tau_pfr,
        A_m2=A_m2, porosity=porosity, output_filename=csv_name
    )

    # Endzustand (letzter Eintrag) schnell über erneutes Setzen berechnen:
    gas_out = ct.Solution(mechanism); gas_out.TPX = T_psr, P_in, Xdict_psr
    # Kein Recompute nötig; wir nehmen den letzten Zustand aus CSV wäre teuer. Stattdessen:
    # Einfacher: FlowReactor-Endzustand erneut berechnen ist Overhead; wir lassen es.
    # Für Kennzahlen lieber PFR nochmal kurz laufen? -> sparen wir uns hier.
    # Wir geben PSR-Out plus Geometrie und Datei zurück.

    t1 = time.perf_counter()
    return {
        'runtime_s': t1 - t0,
        'psr_tau_s': float(tau_psr),
        'pfr_tau_s': float(tau_pfr),
        'psr_T_K': float(T_psr),
        'mdot_total_kg_s': float(mdot_total),
        'pfr': pfr_meta,
        'csv': csv_name
    }

# -------- Deine zwei Cases (wie besprochen) --------
CASE1 = dict(
    NG_T_C=66.6,   NG_mdot_kgph=182.4,
    CO2_T_C=25.0,  CO2_mdot_kgph=0.0,
    O2_T_C=231.9,  O2_mdot_kgph=252.2,
    H2O_T_C=353.4, H2O_mdot_kgph=38.7,
    Q_kW=30.0,
    V_L=134.0
)
CASE2 = dict(
    NG_T_C=67.5,   NG_mdot_kgph=153.3,
    CO2_T_C=67.5,  CO2_mdot_kgph=196.4,
    O2_T_C=231.2,  O2_mdot_kgph=246.9,
    H2O_T_C=353.4, H2O_mdot_kgph=39.4,
    Q_kW=30.0,
    V_L=134.0
)

# -------- Öffentliche Funktion --------
def f(mechanism_path: str, P_bar: float = 30.0, A_m2: float = 1.0e-2, porosity: float = 1.0,
      n_case: int | None = None, tau_pfr_case1: float | None = None, tau_pfr_case2: float | None = None):
    """
    Rechnet Case 1 & 2 (oder nur einen Case mit n_case=1/2).
    Gibt Dict mit Ergebnissen zurück und schreibt je Case eine CSV.
    """
    results = {'mechanism': mechanism_path}
    if n_case in (None, 1):
        res1 = run_case(mechanism_path, CASE1, P_bar=P_bar, A_m2=A_m2, porosity=porosity,
                        pfr_tau_override=tau_pfr_case1, csv_basename="case1_flowreactor")
        results['case1'] = res1
        #print(f"[Case 1] runtime={res1['runtime_s']:.3f}s | τ_PSR={res1['psr_tau_s']:.4f}s | τ_PFR={res1['pfr_tau_s']:.4f}s | "
              #f"A={A_m2} m², ε={porosity}, L={res1['pfr']['L_m']:.3f} m | CSV={res1['csv']}")
    if n_case in (None, 2):
        res2 = run_case(mechanism_path, CASE2, P_bar=P_bar, A_m2=A_m2, porosity=porosity,
                        pfr_tau_override=tau_pfr_case2, csv_basename="case2_flowreactor")
        results['case2'] = res2
        #print(f"[Case 2] runtime={res2['runtime_s']:.3f}s | τ_PSR={res2['psr_tau_s']:.4f}s | τ_PFR={res2['pfr_tau_s']:.4f}s | "
              #f"A={A_m2} m², ε={porosity}, L={res2['pfr']['L_m']:.3f} m | CSV={res2['csv']}")
    return results

# -------- CLI --------
if __name__ == "__main__":
    mech = sys.argv[1] if len(sys.argv) >= 2 else "gri30.yaml"
    f(mech)
