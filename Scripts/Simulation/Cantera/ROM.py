import time
import cantera as ct

def kgph_to_kgs(x): return x / 3600.0
def C_to_K(tC): return tC + 273.15

def make_stream_reservoir(gas_mech, T, P, X, mdot):
    gas = ct.Solution(gas_mech)
    gas.TPX = T, P, X
    res = ct.Reservoir(gas)
    return res, mdot

def run_psr_with_Qdot(
    mechanism, inlets, P_in, V_reactor_m3, Q_wall_kW=0.0,
    T_env=300.0, Kpc=1e-5, t_final_factor=15.0
):
    """
    PSR (IdealGasConstPressureReactor) mit konstanter Wärmeabfuhr Q [kW].
    Realisiert über Wall mit zeitvariabler (h*A), sodass Q ≈ Q_target.
    """
    # Umgebung als Reservoir
    env = ct.Reservoir(ct.Solution(mechanism))
    # Startgas im Reaktor (leer, N2 als Dummy)
    gas0 = ct.Solution(mechanism)
    gas0.TPX = T_env, P_in, "N2:1.0"
    r = ct.IdealGasConstPressureReactor(gas0, volume=V_reactor_m3, energy='on')
    # Inlets anbinden und Gesamt-Massenstrom bestimmen
    m_in = 0.0
    for res, mdot in inlets:
        ct.MassFlowController(res, r, mdot=mdot)
        m_in += mdot
    # Dummy-Auslass-Reservoir für Master-MFC
    dummy_outlet = ct.Reservoir(ct.Solution(mechanism))
    # Master-MassFlowController vom Reaktor zum Dummy-Auslass
    mfc_master = ct.MassFlowController(r, dummy_outlet, mdot=m_in)
    # PressureController mit Master-MFC (als 3. Positionsargument)
    pctrl = ct.PressureController(r, env, mfc_master, K=Kpc)
    # Wall für Wärmeabfuhr (h wird dynamisch geregelt)
    wall = ct.Wall(r, env)
    wall.area = 1.0
    wall.expansion_rate_coeff = 0.0
    wall.heat_transfer_coeff = 0.0
    net = ct.ReactorNet([r])
    Q_target = Q_wall_kW * 1000.0  # W
    t = 0.0
    # Abschätzung der Aufenthaltszeit (tau)



def run_pfr_series(
    mechanism, inlet_state, P_in, mdot, tau_total, n_seg=200, Kpc=1e-5
):
    T_seg, X_seg = inlet_state['T'], inlet_state['X'].copy()
    gas_names = inlet_state['names']

    for _ in range(n_seg):
        gas_k = ct.Solution(mechanism)
        Xdict = {sp: float(x) for sp, x in zip(gas_names, X_seg)}
        gas_k.TPX = T_seg, P_in, Xdict

        up = ct.Reservoir(gas_k)
        env = ct.Reservoir(ct.Solution(mechanism))
        rho = gas_k.density
        tau_seg = tau_total / float(n_seg)
        V_k = mdot * tau_seg / rho

        r = ct.IdealGasConstPressureReactor(gas_k, volume=V_k, energy='on')
        mfc = ct.MassFlowController(up, r, mdot=mdot)
        pc  = ct.PressureController(r, env, master=mfc, K=Kpc)
        net = ct.ReactorNet([r])

        try:
            net.advance_to_steady_state()
        except Exception:
            t = 0.0
            while t < 10.0 * tau_seg:
                t = net.advance(t + 0.2 * tau_seg)

        T_seg = r.T
        X_seg = r.thermo.X

    return T_seg, X_seg

def benchmark_psr_pfr_case(mechanism, case, P_in, n_pfr=200):
    X_natgas = case.get('X_natgas', "CH4:0.96, C2H6:0.03, N2:0.01")
    X_o2     = "O2:1.0"
    X_co2    = "CO2:1.0"
    X_steam  = "H2O:1.0"

    inlets = []
    if case['NG_mdot_kgph'] > 0:
        res, mdot = make_stream_reservoir(
            mechanism, C_to_K(case['NG_T_C']), P_in, X_natgas,
            kgph_to_kgs(case['NG_mdot_kgph'])
        )
        inlets.append((res, mdot))
    if case['CO2_mdot_kgph'] > 0:
        res, mdot = make_stream_reservoir(
            mechanism, C_to_K(case['CO2_T_C']), P_in, X_co2,
            kgph_to_kgs(case['CO2_mdot_kgph'])
        )
        inlets.append((res, mdot))
    if case['O2_mdot_kgph'] > 0:
        res, mdot = make_stream_reservoir(
            mechanism, C_to_K(case['O2_T_C']), P_in, X_o2,
            kgph_to_kgs(case['O2_mdot_kgph'])
        )
        inlets.append((res, mdot))
    if case['H2O_mdot_kgph'] > 0:
        res, mdot = make_stream_reservoir(
            mechanism, C_to_K(case['H2O_T_C']), P_in, X_steam,
            kgph_to_kgs(case['H2O_mdot_kgph'])
        )
        inlets.append((res, mdot))

    V_m3 = case['V_L'] / 1000.0
    r_psr, _ = run_psr_with_Qdot(
        mechanism, inlets, P_in, V_m3, Q_wall_kW=case['Q_kW']
    )

    gas_psr = r_psr.thermo
    T_psr = r_psr.T
    X_psr = gas_psr.X.copy()
    names = gas_psr.species_names[:]

    mdot_total = sum(m for _, m in inlets)
    rho_psr = gas_psr.density
    Vdot_m3_s = mdot_total / rho_psr
    tau_psr = V_m3 / Vdot_m3_s

    tau_pfr = case.get('tau_pfr_s', 2.0 * tau_psr)
    T_pfr, X_pfr = run_pfr_series(
        mechanism,
        inlet_state={'T': T_psr, 'X': X_psr, 'names': names},
        P_in=P_in, mdot=mdot_total, tau_total=tau_pfr, n_seg=n_pfr
    )

    t1 = time.perf_counter()

    idx = {s: i for i, s in enumerate(names)}
    def molfrac(name, X): return float(X[idx[name]]) if name in idx else float('nan')

    return {
        'mechanism': mechanism,
        'runtime_s': t1 - benchmark_psr_pfr_case.t0,
        'psr_tau_s': float(tau_psr),
        'pfr_tau_s': float(tau_pfr),
        'psr_T_K': float(T_psr),
        'pfr_T_K': float(T_pfr),
        'x_H2': molfrac('H2', X_pfr),
        'x_CO': molfrac('CO', X_pfr),
        'x_CO2': molfrac('CO2', X_pfr),
        'x_H2O': molfrac('H2O', X_pfr),
    }

def benchmark_mech(mechanism, P_bar=30.0, n_pfr=200):
    P_in = P_bar * ct.one_atm  # Pa
    benchmark_psr_pfr_case.t0 = time.perf_counter()
    res1 = benchmark_psr_pfr_case(mechanism, case1, P_in=P_in, n_pfr=n_pfr)
    res2 = benchmark_psr_pfr_case(mechanism, case2, P_in=P_in, n_pfr=n_pfr)

    def line(r, tag):
        return (f"{tag} | runtime={r['runtime_s']:.3f}s | "
                f"T_out={r['pfr_T_K']:.1f}K | "
                f"x(H2)={r['x_H2']:.3f} x(CO)={r['x_CO']:.3f} "
                f"x(CO2)={r['x_CO2']:.3f} x(H2O)={r['x_H2O']:.3f} | "
                f"τ_PSR={r['psr_tau_s']:.4f}s τ_PFR={r['pfr_tau_s']:.4f}s")

    print(f"Mechanism: {mechanism}")
    print(line(res1, "Case 1"))
    print(line(res2, "Case 2"))
    return {'case1': res1, 'case2': res2}

case1 = dict(
    NG_T_C=66.6,   NG_mdot_kgph=182.4,
    CO2_T_C=25.0,  CO2_mdot_kgph=0.0,
    O2_T_C=231.9,  O2_mdot_kgph=252.2,
    H2O_T_C=353.4, H2O_mdot_kgph=38.7,
    Q_kW=30.0,
    V_L=134.0,
)

case2 = dict(
    NG_T_C=67.5,   NG_mdot_kgph=153.3,
    CO2_T_C=67.5,  CO2_mdot_kgph=196.4,
    O2_T_C=231.2,  O2_mdot_kgph=246.9,
    H2O_T_C=353.4, H2O_mdot_kgph=39.4,
    Q_kW=30.0,
    V_L=134.0,
)

if __name__ == "__main__":
    benchmark_mech("gri30.yaml")
