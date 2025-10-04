import cantera as ct
from reactors import *
import matplotlib.pyplot as plt

def ct_X_to_string(X: Dict[str, float]) -> str:
    """Hilfsfunktion: dict -> Cantera-Kompositionsstring (nur positive Anteile)."""
    return ", ".join(f"{k}:{v}" for k, v in X.items() if v > 0.0)

def normalize_X(X: Dict[str, float]) -> Dict[str, float]:
    """Sicherheits-Normalisierung der Molenbrüche."""
    s = sum(max(0.0, v) for v in X.values())
    if s <= 0:
        return {k: 0.0 for k in X.keys()}
    return {k: v/s for k, v in X.items()}

def run_network_demo():
    mech = "gri30.yaml"
    P0 = ct.one_atm

    # ---------- 1) Zwei Inlets mischen ----------
    mixer = NonReactiveMixer(mechanism=mech, P_out=P0)
    # Inlet A (z.B. CH4/Luft, warm)
    O2 = 2.0; N2 = 3.76 * O2
    XA = {"CH4":1.0, "O2":O2, "N2":N2}
    mixer.add_stream(mdot=0.08, T=1200.0, P=P0, composition=ct_X_to_string(XA))
    # Inlet B (Luft-Bypass, kalt)
    XB = {"O2":1.0, "N2":3.76}
    mixer.add_stream(mdot=0.02, T=300.0, P=P0, composition=ct_X_to_string(XB))

    mixer.run(Qdot=0.0)  # adiabatisch
    mix_state = {"T": mixer.T_out, "P": mixer.P_out, "X": mixer.X_out()}
    mix_state["X"] = normalize_X(mix_state["X"])  # robustheit

    # ---------- 2) PFR als PSR-Kette ----------
    pfr = PlugFlowReactorPSRChain(
        mechanism=mech,
        T0=mix_state["T"], P0=mix_state["P"], composition=ct_X_to_string(mix_state["X"]),
        mdot=mixer.mdot_out,
        area=1e-2,
        length=1.0,
        n_segments=200,
        # Beispiel: adiabatisch; falls Kühlung: U=..., A_wall=..., T_env=...
        species_to_record=["CH4","O2","CO2","H2O","CO","H2","OH"],
    )
    pfr.run()
    pfr_out = pfr.outlet_state()  # {T,P,X}

    # ---------- 3) CSTR mit PFR-Outlet als Inlet ----------
    cstr = CSTR(
        mechanism=mech,
        V=1e-3,
        P_set=pfr_out["P"],
        mdot_in=mixer.mdot_out,
        T_in=pfr_out["T"],
        P_in=pfr_out["P"],
        composition_in=ct_X_to_string(pfr_out["X"]),
        # z.B. konvektiv gekühlt:
        U=50.0, A_wall=0.2, T_env=300.0,
        # alternativ: Qdot_target=-10_000.0  (nicht zusammen mit U/A_wall)
    )
    cstr.run_steady()
    print(cstr.summary())

    # Falls du die finalen X willst:
    final_state = cstr.outlet_state()
    print(f"FINAL: T={final_state['T']:.1f} K, P={final_state['P']:.0f} Pa")
    top = list(final_state["X"].items())[:6]
    print("Top species (first 6):", {k: f"{v:.3e}" for k,v in top})

def run_network_pox():
    mech = "gri30.yaml"
    P0 = ct.one_atm

    # ---------- 1) Drei Inlets mischen ----------
    mixer = NonReactiveMixer(mechanism=mech, P_out=P0)
    # Inlet A (Methan)
    XA = {"CH4":1}
    mixer.add_stream(mdot = 182.4 / 3600, T = 66.6 + 273.15, P = P0, composition = ct_X_to_string(XA))
    # Inlet B (Sauerstoff)
    XB = {"O2":1}
    mixer.add_stream(mdot = 252.2 / 3600, T = 231.9 + 273.15, P = P0, composition = ct_X_to_string(XB))
    # Inlet C (Wasserdampf)
    XC = {"H2O":1}
    mixer.add_stream(mdot = 38.7 / 3600, T = 353.4 + 273.15, P = P0, composition = ct_X_to_string(XC))

    mixer.run(Qdot= 0) #adiabatisch
    mix_state = {"T": mixer.T_out, "P": mixer.P_out, "X": mixer.X_out()}
    mix_state["X"] = normalize_X(mix_state["X"])  # robustheit

    # ---------- 2) PFR als PSR-Kette ----------
    pfr = PlugFlowReactorPSRChain(
        mechanism=mech,
        T0=1500, P0=mix_state["P"], composition=ct_X_to_string(mix_state["X"]),
        mdot=mixer.mdot_out,
        area=0.3,
        length=0.4,
        n_segments=100,
        # Beispiel: adiabatisch; falls Kühlung: U=..., A_wall=..., T_env=...
        species_to_record=["CH4", "O2", "CO2", "H2O", "CO", "H2", "OH"],
    )
    pfr.run()
    pfr_out = pfr.outlet_state()  # {T,P,X}
    print(pfr_out)
    plt.plot(pfr.z_profile, pfr.T_profile)
    plt.grid()
    plt.show()

    plt.plot(pfr.z_profile, pfr.X_profile("CO2"), label="CO2")
    plt.plot(pfr.z_profile, pfr.X_profile("CO"), label="CO")
    plt.plot(pfr.z_profile, pfr.X_profile("H2"), label="H2")
    plt.plot(pfr.z_profile, pfr.X_profile("H2O"), label="H2O")
    plt.legend()
    plt.grid()
    plt.show()

    # Falls du die finalen X willst:
    #Test
    final_state = pfr_out
    xs = final_state["X"]  # dict: {"species": mole_fraction}
    top_sorted = sorted(xs.items(), key=lambda kv: kv[1], reverse=True)[:8]
    print("Top species:", {k: f"{v:.3e}" for k, v in top_sorted})

if __name__ == "__main__":
    run_network_pox()