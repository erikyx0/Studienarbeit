from PFR import PlugFlowReactorPSRChain
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

#%% Test Cantera PFR

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
    mdot=0.1,  # kg/s
    area=0.01,  # m^2
    length=1.0,  # m
    n_segments=30,  # fein genug für glattes Profil
    # Wärmeverlust-Beispiel (deaktiviert/adiabat):
    # U=50.0, A_wall=0.5, T_env=300.0,
    species_to_record=["CH4", "O2", "CO2", "H2O", "CO", "H2", "OH"],
)
pfr.run()
print(f"Outlet T = {pfr.T_profile[-1]:.1f} K @ z={pfr.z_profile[-1]:.3f} m; "
      f"tau = {pfr.tau_profile[-1] * 1e3:.2f} ms")

plt.plot(pfr.z_profile, pfr.T_profile)
plt.grid()
plt.show()

#%% Parameterstudie Temperaturverlust
def run_case(U):
    mech = "gri30.yaml"
    P0 = ct.one_atm; T0 = 2000.0
    phi = 1.0; O2 = 2.0/phi; N2 = 3.76*O2
    X0 = f"CH4:1, O2:{O2}, N2:{N2}"

    pfr = PlugFlowReactorPSRChain(
        mechanism=mech, T0=T0, P0=P0, composition=X0,
        mdot=10, area=0.01, length=0.05,
        n_segments=60,           # etwas feiner als 30 für glatte Kurven
        U=U, A_wall=0.5, T_env=300.0,
        species_to_record=["CH4","O2","CO2","H2O"]
    )
    pfr.run()
    # leichte Rückgabe (Profile sind klein, ok):
    return float(U), pfr.z_profile, pfr.T_profile

if __name__ == "__main__":
    U_values = [0.0, 10.0, 25.0, 50.0, 75.0, 100.0]

    max_workers = max(1, (os.cpu_count() or 2) - 1)
    futures = []
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for U in U_values:
            futures.append(ex.submit(run_case, U))
        for f in as_completed(futures):
            results.append(f.result())

    # plotten
    results.sort(key=lambda t: t[0])
    plt.figure(figsize=(10,6))
    for U, z, T in results:
        plt.plot(z, T, label=f"U={U:g} W/m²K")
    plt.xlabel("Reaktorlänge z [m]"); plt.ylabel("Temperatur T [K]")
    plt.legend(); plt.grid(); plt.tight_layout(); plt.show()