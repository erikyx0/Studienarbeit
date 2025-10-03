# %% Einfache CH4-Luft-Verbrennung (adiabat, p=const) mit Cantera 3.x
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

# ---- 1) Randbedingungen ----
mechanism = "gri30.yaml"   # oder z.B. "gri30.yaml", "hychem.yaml" (Kerosin), etc.
P0 = ct.one_atm              # Startdruck [Pa]
T0 = 1000.0                  # Starttemperatur [K] (für Zündung realistisch 900–1300 K)
phi = 1.0                    # Äquivalenzverhältnis (1.0 = stöchiometrisch)

# Luft als O2 + 3.76 N2; stöchiometrisch: CH4 + 2 O2 -> ...
O2_per_CH4_stoich = 2.0
O2 = O2_per_CH4_stoich / phi
N2 = 3.76 * O2
X_str = f"CH4:1, O2:{O2}, N2:{N2}"

# ---- 2) Gaszustand & adiabate Gleichgewichtstemp (Referenz) ----
gas = ct.Solution(mechanism)
gas.TPX = T0, P0, X_str

# Kopie für Gleichgewicht
gas_eq = ct.Solution(mechanism)
gas_eq.TPX = T0, P0, X_str
gas_eq.equilibrate("HP")  # adiabates Gleichgewicht
T_ad = gas_eq.T
print(f"Adiabate Gleichgewichtstemp (HP): {T_ad:.1f} K")

# ---- 3) Reaktor: druckkonstant, adiabatisch ----
r = ct.IdealGasConstPressureReactor(gas, energy="on")
sim = ct.ReactorNet([r])

t_end = 0.1      # [s] Simulation bis 100 ms (bei T0=1000 K meist genug)
dt_save = 1e-5    # [s] Ausgabeauflösung
times, Ts, OH, CH4 = [], [], [], []

t = 0.0
while t < t_end:
    t_next = min(t + dt_save, t_end)
    sim.advance(t_next)
    t = t_next
    times.append(t)
    Ts.append(r.T)
    OH.append(r.thermo["OH"].X[0])
    CH4.append(r.thermo["CH4"].X[0])

# ---- 4) einfache Zündverzugsbestimmung (Maximum dT/dt) ----
times = np.array(times)
Ts = np.array(Ts)
dTdt = np.gradient(Ts, times)
tau_idx = np.argmax(dTdt)
tau = times[tau_idx]
print(f"Zündverzug (dT/dt-Max): {tau*1e3:.2f} ms")

# ---- 5) Plots ----
plt.figure()
plt.plot(times*1e3, Ts)
plt.xlabel("Zeit [ms]")
plt.ylabel("Temperatur [K]")
plt.title("CH4-Luft, p=const, adiabatisch")

plt.figure()
plt.plot(times*1e3, OH, label="OH")
plt.plot(times*1e3, CH4, label="CH4")
plt.xlabel("Zeit [ms]")
plt.ylabel("Molenbruch [-]")
plt.title("Speziesverlauf")
plt.legend()
plt.show()
