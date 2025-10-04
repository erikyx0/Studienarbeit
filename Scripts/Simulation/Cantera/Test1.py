from reactors import *

# --- Mechanismus / Gaszustand ---
gas = ct.Solution("gri30.yaml")  # oder dein Mechanismus
P0 = ct.one_atm
X_feed = "CH4:1, O2:0.5, N2:1.88"  # Beispiel
T_feed = 300.0

# --- Knoten bauen ---
net = Network()
net.add(make_reservoir(gas, T_feed, P0, X_feed, name="FEED"))
net.add(make_psr(gas, T0=1100.0, P=P0, X0=X_feed, V=1.0e-4, name="PSR1"))
net.add(make_pfr(gas, T0=1100.0, P=P0, X0=X_feed, area=1e-2, length=1.0, name="PFR1"))
net.add(make_cstr(gas, T0=1100.0, P=P0, X0=X_feed, V=2.0e-4, name="CSTR1"))
net.add(make_reservoir(gas, 300.0, P0, X_feed, name="VENT"))  # Senke

# --- Verbindungen (konstante mdot) ---
mdot = 0.1  # kg/s (Beispiel)
net.connect("FEED", "PSR1", mdot=mdot)
net.connect("PSR1", "PFR1", mdot=mdot)
net.connect("PFR1", "CSTR1", mdot=mdot)
net.connect("CSTR1", "VENT", mdot=mdot)

# Optional: Druckführung am Ende (z.B. Auslass)
# pc = net.add_pressure_controller("CSTR1", "VENT", master_name="CSTR1_TO_VENT", K=1e-4)

# --- Simulation laufen lassen ---
def report(t, nodes):
    psr = nodes["PSR1"].reactor
    pfr = nodes["PFR1"].reactor
    cstr = nodes["CSTR1"].reactor
    print(f"t = {t:8.4f} s | T(PSR)={psr.T:7.1f} K  T(PFR)={pfr.T:7.1f} K  T(CSTR)={cstr.T:7.1f} K")

net.advance(t_end=0.05, dt_report=0.01, reporter=report)
