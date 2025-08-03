import numpy as np 

D = 360
R = D/2
r = (D/2)-135
h = 135



#%% Berechnen Brennzone
V= (h*np.pi/3) * (R**2 + R*r + r**2)
print(f"Brennzone: {V} mm^3")

#%% Berechnen PFR 
V_groß = D**2*np.pi/4 * (1335-135)
V_klein= np.pi * 100**3 / 3
print(f"Rest: {(V_groß + V_klein):.4e} mm^3")