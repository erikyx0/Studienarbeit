import sys
from ROM import f
import os 


# Ordner, in dem ROM.py liegt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
 
gri = "gri30.yaml"
nuig = os.path.join(BASE_DIR, "Mechanismen", "NUIGMech1.1", "nuig_mech.yaml")

res_gri = f(gri)
res_nuig = f(nuig)

print(res_gri["case1"]["runtime_s"])
print(res_nuig["case1"]["runtime_s"])

