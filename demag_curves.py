import matplotlib.pyplot as plt
import numpy as np
import math
from Physics import materials_info


mags=materials_info

n48h = mags.n48h
n48sh = mags.n48sh

mag_type=1

if mag_type == 1:
    Br_20 = n48h['Br_20']
    mu_r = n48h['mu_r']
    electric_resistivity = n48h['electric_resistivity']
    Hc_20 = n48h['H_c']
    B_temp_coef = n48h['T_coef_remanence_flux_density']
    H_temp_coef = n48h['T_coef_intrinsic_coercivity']
    density = n48h['density']
    form_factor = n48h['form_factor']

if mag_type == 2:
    Br_20 = n48sh['Br_20']
    mu_r = n48sh['mu_r']
    electric_resistivity = n48sh['electric_resistivity']
    H_c = n48sh['H_c']
    T_coef_remanence_flux_density = n48sh['T_coef_remanence_flux_density']
    T_coef_intrinsic_coercivity = n48sh['T_coef_intrinsic_coercivity']
    density = n48sh['density']
    BH_max = n48sh['BH_max']
    form_factor = n48sh['form_factor']



# -----------------------------------------------------------------------------------


T=90    # Assumed magnet temperature
i=-10
mu_0 = (0.4 * np.pi)*10**-6


B_demag=[]
Bi_demag=[]
H_demag=[]
Hi_demag=[]

for i in range(-10, Hc_20, -100):
    Hc=Hc_20*(1+H_temp_coef/100 * (T-20))
    Br=Br_20*(1+B_temp_coef/100 * (T-20))
    B=Br*(1-i/Hc)  # The 1.4 approximates a halbach array
    Bi= B + (mu_0 * i)
    i=i-10
    B_demag.append(B)
    H_demag.append(i)
    Bi_demag.append(Bi)

plt.plot(H_demag,B_demag)
plt.show()
# Source: https://www.arnoldmagnetics.com/products/neodymium-iron-boron-magnets/

    




