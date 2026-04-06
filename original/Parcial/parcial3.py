import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import ipywidgets as wid
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import colorcet as cc
import os
import copy
from matplotlib.patches import FancyBboxPatch
import uncertainties as un
from uncertainties.umath import *


# Configuración general de las gráficas
mpl.rcParams.update(
    {
        "legend.fontsize": 20,
        "axes.labelsize": 24,  # Updated from 20 to 24
        "axes.titlesize": 24,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "figure.titlesize": 24,
        "axes.titlepad": 10,
        "text.usetex": True,
        "font.family": "Times New Roman",
        "mathtext.fontset": "dejavusans",
        "font.size": 20,
        "axes.labelweight": "bold",
        "axes.grid.which": "both",
        "axes.grid": True,
        "grid.alpha": 0.5,
        "axes.formatter.limits": (-3, 3),  # Use scientific notation for values outside the range 10^-3 to 10^3
        'figure.subplot.bottom': 0.2,
    }
)

colors = cc.glasbey_dark
# Configuracion de colores
k = 3

# Rotar la lista: número positivo para rotar a la derecha, negativo para rotar a la izquierda
colors = colors[-k:] + colors[:-k]

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors)

# Test
if False:
    plt.figure(figsize=(5, 5))
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16], "o")
    plt.plot([4, 3, 2, 1], [1, 2, 3, 4], "o")
    plt.plot([1, 2, 3, 4], [1, 2, 3, 4], "s")
    plt.plot([4, 3, 2, 1], [1, 4, 9, 16], "s")
    plt.title("Test")
    plt.xlabel("X-axis Label", fontweight="bold")
    plt.ylabel("Y-axis Label")
    plt.show()

# ===== PARÁMETROS FÍSICOS =====
ro = 3.667  # Densidad del NaI (g/cm³)

# Datos de la tabla (energías en MeV)
energias = np.array([0.6, 0.8])  # MeV
photo = np.array([6.822e-2, 6.012e-2])  # cm²/g (Photoelectric)
total_sin_coherente = np.array([7.901e-2, 6.571e-2])  # cm²/g (Total Without Coherent)

# Interpolación lineal para 0.662 MeV
E_target = 0.662  # MeV
frac = (E_target - energias[0]) / (energias[1] - energias[0])

# Interpolar valores
mu_photo = np.interp(E_target, energias, photo)
mu_total_sin = np.interp(E_target, energias, total_sin_coherente)

# Calcular Compton: Total sin coherente - Photoelectric (ignorando pair production)
mu_compton = mu_total_sin - mu_photo

# Convertir a coeficientes lineales
mu_f = mu_photo * ro
mu_c = mu_compton * ro

t = 7.5  # cm (3 pulgadas)
mu_total = mu_compton * ro
P_int = 1 - np.exp(-mu_total * t)

print(f"Parámetros interpolados (0.662 MeV):")
print(f"μ_f = {mu_f:.4f} cm⁻¹")
print(f"μ_c = {mu_c:.4f} cm⁻¹")
print(f"μ_total = {mu_total:.4f} cm⁻¹")

# ===== RESTOR CODE =====
# Parámetros del detector
n_canales = 800
FWHM_resolution = 0.1

# Función de distribución Compton (igual que antes)
def compton_distribution(E_e):
    epsilon_gamma = 662/511
    epsilon_e = E_e/662
    term1 = 2
    term2 = (epsilon_e**2)/(epsilon_gamma**2 * (1 - epsilon_e)**2)
    term3 = (epsilon_e/(1 - epsilon_e))*(epsilon_e - 2/epsilon_gamma)
    return term1 + term2 + term3

E_e_values = np.arange(0, 478)
prob_raw = np.array([compton_distribution(e) for e in E_e_values])
prob_normalized = prob_raw/prob_raw.sum()
cdf = np.cumsum(prob_normalized)

# Funciones de muestreo (igual que antes)
def sample_compton_energy():
    r = np.random.rand()
    return E_e_values[np.searchsorted(cdf, r)]

def apply_fwhm(E_true):
    sigma = abs((2.01*(E_true)**(0.5)-1.16)/2.35)
    return int(np.clip(np.random.normal(E_true, sigma), 0, n_canales-1))
    
# Simulación con barra de progreso
n_photons = 10000000
spectrum_raw = np.zeros(n_canales)
spectrum_fwhm = np.zeros(n_canales)
progress_step = n_photons//100

for i in range(n_photons):
    if i % progress_step == 0:
        print(f"\rProgreso: {i/progress_step}%", end='', flush=True)
    
    if np.random.rand() < P_int:
        if np.random.rand() < mu_f/(mu_f + mu_c):
            E_true = 662
        else:
            E_true = sample_compton_energy()
        
        spectrum_raw[min(E_true, 799)] += 1
        E_measured = apply_fwhm(E_true)
        spectrum_fwhm[E_measured] += 1

print("\rProgreso: 100%  ")

# Visualización (igual que antes)
plt.figure(figsize=(12,6))
plt.subplot(121)
plt.plot(spectrum_raw, color='darkred', lw=2)
plt.title('Espectro bruto')
plt.axvline(662, color='darkblue', ls='--', lw=1.5)
plt.axvline(477, color='purple', ls='--', lw=1.5)

plt.subplot(122)
plt.plot(spectrum_fwhm, color='darkred', lw=2)
plt.title('Espectro con FWHM')
plt.axvline(662, c='darkblue', ls='--', lw=1.5)
plt.axvline(477, color='purple', ls='--', lw=1.5)

# Guardar la figura
plt.savefig('spectrum.pdf')
plt.show()

