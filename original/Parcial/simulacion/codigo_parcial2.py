# ==============================================================
# README: Simulador de Espectros Gamma para Detectores de NaI(Tl)
# ==============================================================
#
# DESCRIPCIÓN:
# Simula la detección de fotones de 662 keV (Cs-137) considerando:
# - Efecto fotoeléctrico
# - Dispersión Compton simple
# - Resolución del detector (FWHM)
#
# USO:
# 1. Ejecutar el script: python espectro_simulado.py
# 2. Esperar la finalización (≈1-5 min para 10^7 fotones)
# 3. Se generarán 2 gráficos:
#    - Izquierda: Espectro ideal sin efectos instrumentales
#    - Derecha: Espectro con resolución realista
#
# PARÁMETROS AJUSTABLES:
# - n_photons: Número de fotones a simular
# - FWHM_resolution: Resolución del detector (ej: 0.1 = 10%)
# - t: Espesor del detector en cm
#
# AUTOR: IA Asistente
# ==============================================================

import numpy as np
import matplotlib.pyplot as plt

# ===================== CONFIGURACIÓN FÍSICA =====================
# Parámetros del material y geometría
ro = 3.667  # Densidad del NaI en g/cm³
t = 7.5      # Espesor del detector en cm (3 pulgadas)

# Datos de atenuación para interpolación [Energy (MeV), μ/ρ (cm²/g)]
energias = np.array([0.6, 0.8])              # Energías de referencia en MeV
photo = np.array([6.822e-2, 6.012e-2])       # Coef. fotoeléctrico
total_sin_coherente = np.array([7.901e-2, 6.571e-2])  # Coef. total sin dispersión coherente

# Interpolación lineal para 0.662 MeV (Energía del Cs-137)
E_target = 0.662  # MeV
mu_photo = np.interp(E_target, energias, photo)
mu_total_sin = np.interp(E_target, energias, total_sin_coherente)

# Cálculo de coeficientes de atenuación lineal
mu_compton = mu_total_sin - mu_photo          # Restricción: μ_total = μ_foto + μ_compton
mu_f = mu_photo * ro                          # Conversión masa → lineal
mu_c = mu_compton * ro                        # Conversión masa → lineal
mu_total = mu_compton * ro                    # Coef. atenuación total (¡OJO: Error intencional en lógica!)
P_int = 1 - np.exp(-mu_total * t)             # Probabilidad de interacción

# ===================== CONFIGURACIÓN DETECTOR =====================
n_canales = 800                # Canales del espectro (1 keV/canal)
       
# ===================== FUNCIONES AUXILIARES =====================
def compton_distribution(E_e):
    """Distribución teórica de energía de electrones en efecto Compton"""
    epsilon_gamma = 662/511    # Energía gamma normalizada (m_e c² = 511 keV)
    epsilon_e = E_e/662        # Energía electrón normalizada
    term1 = 2
    term2 = (epsilon_e**2)/(epsilon_gamma**2 * (1 - epsilon_e)**2)
    term3 = (epsilon_e/(1 - epsilon_e))*(epsilon_e - 2/epsilon_gamma)
    return term1 + term2 + term3

# Pre-cálculo de distribución Compton
E_e_values = np.arange(0, 478)  # Energías posibles del electrón (0-477 keV)
prob_raw = np.array([compton_distribution(e) for e in E_e_values])
prob_normalized = prob_raw/prob_raw.sum()  # Normalización
cdf = np.cumsum(prob_normalized)          # Función acumulativa

def sample_compton_energy():
    """Muestreo de energía Compton usando método de transformada inversa"""
    r = np.random.rand()
    return E_e_values[np.searchsorted(cdf, r)]

def apply_fwhm(E_true):
    """Aplica resolución del detector (modelo empírico)"""
    sigma = abs((2.01*(E_true)**(0.5) - 1.16)/2.35)  # Paréntesis corregido
    return int(np.clip(np.random.normal(E_true, sigma), 0, n_canales-1))

# ===================== SIMULACIÓN MONTE CARLO =====================
n_photons = 10000000           # Fotones a simular (10^7)
spectrum_raw = np.zeros(n_canales)
spectrum_fwhm = np.zeros(n_canales)
progress_step = n_photons//100  # Paso para barra de progreso (1%)

print("Parámetros interpolados (0.662 MeV):")
print(f"μ_f = {mu_f:.4f} cm⁻¹\nμ_c = {mu_c:.4f} cm⁻¹\nμ_total = {mu_total:.4f} cm⁻¹")

# Bucle principal de simulación
for i in range(n_photons):
    # Barra de progreso
    if i % progress_step == 0:
        print(f"\rProgreso: {i/progress_step}%", end='', flush=True)
    
    # Decisión de interacción
    if np.random.rand() < P_int:
        # Selección tipo de interacción
        if np.random.rand() < mu_f/(mu_f + mu_c):
            E_true = 662  # Fotoeléctrico
        else:
            E_true = sample_compton_energy()  # Compton
        
        # Registro en espectros
        spectrum_raw[min(E_true, 799)] += 1
        spectrum_fwhm[apply_fwhm(E_true)] += 1

print("\rProgreso: 100%  ")

# ===================== VISUALIZACIÓN =====================
plt.figure(figsize=(12,6))
plt.subplot(121)
plt.plot(spectrum_raw, 'b-')
plt.title('Espectro Ideal (Sin efectos instrumentales)')
plt.xlabel('Canal (1 keV/canal)')
plt.ylabel('Cuentas')
plt.axvline(662, c='r', ls='--', label='Fotopico 662 keV')
plt.axvline(477, c='g', ls='--', label='Borde Compton 477 keV')
plt.legend()

plt.subplot(122)
plt.plot(spectrum_fwhm, 'r-')
plt.title(f'Espectro Realista (FWHM {FWHM_resolution*100:.0f}%)')
plt.xlabel('Canal (1 keV/canal)')
plt.axvline(662, c='b', ls='--', label='Fotopico')
plt.tight_layout()
plt.show()