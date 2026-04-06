import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from uncertainties import ufloat

def linear_func(x, a0, a1):
    return a0 + a1 * x

def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def load_spectrum(file_path):
    data = pd.read_csv(file_path)
    if data.shape[1] < 2:
        raise ValueError("El archivo CSV debe tener al menos dos columnas: canal e intensidad.")
    channels = data.iloc[:, 0].values
    counts = data.iloc[:, 1].values
    return channels, counts

def find_peaks_in_spectrum(channels, counts, height_threshold):
    peaks, _ = find_peaks(counts, height=height_threshold)
    peak_channels = channels[peaks]
    peak_counts = counts[peaks]
    return peak_channels, peak_counts

def calibrate_energy(peak_channels, known_energies):
    popt, pcov = curve_fit(linear_func, peak_channels, known_energies)
    perr = np.sqrt(np.diag(pcov))
    a0 = ufloat(popt[0], perr[0])
    a1 = ufloat(popt[1], perr[1])
    return a0, a1

def fit_gaussian(x, y):
    mean_guess = x[np.argmax(y)]
    sigma_guess = np.std(x)
    A_guess = max(y)
    popt, pcov = curve_fit(gaussian, x, y, p0=[A_guess, mean_guess, sigma_guess])
    perr = np.sqrt(np.diag(pcov))
    return [ufloat(popt[i], perr[i]) for i in range(3)]

def analyze_new_spectrum():
    file_path = input("Ingrese el nombre del archivo CSV con el nuevo espectro: ")
    channels, counts = load_spectrum(file_path)
    
    a0 = float(input("Ingrese el valor de a0: "))
    a1 = float(input("Ingrese el valor de a1: "))
    energies = a0 + a1 * channels
    
    plt.plot(energies, counts)
    plt.xlabel("Energía (keV)")
    plt.ylabel("Cuentas")
    plt.title("Espectro Gamma - Ajustado en Energía")
    plt.show()
    
    height_threshold = float(input("Ingrese un umbral de altura para la detección de picos: "))
    peak_channels, peak_counts = find_peaks_in_spectrum(channels, counts, height_threshold)
    peak_energies = a0 + a1 * peak_channels
    
    plt.plot(energies, counts, label="Espectro")
    plt.scatter(peak_energies, peak_counts, color='red', label="Picos detectados")
    plt.xlabel("Energía (keV)")
    plt.ylabel("Cuentas")
    plt.title("Detección de Picos en el Espectro Gamma")
    plt.legend()
    plt.show()
    
    print("Picos detectados en las energías:")
    for i, peak in enumerate(peak_energies):
        print(f"{i + 1}: Energía {peak:.2f} keV, Cuentas {peak_counts[i]}")
    
    selected_indices = input("Ingrese los índices de los picos a estudiar (separados por comas): ")
    selected_indices = list(map(int, selected_indices.split(',')))
    
    for i in selected_indices:
        idx = np.where(channels == peak_channels[i - 1])[0][0]
        window = 10
        x_data = energies[idx - window:idx + window]
        y_data = counts[idx - window:idx + window]
        A, mu, sigma = fit_gaussian(x_data, y_data)
        print(f"Ajuste Gaussiano para el pico en {mu} keV:")
        print(f"  Amplitud (A) = {A}")
        print(f"  Media (mu) = {mu} keV")
        print(f"  Desviación estándar (sigma) = {sigma} keV")
    
if __name__ == "__main__":
    analyze_new_spectrum()