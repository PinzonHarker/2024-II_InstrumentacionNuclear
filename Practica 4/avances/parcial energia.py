import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from uncertainties import ufloat

def linear_func(x, a0, a1):
    return a0 + a1 * x

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

def main():
    file_path = input("Ingrese el nombre del archivo CSV con el espectro: ")
    channels, counts = load_spectrum(file_path)
    
    plt.plot(channels, counts)
    plt.xlabel("Canal")
    plt.ylabel("Cuentas")
    plt.title("Espectro Gamma")
    plt.show()
    
    height_threshold = float(input("Ingrese un umbral de altura para la detección de picos: "))
    peak_channels, peak_counts = find_peaks_in_spectrum(channels, counts, height_threshold)
    
    plt.plot(channels, counts, label="Espectro")
    plt.scatter(peak_channels, peak_counts, color='red', label="Picos detectados")
    plt.xlabel("Canal")
    plt.ylabel("Cuentas")
    plt.title("Detección de Picos en el Espectro Gamma")
    plt.legend()
    plt.show()
    
    print("Picos detectados en los canales:")
    for i, peak in enumerate(peak_channels):
        print(f"{i + 1}: Canal {peak}, Cuentas {peak_counts[i]}")
    
    selected_indices = input("Ingrese los índices de los picos seleccionados para calibrar (separados por comas): ")
    selected_indices = list(map(int, selected_indices.split(',')))
    
    selected_channels = [peak_channels[i - 1] for i in selected_indices]
    known_energies = list(map(float, input("Ingrese los valores de energía correspondientes (keV), separados por comas: ").split(',')))
    
    if len(selected_channels) != len(known_energies):
        print("Error: La cantidad de canales seleccionados y las energías ingresadas no coinciden.")
        return
    
    a0, a1 = calibrate_energy(selected_channels, known_energies)
    print(f"Parámetros de calibración:")
    print(f"a0 = {a0} keV")
    print(f"a1 = {a1} keV/canal")

if __name__ == "__main__":
    main()
