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

def fit_gaussian(channels, counts, peak_pos):
    window = 10  # Define un rango para el ajuste
    mask = (channels >= peak_pos - window) & (channels <= peak_pos + window)
    x_data = channels[mask]
    y_data = counts[mask]
    
    if len(x_data) < 3:
        return peak_pos  # Si no hay suficientes puntos, devolver el pico detectado
    
    p0 = [max(y_data), peak_pos, 3]  # Valores iniciales para el ajuste
    try:
        popt, _ = curve_fit(gaussian, x_data, y_data, p0=p0)
        return popt[1]  # Devuelve el centroide ajustado (mu)
    except:
        return peak_pos  # Si el ajuste falla, devolver el pico detectado

def load_spectrum(file_path):
    data = pd.read_csv(file_path)
    if data.shape[1] < 2:
        raise ValueError("El archivo CSV debe tener al menos dos columnas: canal e intensidad.")
    channels = data.iloc[:, 0].values
    counts = data.iloc[:, 1].values
    return channels, counts

def find_peaks_in_spectrum(channels, counts, height_threshold, channel_range=None):
    if channel_range:
        mask = (channels >= channel_range[0]) & (channels <= channel_range[1])
        channels = channels[mask]
        counts = counts[mask]
    peaks, _ = find_peaks(counts, height=height_threshold)
    peak_channels = channels[peaks]
    peak_counts = counts[peaks]
    
    centroids = [fit_gaussian(channels, counts, peak) for peak in peak_channels]
    return np.array(centroids), peak_counts

def calibrate_energy(peak_channels, known_energies):
    popt, pcov = curve_fit(linear_func, peak_channels, known_energies)
    perr = np.sqrt(np.diag(pcov))
    a0 = ufloat(popt[0], perr[0])
    a1 = ufloat(popt[1], perr[1])
    return a0, a1

def analyze_spectrum(channels, counts, height_threshold, channel_range=None):
    peak_channels, peak_counts = find_peaks_in_spectrum(channels, counts, height_threshold, channel_range)
    max_view = max(peak_channels)+100
    plt.plot(channels, counts, label="Espectro")
    plt.scatter(peak_channels, peak_counts, color='red', label="Centroides de Gaussiana")
    plt.xlabel("Canal")
    plt.xlim(0, max_view)
    plt.yscale("log") 
    plt.ylabel("Cuentas")
    plt.title("Ajuste de Gaussiana en Picos del Espectro Gamma")
    plt.legend()
    plt.show()
    
    print("Centroides ajustados en los canales:")
    for i, peak in enumerate(peak_channels):
        print(f"{i + 1}: Canal {peak:.2f}, Cuentas {peak_counts[i]}")
    
    return peak_channels, peak_counts

def main():
    file_path = input("Ingrese el nombre del archivo CSV con el espectro: ")
    channels, counts = load_spectrum(file_path)
    height_threshold = float(input("Ingrese un umbral de altura para la detección de picos: "))
    peak_channels, peak_counts = analyze_spectrum(channels, counts, height_threshold)
    
    add_region = input("¿Desea agregar una región de análisis adicional? (s/n): ").strip().lower()
    if add_region == 's':
        height_threshold2 = float(input("Ingrese un nuevo umbral de altura para la segunda región: "))
        channel_min = float(input("Ingrese el canal mínimo para la región adicional: "))
        channel_max = float(input("Ingrese el canal máximo para la región adicional: "))
        additional_peaks, additional_counts = analyze_spectrum(channels, counts, height_threshold2, (channel_min, channel_max))
        peak_channels = np.concatenate((peak_channels, additional_peaks))
        peak_counts = np.concatenate((peak_counts, additional_counts))
        print("Nuevos Centroides ajustados en los canales:")
        for i, peak in enumerate(peak_channels):
            print(f"{i + 1}: Canal {peak:.2f}, Cuentas {peak_counts[i]}")
    
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