import pandas as pd
import numpy as np
import os

def load_spectrum(file_path):
    data = pd.read_csv(file_path)
    if data.shape[1] < 2:
        raise ValueError("El archivo CSV debe tener al menos dos columnas: canal e intensidad.")
    channels = data.iloc[:, 0].values
    counts = data.iloc[:, 1].values
    return channels, counts

def save_spectrum(file_path, channels, counts):
    df = pd.DataFrame({"Canal": channels, "Cuentas": counts})
    df.to_csv(file_path, index=False)

def process_spectra(background_file, input_dir, output_dir):
    # Cargar el fondo
    background_channels, background_counts = load_spectrum(background_file)
    
    # Asegurar que el directorio de salida existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Procesar todos los archivos en el directorio de entrada
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # Cargar espectro
            channels, counts = load_spectrum(input_path)
            
            # Verificar que los canales coincidan con los del fondo
            if not np.array_equal(channels, background_channels):
                print(f"Advertencia: Los canales en {filename} no coinciden con el fondo.")
                continue
            
            # Restar el fondo
            corrected_counts = counts - background_counts
            corrected_counts[corrected_counts < 0] = 0  # Evitar valores negativos
            
            # Guardar el espectro corregido
            save_spectrum(output_path, channels, corrected_counts)
            print(f"Procesado: {filename} -> {output_path}")

if __name__ == "__main__":
    background_file = input("Ingrese el archivo CSV con el fondo: ")
    input_dir = input("Ingrese el directorio con los espectros a corregir: ")
    output_dir = input("Ingrese el directorio donde guardar los espectros corregidos: ")
    process_spectra(background_file, input_dir, output_dir)