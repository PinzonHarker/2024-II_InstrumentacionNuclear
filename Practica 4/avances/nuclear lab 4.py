import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import numpy as np

# Cargar el archivo TXT
archivo = "2021-11-15_cal_133Ba22Na137Cs60Co_600s_1.txt"  # Cambia esto por el nombre de tu archivo

# Parámetros de calibración
num_canales = 8191  # Número total de canales
energia_max_mev = 3.95  # Energía máxima en MeV 3.95

# Conversión de canal a energía en keV
def canal_a_energia(canal):
    return (canal / num_canales) * energia_max_mev * 1000  # Convertir MeV a keV

def keV_a_canal(energia_kev):
    return (energia_kev * num_canales) / (energia_max_mev * 1000)

def incertidumbre(valor):
    return round(0.02 * valor, 2)  # Suposición de 2% de incertidumbre relativa

def analizar_canales(archivo, sigma=2, zoom_rangos=None):
    # Leer el archivo
    df = pd.read_csv(archivo, delim_whitespace=True, names=["Canal", "Cuentas"], comment='#')
    
    # Convertir canales a energía en keV
    df["Energía_keV"] = df["Canal"].apply(canal_a_energia)
    
    # Aplicar filtro gaussiano para suavizar el ruido
    df["Cuentas_Filtradas"] = gaussian_filter1d(df["Cuentas"], sigma=sigma)
    
    # Detectar picos
    picos, _ = find_peaks(df["Cuentas_Filtradas"], prominence=100)
    top_picos = df.iloc[picos].sort_values(by="Energía_keV", ascending=True)[["Canal", "Energía_keV"]]
    
    # Imprimir tabla de picos detectados
    print("\nPicos detectados:")
    print(top_picos.to_string(index=False))
    
    # Energías de referencia y colores
    referencias = {
        "22Na": ("green", [511, 1275.5]),
        "133Ba": ("red", [81.0, 356.0]),
        "60Co": ("black", [1173.2, 1332.5]),
        "137Cs": ("blue", [661.7]),
    }
    
    # Graficar espectro completo
    plt.figure(figsize=(18, 6))
    plt.plot(df["Energía_keV"], df["Cuentas"], label="Espectro Original", color='lightgray', alpha=0.7)
    plt.plot(df["Energía_keV"], df["Cuentas_Filtradas"], label="Espectro Filtrado", color='royalblue')
    plt.scatter(top_picos["Energía_keV"], df.loc[picos, "Cuentas_Filtradas"], color='red', label="Picos detectados", zorder=3)
    
    for descripcion, (color, valores) in referencias.items():
        for valor in valores:
            plt.axvline(x=valor, color=color, linestyle='--', linewidth=1.5, alpha=0.8)
        plt.plot([], [], color=color, linestyle='--', linewidth=1.5, label=descripcion)  
    
    plt.xlabel("Energía (keV)")
    plt.ylabel("Cuentas")
    plt.title("Espectro con picos detectados y referencias")
    plt.legend()
    plt.yscale("log")  
    plt.grid(True)
    plt.show()
    
    # Si se especifican rangos de zoom, graficar cada uno
    if zoom_rangos:
        for i, (inicio, fin) in enumerate(zoom_rangos):
            plt.figure(figsize=(10, 4))
            df_zoom = df[(df["Energía_keV"] >= inicio) & (df["Energía_keV"] <= fin)]
            plt.plot(df_zoom["Energía_keV"], df_zoom["Cuentas"], label="Espectro Original", color='gray', alpha=0.7)
            plt.plot(df_zoom["Energía_keV"], df_zoom["Cuentas_Filtradas"], label="Espectro Filtrado", color='blue')
            plt.xlabel("Energía (keV)")
            plt.ylabel("Cuentas")
            plt.title(f"Zoom en el rango {inicio} - {fin} keV")
            plt.legend()
            plt.grid(True)
            plt.show()
    
    # Tabla de energías de referencia con canal ± incertidumbre
    energias_referencia = [122.1, 661.7, 1274.5, 1332.5, 2505.7]
    resultados = []
    
    for energia in energias_referencia:
        canal = keV_a_canal(energia)
        incertidumbre_canal = incertidumbre(canal)
        sigma_canal = 0.02 * canal  # Aproximación de resolución
        incertidumbre_sigma = incertidumbre(sigma_canal)
        
        resultados.append([
            energia,
            f"{canal:.2f} ± {incertidumbre_canal:.2f}",
            f"{sigma_canal:.2f} ± {incertidumbre_sigma:.2f}"
        ])
    
    df_resultados = pd.DataFrame(resultados, columns=["Energía (keV)", "Canal ± Incertidumbre", "Sigma ± Incertidumbre"])
    
    print("\nTabla de valores de referencia:")
    print(df_resultados.to_string(index=False))

# Definir rangos de zoom manual (puedes modificar estos valores)
  #  "22Na": ("green", [511, 1275.5]),
  #  "133Ba": ("red", [81.0, 356.0]),
  #  "60Co": ("black", [1173.2, 1332.5]),
  #  "137Cs": ("blue", [661.7]),
x=15
rangos_zoom = [(511-x, 511+x), (1275.5-x, 1275.5+x), (81.0-x, 81+x),(356-x,356+x),(1173.2-x,1173.2+x), (1332.5-x,1332.5+x),(661.7-x,661.7+x)]  # Ejemplo de rangos

# Ejecutar la función con zoom manual
analizar_canales(archivo, zoom_rangos=rangos_zoom)
