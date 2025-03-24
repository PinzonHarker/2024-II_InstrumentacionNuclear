import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import numpy as np

# Cargar el archivo TXT
archivo ="calibracionNaCoCoCs-Ge8K.xy"  # Cambia esto por el nombre de tu archivo

# Parámetros de calibración
num_canales = 8000  # Número total de canales
energia_max_mev = 2.8  # Energía máxima en MeV

# Conversión de canal a energía en keV
def canal_a_energia(canal):
    return (canal / num_canales) * energia_max_mev * 1000  # Convertir MeV a keV

# Función para calcular el borde de Compton en keV
def borde_compton(energia_foton):
    m_e = 1  # Energía en keV del electrón en reposo
    return energia_foton * (1 - (1 / (1 + (2 * energia_foton / m_e))))

def analizar_canales(archivo, top_n=10, sigma=1):
    # Leer el archivo, asumiendo que tiene dos columnas sin encabezado
    df = pd.read_csv(archivo, delim_whitespace=True, names=["Canal", "Cuentas"], comment='#')
    
    # Convertir canales a energía en keV
    df["Energía_keV"] = df["Canal"].apply(canal_a_energia)
    
    # Aplicar filtro gaussiano para suavizar el ruido
    df["Cuentas_Filtradas"] = gaussian_filter1d(df["Cuentas"], sigma=sigma)
    
    # Ajustar height dinámicamente como un porcentaje del máximo
    height_factor = 0.05 * max(df["Cuentas_Filtradas"])  # 5% del valor máximo
    
    # Detectar picos en la señal filtrada usando prominence en lugar de height
    picos, _ = find_peaks(df["Cuentas_Filtradas"], prominence=1700)
    
    # Obtener todos los picos detectados sin filtrar top_n inicialmente
    top_picos = df.iloc[picos].sort_values(by="Cuentas_Filtradas", ascending=False).copy()
    
    # Ordenar los picos por energía en keV en orden ascendente
    top_picos = top_picos.sort_values(by="Energía_keV", ascending=True)
    
    # Calcular los bordes de Compton para cada pico detectado en keV
    top_picos["Borde_Compton"] = top_picos["Energía_keV"].apply(borde_compton)
    
    print("Picos detectados en el espectro:")
    print(top_picos[["Energía_keV", "Cuentas_Filtradas", "Borde_Compton"]])
    
    # Graficar espectro completo antes y después del filtrado
    plt.figure(figsize=(18, 6))
    plt.plot(df["Energía_keV"], df["Cuentas"], label="Espectro Original", color='lightgray', alpha=0.7)
    plt.plot(df["Energía_keV"], df["Cuentas_Filtradas"], label="Espectro Filtrado (Gaussiano)", color='royalblue')
    plt.scatter(top_picos["Energía_keV"], top_picos["Cuentas_Filtradas"], color='red', label="Picos detectados", zorder=3)
    
    # Resaltar las regiones de Compton
    #for _, row in top_picos.iterrows():
     #   plt.axvline(x=row["Borde_Compton"], color='orange', linestyle='--', alpha=1, label="Borde Compton")
    
    # ➤ AÑADE ESTE BLOQUE AQUÍ
# Lista de energías específicas en keV donde deseas colocar líneas verticales
    valores_especificos = [511, 1275.5]  # Cambia estos valores según necesites

# Graficar líneas verticales en los valores específicos
    for valor in valores_especificos:
        plt.axvline(x=valor, color='green', linestyle='--', linewidth=1.5, alpha=0.8, label=f"{valor} keV")

# Ajustar la leyenda para evitar repeticiones
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # ➤ AÑADE ESTE BLOQUE AQUÍ
# Lista de energías específicas en keV donde deseas colocar líneas verticales
    valores_especificos1 = [122.1, 136.5]  # Cambia estos valores según necesites

# Graficar líneas verticales en los valores específicos
    for valor in valores_especificos1:
        plt.axvline(x=valor, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label=f"{valor} keV")

# Ajustar la leyenda para evitar repeticiones
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # ➤ AÑADE ESTE BLOQUE AQUÍ
# Lista de energías específicas en keV donde deseas colocar líneas verticales
    valores_especificos2 = [1173.2, 1332.5]  # Cambia estos valores según necesites

# Graficar líneas verticales en los valores específicos
    for valor in valores_especificos2:
        plt.axvline(x=valor, color='black', linestyle='--', linewidth=1.5, alpha=0.8, label=f"{valor} keV")

# Ajustar la leyenda para evitar repeticiones
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    
    # ➤ AÑADE ESTE BLOQUE AQUÍ
# Lista de energías específicas en keV donde deseas colocar líneas verticales
    valores_especificos3 = [661.7]  # Cambia estos valores según necesites

# Graficar líneas verticales en los valores específicos
    for valor in valores_especificos3:
        plt.axvline(x=valor, color='blue', linestyle='--', linewidth=1.5, alpha=0.8, label=f"{valor} keV")

# Ajustar la leyenda para evitar repeticiones
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.xlabel("Energía (keV)")
    plt.ylabel("Cuentas")
    plt.title("Espectro con picos detectados, filtrado de ruido y bordes de Compton")
    plt.legend()
    plt.yscale("log")  # Escala logarítmica para mejor visualización
    plt.grid(True)
    plt.show()

# Ejecutar la función
analizar_canales(archivo)
