# Definimos los valores del dado y su probabilidad
valores = np.array([1, 2, 3, 4, 5, 6])
probabilidades = np.ones(6) / 6  # Probabilidad uniforme para cada valor
probabilidades[1] += 1 / 12
probabilidades[4] -= 1 / 12

# Primer momento: media (valor esperado)
media = np.sum(valores * probabilidades)

# Segundo momento: varianza y desviación estándar
varianza = np.sum((valores - media) ** 2 * probabilidades)
desviacion_estandar = np.sqrt(varianza)

# Tercer momento: asimetría
asimetria = np.sum((valores - media) ** 3 * probabilidades) / (desviacion_estandar**3)

# Graficamos la distribución de probabilidad
plt.figure(figsize=(10, 6))
plt.scatter(
    valores,
    probabilidades,
    marker="o",
    color="magenta",
    edgecolors="blue",
    linewidths=0.6,
    s=9**2,
)

# Media
plt.axvline(media, color="red", linestyle="-", label=f"$\\mu$ = {media:.2f}")

# Marcamos el radio de la desviación estándar
plt.axvline(
    media - desviacion_estandar,
    color="#3953b2",
    linestyle="--",
    label=f"$\\mu - \\sigma$ = {media - desviacion_estandar:.2f}",
)
plt.axvline(
    media + desviacion_estandar,
    color="#3953b2",
    linestyle="--",
    label=f"$\\mu + \\sigma$ = {media + desviacion_estandar:.2f}",
)

# Graficar la asimetría como una línea vertical
plt.axvline(
    media + asimetria * desviacion_estandar,
    color="#165700",
    linestyle="--",
    label=f"$t$ = {asimetria:.2f}",
)

# Título y etiquetas
plt.title("Distribución de Probabilidad de un Dado Piramidal", pad=14, fontsize=18)
plt.xlabel("Valor Aleatorio $x$", fontsize=16)
plt.ylabel("Probabilidad", fontsize=16)
plt.legend()
