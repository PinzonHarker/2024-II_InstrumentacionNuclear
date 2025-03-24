from uncertainties import ufloat

# Definir valores con sus incertidumbres
ir = ufloat(135923.25, 978.03)
im = ufloat(276.40, 1.92)
cr = ufloat(0.44800, 0.003)

# Calcular la expresión con incertidumbre
resultado = (im * cr)*100 / ir
resultado2 = cr*100*0.000117*10**4
resultado3 = resultado*100*0.000117*10**4

# Imprimir resultado
print(f"Resultado: {resultado:.6f}")
print(f"Resultado2: {resultado2:.6f}")
print(f"Resultado3: {resultado3:.6f}")
0.000117