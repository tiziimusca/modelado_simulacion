import sympy as sp
import math
import numpy as np
import matplotlib.pyplot as plt

DEC = 8  # cantidad de decimales


def fmt(x):
    return f"{x:.{DEC}f}"


def formatear_polinomio(poly):
    coef = poly.c
    grado = len(coef) - 1
    terminos = []

    for i, c in enumerate(coef):
        potencia = grado - i

        if abs(c) < 1e-12:
            continue

        c_str = f"{c:.{DEC}f}"

        if potencia == 0:
            terminos.append(f"{c_str}")
        elif potencia == 1:
            terminos.append(f"{c_str}x")
        else:
            terminos.append(f"{c_str}x^{potencia}")

    return " + ".join(terminos).replace("+ -", "- ")


def lagrange_prolijo_8dec(x_vals, y_vals, x_eval, f_real=None, M=None):

    n = len(x_vals)

    print("\n" + "="*75)
    print("📌 INTERPOLACIÓN DE LAGRANGE (8 DECIMALES)")
    print("="*75)

    # ------------------------
    # DATOS
    # ------------------------
    print("\n🔹 Datos:")
    for i in range(n):
        print(f"   x{i} = {fmt(x_vals[i])}   →   y{i} = {fmt(y_vals[i])}")

    print(f"\n🔹 Evaluar en: x = {fmt(x_eval)}")

    # ------------------------
    # CONSTRUCCIÓN
    # ------------------------
    print("\n" + "-"*75)
    print("🔹 Construcción del polinomio")
    print("-"*75)

    P_poly = np.poly1d([0])

    for i in range(n):

        print(f"\n➡️  Base L_{i}(x):")

        Li_poly = np.poly1d([1])

        for j in range(n):
            if i != j:
                numerador = np.poly1d([1, -x_vals[j]])
                denominador = (x_vals[i] - x_vals[j])

                print(
                    f"   (x - {fmt(x_vals[j])}) / ({fmt(x_vals[i])} - {fmt(x_vals[j])})")

                Li_poly *= numerador / denominador

        print(f"   L_{i}(x) = {formatear_polinomio(Li_poly)}")

        termino = np.poly1d(y_vals[i] * Li_poly)

        print(f"   y{i} * L_{i}(x) = {fmt(y_vals[i])} * L_{i}(x)")
        print(f"   → {formatear_polinomio(termino)}")

        P_poly = np.poly1d(P_poly + termino)

    # ------------------------
    # RESULTADO FINAL
    # ------------------------
    print("\n" + "="*75)
    print("✅ Polinomio interpolante final:")
    print(f"P(x) = {formatear_polinomio(P_poly)}")
    print("="*75)

    # ------------------------
    # EVALUACIÓN
    # ------------------------
    P_val = P_poly(x_eval)

    print("\n🔹 Evaluación:")
    print(f"   P({fmt(x_eval)}) = {fmt(P_val)}")

    # ------------------------
    # ERROR REAL
    # ------------------------
    if f_real is not None:
        real_val = f_real(x_eval)
        error = abs(real_val - P_val)

        print("\n🔹 Error real:")
        print(f"   f({fmt(x_eval)}) = {fmt(real_val)}")
        print(f"   Error = |f - P| = {fmt(error)}")

    # ------------------------
    # COTA
    # ------------------------
    if M is not None:
        print("\n🔹 Cota del error:")

        producto = 1
        for xi in x_vals:
            diff = abs(x_eval - xi)
            print(f"   |{fmt(x_eval)} - {fmt(xi)}| = {fmt(diff)}")
            producto *= diff

        cota = (M / math.factorial(n)) * producto

        print(f"\n   Producto = {fmt(producto)}")
        print(f"   Cota ≤ {fmt(cota)}")

    print("\n" + "="*75 + "\n")

    return P_poly


def cota_error_global(x_vals, M, a, b, num_puntos=1000):

    xs = np.linspace(a, b, num_puntos)

    max_producto = 0

    for x in xs:
        producto = 1
        for xi in x_vals:
            producto *= abs(x - xi)

        if producto > max_producto:
            max_producto = producto

    n = len(x_vals)

    cota = (M / math.factorial(n)) * max_producto

    print("\n🔹 COTA GLOBAL DEL ERROR")
    print(f"Intervalo: [{a}, {b}]")
    print(f"Máx producto = {max_producto:.8f}")
    print(f"Cota global ≤ {cota:.8f}")

    return cota


def graficar_interpolacion(x_vals, y_vals, P_poly, titulo, f_real=None, a=None, b=None):

    if a is None:
        a = min(x_vals)
    if b is None:
        b = max(x_vals)

    margen = 0.15 * (b - a if b > a else 1)
    xs = np.linspace(a - margen, b + margen, 500)
    ys_poly = P_poly(xs)

    plt.figure(figsize=(9, 5))
    plt.plot(xs, ys_poly, linewidth=2, label="P(x) interpolante")

    if f_real is not None:
        plt.plot(xs, f_real(xs), linestyle="--",
                 linewidth=2, label="Función real")

    plt.scatter(x_vals, y_vals, color="red", s=55, label="Nodos")
    plt.title(titulo)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.show()


def imprimir_tabla(titulo, encabezados, filas, anchos=None, decimales=6, max_filas=30):
    print("\n" + "=" * 110)
    print(titulo)
    print("=" * 110)

    if anchos is None:
        anchos = [18] * len(encabezados)

    encabezado_txt = "".join(
        f"{encabezados[i]:<{anchos[i]}}" for i in range(len(encabezados)))
    print(encabezado_txt)
    print("-" * sum(anchos))

    # Limitar cantidad de filas a imprimir para no saturar la consola
    filas_a_mostrar = filas[:max_filas]

    for fila in filas_a_mostrar:
        linea = ""
        for i, valor in enumerate(fila):
            if isinstance(valor, (float, np.floating)):
                linea += f"{valor:<{anchos[i]}.{decimales}f}"
            else:
                linea += f"{str(valor):<{anchos[i]}}"
        print(linea)

    if len(filas) > max_filas:
        print(f"... (mostrando {max_filas} de {len(filas)} registros) ...")


def ejecutar_ejemplos_aplicados():
    ejemplos = [
        {
            "area": "Física",
            "descripcion": "Aproximación de una trayectoria suave: f(x) = sin(x)",
            "funcion": lambda x: np.sin(x),
            "x_vals": np.array([0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]),
            "x_eval": np.pi / 3,
        },
        {
            "area": "Ingeniería",
            "descripcion": "Respuesta térmica: f(x) = e^(-x/2)",
            "funcion": lambda x: np.exp(-x / 2),
            "x_vals": np.array([0.0, 1.0, 2.0, 3.0]),
            "x_eval": 1.5,
        },
        {
            "area": "Economía",
            "descripcion": "Curva de costo: f(x) = 30 + 8x - 0.5x^2",
            "funcion": lambda x: 30 + 8 * x - 0.5 * x**2,
            "x_vals": np.array([0.0, 1.5, 3.0, 4.5]),
            "x_eval": 2.5,
        },
        {
            "area": "Ciencia de datos",
            "descripcion": "Pérdida suavizada: f(x) = (x - 0.7)^2 + 0.1",
            "funcion": lambda x: (x - 0.7)**2 + 0.1,
            "x_vals": np.array([0.0, 0.5, 1.0, 1.5]),
            "x_eval": 0.9,
        },
        {
            "area": "Redes neuronales",
            "descripcion": "Activación: f(x) = tanh(x)",
            "funcion": lambda x: np.tanh(x),
            "x_vals": np.array([-1.5, -0.5, 0.5, 1.5]),
            "x_eval": 0.2,
        },
        {
            "area": "Optimización en machine learning",
            "descripcion": "Función objetivo: f(x) = (x - 0.03)^2 + 0.02",
            "funcion": lambda x: (x - 0.03)**2 + 0.02,
            "x_vals": np.array([0.0, 0.02, 0.05, 0.1]),
            "x_eval": 0.04,
        },
    ]

    filas = []

    for ejemplo in ejemplos:
        x_vals = ejemplo["x_vals"]
        y_vals = np.array([ejemplo["funcion"](x) for x in x_vals], dtype=float)

        print("\n" + "=" * 80)
        print(f"EJEMPLO - {ejemplo['area']}")
        print(ejemplo["descripcion"])
        print("=" * 80)

        P_poly = lagrange_prolijo_8dec(
            x_vals,
            y_vals,
            x_eval=ejemplo["x_eval"],
            f_real=ejemplo["funcion"],
            M=None,
        )

        P_eval = P_poly(ejemplo["x_eval"])
        f_eval = ejemplo["funcion"](ejemplo["x_eval"])

        filas.append([
            ejemplo["area"],
            ejemplo["x_eval"],
            P_eval,
            f_eval,
            abs(f_eval - P_eval),
        ])

        graficar_interpolacion(
            x_vals,
            y_vals,
            P_poly,
            titulo=f"{ejemplo['area']}, {ejemplo['descripcion']} - Interpolación de Lagrange",
            f_real=ejemplo["funcion"],
            a=float(np.min(x_vals)),
            b=float(np.max(x_vals)),
        )

    imprimir_tabla(
        "RESUMEN DE EJEMPLOS APLICADOS",
        ["Área", "x_eval", "P(x_eval)", "f(x_eval)", "Error"],
        filas,
        anchos=[24, 14, 16, 16, 16],
        decimales=8,
    )


def derivada_simbolica_max(expr, var, orden, a, b):

    deriv = sp.diff(expr, var, orden)

    print(f"\nDerivada de orden {orden}:")
    print(deriv)

    f_lambd = sp.lambdify(var, deriv, "numpy")

    xs = np.linspace(a, b, 1000)
    valores = np.abs(f_lambd(xs))

    M = np.max(valores)

    print(f"\nMáximo en [{a}, {b}] ≈ {M:.8f}")

    return M


def f(x):
    return np.sin(x)


x = sp.symbols('x')
expr = sp.sin(x)

x_vals = [0, np.pi/2, np.pi]
y_vals = [f(x) for x in x_vals]

# valor máximo de la derivada de orden n+1 en el intervalo [a,b]
M = derivada_simbolica_max(expr, x, orden=3, a=x_vals[0], b=x_vals[-1])

lagrange_prolijo_8dec(x_vals, y_vals, x_eval=np.pi/4, f_real=f, M=M)
cota_error_global(x_vals, M=M, a=x_vals[0], b=x_vals[-1])


if __name__ == "__main__":
    ejecutar_ejemplos_aplicados()
