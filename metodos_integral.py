import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


def obtener_M2_M4(expr, var, a, b, puntos=2000):

    print("\n" + "="*60)
    print("📌 CÁLCULO AUTOMÁTICO DE M2 Y M4")
    print("="*60)

    # Derivadas
    f2 = sp.diff(expr, var, 2)
    f4 = sp.diff(expr, var, 4)

    print("\n🔹 f''(x) =")
    print(f2)

    print("\n🔹 f''''(x) =")
    print(f4)

    # Convertir a funciones numéricas
    f2_num = sp.lambdify(var, f2, "numpy")
    f4_num = sp.lambdify(var, f4, "numpy")

    xs = np.linspace(a, b, puntos)

    vals_f2 = np.abs(f2_num(xs))
    vals_f4 = np.abs(f4_num(xs))

    M2 = np.max(vals_f2)
    M4 = np.max(vals_f4)

    print("\n🔹 Resultados:")
    print(f"M2 = max |f''(x)| ≈ {M2:.8f}")
    print(f"M4 = max |f''''(x)| ≈ {M4:.8f}")

    print("="*60 + "\n")

    return M2, M4


x = sp.symbols('x')

# =========================================================
# CONFIGURACIÓN
# =========================================================

expr = np.e**(-x**2)  # CAMBIAR POR LA FUNCIÓN simbólica que quieras integrar


def f(x):
    # if (abs(x)) < 1e-8:
    #    return 1  # CAMBIAR SI HAY QUE APLICAR LHOPITAL
    return np.e**(-x**2)  # CAMBIAR POR LA FUNCIÓN QUE QUIERAS INTEGRAR


a = 0
b = 1

# Subintervalos para métodos compuestos
n_rect = 4            # rectángulo medio
n_trap = 4            # trapecio compuesto
n_simp13_comp = 4     # debe ser PAR
n_simp38_comp = 6     # debe ser múltiplo de 3

# Si conocés la integral exacta, ponela acá.
# Si NO la conocés, dejá None y el script usa una referencia numérica fina.
valor_exacto = None

# Si conocés cotas para derivadas, ponelas acá.
# Sirven para errores teóricos de truncamiento.

M2, M4 = obtener_M2_M4(expr, x, a, b)

# Mostrar gráficos
MOSTRAR_GRAFICOS = True


# =========================================================
# UTILIDADES
# =========================================================

def evaluar_lista(func, xs):
    return np.array([func(x) for x in xs], dtype=float)


def imprimir_tabla(titulo, encabezados, filas, anchos=None, decimales=8):
    print("\n" + "=" * 120)
    print(titulo)
    print("=" * 120)

    if anchos is None:
        anchos = [18] * len(encabezados)
    elif len(anchos) < len(encabezados):
        anchos = anchos + [18] * (len(encabezados) - len(anchos))
    elif len(anchos) > len(encabezados):
        anchos = anchos[:len(encabezados)]

    encabezado_txt = "".join(
        f"{encabezados[i]:<{anchos[i]}}" for i in range(len(encabezados)))
    print(encabezado_txt)
    print("-" * sum(anchos))

    for fila in filas:
        linea = ""
        for i, valor in enumerate(fila):
            if isinstance(valor, (float, np.floating)):
                linea += f"{valor:<{anchos[i]}.{decimales}f}"
            else:
                linea += f"{str(valor):<{anchos[i]}}"
        print(linea)


def referencia_numerica(func, a, b, n=4000):
    if n % 2 != 0:
        n += 1
    return simpson_13_compuesta(func, a, b, n)["area"]


def error_real(area, referencia):
    return abs(referencia - area)


# =========================================================
# RECTÁNGULO MEDIO COMPUESTO
# =========================================================

def rectangulo_medio(func, a, b, n):
    h = (b - a) / n
    x_i = np.array([a + i * h for i in range(n)], dtype=float)
    x_medios = x_i + h / 2
    f_medios = evaluar_lista(func, x_medios)
    area = h * np.sum(f_medios)

    filas = []
    for i in range(n):
        filas.append([i, x_i[i], x_medios[i], f_medios[i]])

    error_teorico = None
    if M2 is not None:
        error_teorico = ((b - a)**3 / (24 * n**2)) * M2

    return {
        "metodo": f"Rectángulo medio (n={n})",
        "h": h,
        "area": area,
        "tabla": filas,
        "encabezados": ["i", "x_i", "x̄_i", "f(x̄_i)"],
        "xs": x_i,
        "xmed": x_medios,
        "fmed": f_medios,
        "error_teorico": error_teorico,
        "restriccion": "ninguna"
    }


# =========================================================
# TRAPECIO SIMPLE
# =========================================================

def trapecio_simple(func, a, b):
    h = b - a
    fa = func(a)
    fb = func(b)
    area = h * (fa + fb) / 2

    filas = [
        [0, a, fa],
        [1, b, fb]
    ]

    error_teorico = None
    if M2 is not None:
        error_teorico = ((b - a)**3 / 12) * M2

    return {
        "metodo": "Trapecio simple",
        "h": h,
        "area": area,
        "tabla": filas,
        "encabezados": ["i", "x_i", "f(x_i)"],
        "xs": np.array([a, b], dtype=float),
        "ys": np.array([fa, fb], dtype=float),
        "error_teorico": error_teorico,
        "restriccion": "ninguna"
    }


# =========================================================
# TRAPECIO COMPUESTO
# =========================================================

def trapecio_compuesto(func, a, b, n):
    h = (b - a) / n
    xs = np.linspace(a, b, n + 1)
    ys = evaluar_lista(func, xs)

    area = (h / 2) * (ys[0] + 2 * np.sum(ys[1:-1]) + ys[-1])

    filas = []
    for i in range(len(xs)):
        coef = 1 if (i == 0 or i == len(xs) - 1) else 2
        filas.append([i, xs[i], ys[i], coef])

    error_teorico = None
    if M2 is not None:
        error_teorico = abs(-(((b - a) ** 3) / (12 * n**2)) * M2)

    return {
        "metodo": f"Trapecio compuesto (n={n})",
        "h": h,
        "area": area,
        "tabla": filas,
        "encabezados": ["i", "x_i", "f(x_i)", "coef"],
        "xs": xs,
        "ys": ys,
        "error_teorico": error_teorico,
        "restriccion": "ninguna"
    }


# =========================================================
# SIMPSON 1/3 SIMPLE
# =========================================================

def simpson_13_simple(func, a, b):
    h = (b - a) / 2
    x0 = a
    x1 = (a + b) / 2
    x2 = b

    f0 = func(x0)
    f1 = func(x1)
    f2 = func(x2)

    area = (h / 3) * (f0 + 4 * f1 + f2)

    filas = [
        [0, x0, f0, 1],
        [1, x1, f1, 4],
        [2, x2, f2, 1]
    ]

    error_teorico = None
    if M4 is not None:
        error_teorico = ((b - a)**5 / 2880) * M4

    return {
        "metodo": "Simpson 1/3 simple",
        "h": h,
        "area": area,
        "tabla": filas,
        "encabezados": ["i", "x_i", "f(x_i)", "coef"],
        "xs": np.array([x0, x1, x2], dtype=float),
        "ys": np.array([f0, f1, f2], dtype=float),
        "error_teorico": error_teorico,
        "restriccion": "ninguna"
    }


# =========================================================
# SIMPSON 1/3 COMPUESTA
# =========================================================

def simpson_13_compuesta(func, a, b, n):
    if n % 2 != 0:
        raise ValueError("Para Simpson 1/3 compuesta, n debe ser PAR.")

    h = (b - a) / n
    xs = np.linspace(a, b, n + 1)
    ys = evaluar_lista(func, xs)

    suma_impares = np.sum(ys[1:-1:2])
    suma_pares = np.sum(ys[2:-1:2])

    area = (h / 3) * (ys[0] + 4 * suma_impares + 2 * suma_pares + ys[-1])

    filas = []
    for i in range(len(xs)):
        if i == 0 or i == len(xs) - 1:
            coef = 1
        elif i % 2 == 1:
            coef = 4
        else:
            coef = 2
        filas.append([i, xs[i], ys[i], coef])

    error_teorico = None
    if M4 is not None:
        error_teorico = abs(-(((b - a) ** 5) / (180 * n**4)) * M4)

    return {
        "metodo": f"Simpson 1/3 compuesta (n={n})",
        "h": h,
        "area": area,
        "tabla": filas,
        "encabezados": ["i", "x_i", "f(x_i)", "coef"],
        "xs": xs,
        "ys": ys,
        "error_teorico": error_teorico,
        "restriccion": "n par"
    }


# =========================================================
# SIMPSON 3/8 SIMPLE
# =========================================================

def simpson_38_simple(func, a, b):
    h = (b - a) / 3
    x0 = a
    x1 = a + h
    x2 = a + 2 * h
    x3 = b

    f0 = func(x0)
    f1 = func(x1)
    f2 = func(x2)
    f3 = func(x3)

    area = (3 * h / 8) * (f0 + 3 * f1 + 3 * f2 + f3)

    filas = [
        [0, x0, f0, 1],
        [1, x1, f1, 3],
        [2, x2, f2, 3],
        [3, x3, f3, 1]
    ]

    error_teorico = None
    if M4 is not None:
        error_teorico = abs(-(3 * h**5 / 80) * M4)

    return {
        "metodo": "Simpson 3/8 simple",
        "h": h,
        "area": area,
        "tabla": filas,
        "encabezados": ["i", "x_i", "f(x_i)", "coef"],
        "xs": np.array([x0, x1, x2, x3], dtype=float),
        "ys": np.array([f0, f1, f2, f3], dtype=float),
        "error_teorico": error_teorico,
        "restriccion": "ninguna"
    }


# =========================================================
# SIMPSON 3/8 COMPUESTA
# =========================================================
def simpson_38_compuesta(func, a, b, n):
    if n % 3 != 0:
        raise ValueError(
            "Para Simpson 3/8 compuesta, n debe ser múltiplo de 3.")

    h = (b - a) / n
    xs = np.linspace(a, b, n + 1)
    ys = evaluar_lista(func, xs)

    suma_3 = np.sum([ys[i] for i in range(1, n) if i % 3 != 0])
    suma_2 = np.sum([ys[i] for i in range(3, n, 3)])

    area = (3*h/8) * (ys[0] + 3*suma_3 + 2*suma_2 + ys[-1])

    filas = []
    for i in range(len(xs)):
        if i == 0 or i == n:
            coef = 1
        elif i % 3 == 0:
            coef = 2
        else:
            coef = 3
        filas.append([i, xs[i], ys[i], coef])

    error_teorico = None
    if M4 is not None:
        error_teorico = abs(-((b - a)**5 / (80 * n**4)) * M4)

    return {
        "metodo": f"Simpson 3/8 compuesta (n={n})",
        "h": h,
        "area": area,
        "tabla": filas,
        "encabezados": ["i", "x_i", "f(x_i)", "coef"],
        "xs": xs,
        "ys": ys,
        "error_teorico": error_teorico,
        "restriccion": "n múltiplo de 3"
    }

# =========================================================
# GRÁFICOS
# =========================================================


def graficar_rectangulo_medio(func, a, b, resultado):
    xs = np.linspace(a, b, 500)
    ys = evaluar_lista(func, xs)

    plt.figure(figsize=(9, 5))
    plt.plot(xs, ys, linewidth=2, label="f(x)")

    h = resultado["h"]
    for xi, xbar, ybar in zip(resultado["xs"], resultado["xmed"], resultado["fmed"]):
        plt.plot([xi, xi, xi + h, xi + h, xi],
                 [0, ybar, ybar, 0, 0], linewidth=1.2)

    plt.scatter(resultado["xmed"], resultado["fmed"], label="Puntos medios")
    plt.title(resultado["metodo"])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.show()


def graficar_nodos(func, a, b, resultado):
    xs = np.linspace(a, b, 500)
    ys = evaluar_lista(func, xs)

    plt.figure(figsize=(9, 5))
    plt.plot(xs, ys, linewidth=2, label="f(x)")
    plt.plot(resultado["xs"], resultado["ys"], marker="o", label="Nodos")
    plt.title(resultado["metodo"])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.show()


def graficar_ejemplo_aplicado(func, a, b, area, titulo, puntos_muestra=2000):
    xs = np.linspace(a, b, 500)
    ys = evaluar_lista(func, xs)

    plt.figure(figsize=(9, 5))
    plt.plot(xs, ys, linewidth=2, label="f(x)")
    plt.fill_between(xs, ys, alpha=0.25, label=f"Área aproximada = {area:.6f}")

    puntos = np.linspace(a, b, min(puntos_muestra, 60))
    plt.scatter(puntos, evaluar_lista(func, puntos), s=18,
                alpha=0.6, color="orange", label="Muestras")

    plt.title(titulo)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    plt.show()


def ejecutar_ejemplos_aplicados():
    ejemplos = [
        {
            "area": "Física",
            "descripcion": "Trabajo realizado por una fuerza variable F(x) = 5 e^(-0.5x)",
            "funcion": lambda x: 5 * np.exp(-0.5 * x),
            "intervalo": (0, 4),
            "metodo": simpson_13_compuesta,
            "n": 10,
        },
        {
            "area": "Ingeniería",
            "descripcion": "Carga térmica q(x) = 20/(1+x^2)",
            "funcion": lambda x: 20 / (1 + x**2),
            "intervalo": (0, 3),
            "metodo": simpson_38_compuesta,
            "n": 12,
        },
        {
            "area": "Economía",
            "descripcion": "Costo acumulado C(x) = 30 + 8x - 0.5x^2",
            "funcion": lambda x: 30 + 8 * x - 0.5 * x**2,
            "intervalo": (0, 5),
            "metodo": trapecio_compuesto,
            "n": 10,
        },
        {
            "area": "Ciencia de datos",
            "descripcion": "Error medio cuadrático aproximado en [0,1]",
            "funcion": lambda x: (x - 0.7)**2 + 0.1,
            "intervalo": (0, 1),
            "metodo": simpson_13_simple,
            "n": None,
        },
        {
            "area": "Redes neuronales",
            "descripcion": "Activación promedio de una capa: tanh(x) + 1",
            "funcion": lambda x: np.tanh(x) + 1,
            "intervalo": (0, 2),
            "metodo": trapecio_simple,
            "n": None,
        },
        {
            "area": "Optimización en machine learning",
            "descripcion": "Loss de validación L(x) = (x-0.03)^2 + 0.02",
            "funcion": lambda x: (x - 0.03)**2 + 0.02,
            "intervalo": (0, 0.1),
            "metodo": simpson_13_compuesta,
            "n": 8,
        },
    ]

    filas = []

    for i, ejemplo in enumerate(ejemplos, start=1):
        a_ej, b_ej = ejemplo["intervalo"]
        metodo = ejemplo["metodo"]

        if ejemplo["n"] is None:
            resultado = metodo(ejemplo["funcion"], a_ej, b_ej)
        else:
            resultado = metodo(ejemplo["funcion"], a_ej, b_ej, ejemplo["n"])

        filas.append([
            ejemplo["area"],
            ejemplo["descripcion"],
            resultado["area"],
            resultado["metodo"],
        ])

        if MOSTRAR_GRAFICOS:
            graficar_ejemplo_aplicado(
                ejemplo["funcion"],
                a_ej,
                b_ej,
                resultado["area"],
                f"{ejemplo['area']}, {ejemplo['descripcion']} - Ejemplo aplicado con integración",
            )

    imprimir_tabla(
        "EJEMPLOS APLICADOS POR AREA",
        ["Area", "Descripcion", "Integral aprox.", "Metodo usado"],
        filas,
        anchos=[20, 45, 18, 28],
        decimales=8,
    )


# =========================================================
# EJECUCIÓN
# =========================================================

if valor_exacto is None:
    referencia = referencia_numerica(f, a, b, n=4000)
    referencia_txt = "Referencia numérica fina"
else:
    referencia = valor_exacto
    referencia_txt = "Valor exacto"

resultados = []

r_rect = rectangulo_medio(f, a, b, n_rect)
resultados.append(r_rect)

r_ts = trapecio_simple(f, a, b)
resultados.append(r_ts)

r_tc = trapecio_compuesto(f, a, b, n_trap)
resultados.append(r_tc)

r_s13s = simpson_13_simple(f, a, b)
resultados.append(r_s13s)

r_s13c = simpson_13_compuesta(f, a, b, n_simp13_comp)
resultados.append(r_s13c)

r_s38 = simpson_38_simple(f, a, b)
resultados.append(r_s38)

r_s38c = simpson_38_compuesta(f, a, b, n_simp38_comp)  # n múltiplo de 3
resultados.append(r_s38c)
# =========================================================
# TABLAS DETALLADAS DE CADA MÉTODO
# =========================================================

for r in resultados:
    imprimir_tabla(
        f"DETALLE - {r['metodo']}",
        r["encabezados"],
        r["tabla"],
        anchos=[8, 18, 18, 12] if len(r["encabezados"]) == 4 else [8, 18, 18],
        decimales=10
    )

    filas_resumen = [
        ["h", r["h"]],
        ["Área aproximada", r["area"]],
        ["Error de truncamiento", r["error_teorico"]],
        ["Restricción", r.get("restriccion", "—")]
    ]

    if "error_teorico" in r and r["error_teorico"] is not None:
        filas_resumen.append(["Error teórico (cota)", r["error_teorico"]])

    imprimir_tabla(
        f"RESUMEN - {r['metodo']}",
        ["Concepto", "Valor"],
        filas_resumen,
        anchos=[30, 20],
        decimales=10
    )

# =========================================================
# TABLA COMPARATIVA FINAL
# =========================================================

filas_comp = []
for r in resultados:
    err_teo = r["error_teorico"] if (
        "error_teorico" in r and r["error_teorico"] is not None) else "—"
    filas_comp.append([
        r["metodo"],
        r["area"],
        r["error_teorico"],
        r.get("restriccion", "—")
    ])

imprimir_tabla(
    f"TABLA COMPARATIVA FINAL ({referencia_txt})",
    ["Método", "Área aprox", "Error truncamiento", "Restricción"],
    filas_comp,
    anchos=[35, 20, 20, 20, 20],
    decimales=10
)

print("\n" + "=" * 120)
print("RESULTADO GLOBAL")
print("=" * 120)
print(f"Referencia usada = {referencia:.10f} ({referencia_txt})")

# =========================================================
# GRÁFICOS
# =========================================================

if MOSTRAR_GRAFICOS:
    graficar_rectangulo_medio(f, a, b, r_rect)
    graficar_nodos(f, a, b, r_ts)
    graficar_nodos(f, a, b, r_tc)
    graficar_nodos(f, a, b, r_s13s)
    graficar_nodos(f, a, b, r_s13c)
    graficar_nodos(f, a, b, r_s38)
    graficar_nodos(f, a, b, r_s38c)

ejecutar_ejemplos_aplicados()
