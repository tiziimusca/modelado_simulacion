import numpy as np
import matplotlib.pyplot as plt


def euler(f, t0, y0, tf, h):
    n = int(np.ceil((tf - t0) / h))
    ts = np.zeros(n + 1)
    ys = np.zeros(n + 1)

    ts[0] = t0
    ys[0] = y0

    for i in range(n):
        ts[i + 1] = ts[i] + h
        ys[i + 1] = ys[i] + h * f(ts[i], ys[i])

    return ts, ys


def rk4(f, t0, y0, tf, h):
    n = int(np.ceil((tf - t0) / h))
    ts = np.zeros(n + 1)
    ys = np.zeros(n + 1)

    ts[0] = t0
    ys[0] = y0

    for i in range(n):
        t = ts[i]
        y = ys[i]

        k1 = f(t, y)
        k2 = f(t + h / 2, y + h * k1 / 2)
        k3 = f(t + h / 2, y + h * k2 / 2)
        k4 = f(t + h, y + h * k3)

        ts[i + 1] = t + h
        ys[i + 1] = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return ts, ys


def graficar_solucion(ts, y_euler, y_rk4, titulo, etiqueta_y):
    plt.figure(figsize=(9, 5))
    plt.plot(ts, y_euler, "o-", markersize=3, linewidth=1.3, label="Euler")
    plt.plot(ts, y_rk4, "-", linewidth=2.2, label="RK4")
    plt.title(titulo)
    plt.xlabel("t")
    plt.ylabel(etiqueta_y)
    plt.grid(True)
    plt.legend()
    plt.show()


def imprimir_resumen(filas):
    print("\n" + "=" * 90)
    print("RESUMEN FINAL DE EJEMPLOS (EDO)")
    print("=" * 90)
    print(f"{'Área':<30}{'y(tf) Euler':<20}{'y(tf) RK4':<20}{'Diferencia':<20}")
    print("-" * 90)
    for fila in filas:
        print(f"{fila[0]:<30}{fila[1]:<20.8f}{fila[2]:<20.8f}{fila[3]:<20.8f}")


def ejecutar_ejemplos():
    ejemplos = [
        {
            "area": "Física",
            "descripcion": "Enfriamiento de Newton",
                    "f": lambda t, y: -0.4 * (y - 20),
                    "t0": 0.0,
                    "y0": 90.0,
                    "tf": 10.0,
                    "h": 0.25,
                    "etiqueta_y": "Temperatura",
        },
        {
            "area": "Ingeniería",
            "descripcion": "Carga de capacitor RC",
                    "f": lambda t, y: (5 - y) / 2,
                    "t0": 0.0,
                    "y0": 0.0,
                    "tf": 10.0,
                    "h": 0.2,
                    "etiqueta_y": "Voltaje",
        },
        {
            "area": "Economía",
            "descripcion": "Ajuste de precio al equilibrio",
                    "f": lambda t, y: 0.6 * (50 - y),
                    "t0": 0.0,
                    "y0": 20.0,
                    "tf": 8.0,
                    "h": 0.2,
                    "etiqueta_y": "Precio",
        },
        {
            "area": "Ciencia de datos",
            "descripcion": "Pérdida de entrenamiento decreciente",
                    "f": lambda t, y: -0.9 * y + 0.05,
                    "t0": 0.0,
                    "y0": 2.0,
                    "tf": 6.0,
                    "h": 0.15,
                    "etiqueta_y": "Loss",
        },
        {
            "area": "Redes neuronales",
            "descripcion": "Dinámica de activación",
                    "f": lambda t, y: -y + np.tanh(1.5 * np.sin(t)),
                    "t0": 0.0,
                    "y0": 0.1,
                    "tf": 10.0,
                    "h": 0.1,
                    "etiqueta_y": "Activación",
        },
        {
            "area": "Optimización en machine learning",
            "descripcion": "Descenso continuo en una loss cuadrática",
                    "f": lambda t, y: -2.0 * (y - 0.03),
                    "t0": 0.0,
                    "y0": 0.6,
                    "tf": 5.0,
                    "h": 0.1,
                    "etiqueta_y": "Parámetro",
        },
    ]

    filas_resumen = []

    for i, ej in enumerate(ejemplos, start=1):
        print("\n" + "=" * 90)
        print(f"Ejemplo {i} - {ej['area']}")
        print(ej["descripcion"])
        print("=" * 90)

        ts, y_euler = euler(ej["f"], ej["t0"], ej["y0"], ej["tf"], ej["h"])
        _, y_rk4 = rk4(ej["f"], ej["t0"], ej["y0"], ej["tf"], ej["h"])

        print(f"y(tf) Euler = {y_euler[-1]:.8f}")
        print(f"y(tf) RK4   = {y_rk4[-1]:.8f}")
        print(f"|Euler - RK4| = {abs(y_euler[-1] - y_rk4[-1]):.8f}")

        filas_resumen.append([
            ej["area"],
            y_euler[-1],
            y_rk4[-1],
            abs(y_euler[-1] - y_rk4[-1]),
        ])

        graficar_solucion(
            ts,
            y_euler,
            y_rk4,
            titulo=f"{ej['area']} - {ej['descripcion']}",
            etiqueta_y=ej["etiqueta_y"],
        )

    imprimir_resumen(filas_resumen)


if __name__ == "__main__":
    ejecutar_ejemplos()
