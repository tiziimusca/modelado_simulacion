import numpy as np
import matplotlib.pyplot as plt


def resolver_sistema_gauss(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1)

    if A.shape[0] != A.shape[1]:
        raise ValueError("La matriz A debe ser cuadrada.")
    if A.shape[0] != b.shape[0]:
        raise ValueError("A y b deben tener el mismo número de ecuaciones.")

    n = A.shape[0]
    Ab = np.hstack([A, b.reshape(-1, 1)])

    for k in range(n):
        pivote = k + np.argmax(np.abs(Ab[k:, k]))
        if abs(Ab[pivote, k]) < 1e-12:
            raise ValueError(
                "El sistema no tiene solución única o es singular.")

        if pivote != k:
            Ab[[k, pivote]] = Ab[[pivote, k]]

        for i in range(k + 1, n):
            factor = Ab[i, k] / Ab[k, k]
            Ab[i, k:] -= factor * Ab[k, k:]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        suma = np.dot(Ab[i, i + 1:n], x[i + 1:n])
        x[i] = (Ab[i, -1] - suma) / Ab[i, i]

    return x


def formatear_ecuacion_2x2(A, b):
    a1, a2 = A
    return f"{a1:.3f}x + {a2:.3f}y = {b:.3f}"


def graficar_sistema_2x2(A, b, solucion, titulo):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    x_min = min(solucion[0] - 5, -5)
    x_max = max(solucion[0] + 5, 5)
    xs = np.linspace(x_min, x_max, 400)

    plt.figure(figsize=(9, 6))

    for i in range(2):
        a1, a2 = A[i]
        if abs(a2) < 1e-12:
            x_linea = np.full_like(xs, b[i] / a1)
            y_linea = xs
            plt.plot(x_linea, y_linea, linewidth=2, label=f"Ecuación {i+1}")
        else:
            ys = (b[i] - a1 * xs) / a2
            plt.plot(xs, ys, linewidth=2, label=f"Ecuación {i+1}")

    plt.scatter([solucion[0]], [solucion[1]], color="red",
                s=80, zorder=5, label="Solución")
    plt.annotate(
        f"({solucion[0]:.3f}, {solucion[1]:.3f})",
        (solucion[0], solucion[1]),
        textcoords="offset points",
        xytext=(8, 8),
    )

    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title(titulo)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.show()


def ejecutar_ejemplos():
    ejemplos = [
        {
            "area": "Física",
            "descripcion": "Equilibrio de fuerzas en 2D",
            "A": [[2, 1], [1, -1]],
            "b": [10, 2],
        },
        {
            "area": "Ingeniería",
            "descripcion": "Corrientes en un circuito resistivo",
            "A": [[3, 2], [1, -2]],
            "b": [18, 0],
        },
        {
            "area": "Economía",
            "descripcion": "Equilibrio oferta-demanda",
            "A": [[2, -1], [1, 1]],
            "b": [8, 12],
        },
        {
            "area": "Ciencia de datos",
            "descripcion": "Ajuste lineal con 2 parámetros",
            "A": [[5, 2], [2, 3]],
            "b": [17, 14],
        },
        {
            "area": "Redes neuronales",
            "descripcion": "Solución de pesos para una capa lineal",
            "A": [[4, 1], [2, 5]],
            "b": [11, 13],
        },
        {
            "area": "Optimización en machine learning",
            "descripcion": "Paso de optimización en dos variables",
            "A": [[6, -1], [1, 2]],
            "b": [5, 8],
        },
    ]

    filas = []

    for ejemplo in ejemplos:
        A = ejemplo["A"]
        b = ejemplo["b"]
        solucion = resolver_sistema_gauss(A, b)

        print("\n" + "=" * 72)
        print(f"{ejemplo['area']}")
        print(ejemplo["descripcion"])
        print("=" * 72)
        print("Sistema:")
        print(f"  {formatear_ecuacion_2x2(A[0], b[0])}")
        print(f"  {formatear_ecuacion_2x2(A[1], b[1])}")
        print(f"Solución: x = {solucion[0]:.8f}, y = {solucion[1]:.8f}")

        filas.append([
            ejemplo["area"],
            solucion[0],
            solucion[1],
        ])

        graficar_sistema_2x2(
            A,
            b,
            solucion,
            titulo=f"{ejemplo['area']}, {ejemplo['descripcion']} - Sistema lineal 2x2",
        )

    print("\n" + "=" * 72)
    print("RESUMEN DE SOLUCIONES")
    print("=" * 72)
    print(f"{'Área':<28}{'x':<18}{'y':<18}")
    print("-" * 64)
    for fila in filas:
        print(f"{fila[0]:<28}{fila[1]:<18.8f}{fila[2]:<18.8f}")


if __name__ == "__main__":
    ejecutar_ejemplos()
