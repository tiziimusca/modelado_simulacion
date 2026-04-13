import matplotlib.pyplot as plt
import numpy as np


def punto_fijo(g, x0, tolerancia=1e-6, max_iter=50, mostrar_grafico=True, titulo_grafico="Método de Punto Fijo"):

    print("Inicio del método de punto fijo")
    print(f"x0 = {x0}")
    print("-" * 50)

    historial = [x0]

    for i in range(1, max_iter + 1):

        x1 = g(x0)
        error = abs(x1 - x0)

        historial.append(x1)

        print(f"Iteración {i}")
        print(f"x{i} = {x1:.8f}")
        print(f"error = {error:.8f}")

        if error < tolerancia:
            print("\n✅ Convergencia alcanzada")
            print(f"Punto fijo aproximado: {x1}")
            print(f"Iteraciones: {i}")

            if mostrar_grafico:
                graficar_punto_fijo(g, historial, titulo_grafico)

            return x1

        x0 = x1
        print()

    print("⚠️ Se alcanzó el máximo de iteraciones")
    print(f"Aproximación final: {x1}")

    if mostrar_grafico:
        graficar_punto_fijo(g, historial, titulo_grafico)

    return x1


def graficar_punto_fijo(g, historial, titulo_grafico):

    xmin = min(historial) - 1
    xmax = max(historial) + 1

    x = np.linspace(xmin, xmax, 400)

    plt.figure(figsize=(8, 5))

    plt.plot(x, g(x), label="g(x)")
    plt.plot(x, x, label="y = x")

    for i in range(len(historial)-1):
        x0 = historial[i]
        x1 = historial[i+1]

        # paso vertical
        plt.plot([x0, x0], [x0, x1])

        # paso horizontal
        plt.plot([x0, x1], [x1, x1])

        plt.scatter(x1, x1)
        plt.text(x1, x1, str(i+1))

    plt.title(titulo_grafico)
    plt.xlabel("x")
    plt.ylabel("y")

    plt.legend()
    plt.show()


def g(x):
    # return (x + 2)**(1/3)
    # return x**3 - x - 2
    # return np.cos(x) + x
    # return np.exp(x) - 2 - x
    # return x**2 - 2
    # return np.sin(x) - 0.5
    # return 0.4*np.exp(x**2)
    # return np.exp(-x)
    # return x**2-3
    # return np.pi + 0.5*np.sin(x/2)
    # return 1-0.5 * x
    # return 2/np.pi + 4/(np.pi*x)
    # return np.e**-x
    # return np.log(x)-1
    return 2/np.pi + 4/(np.pi*x)


punto_fijo(g, 0.5, tolerancia=1e-6, max_iter=50,
           mostrar_grafico=True, titulo_grafico="Ejemplo de Punto Fijo")


def derivada(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)


d = derivada(g, 1.4)

print("f'(x) ≈", d)


def ejecutar_ejemplos_aplicados():
    ejemplos = [
        {
            "area": "Física",
            "descripcion": "Equilibrio de un resorte no lineal: x = 1.2 / (1 + x^2)",
            "g": lambda x: 1.2 / (1 + x**2),
            "x0": 0.5,
        },
        {
            "area": "Ingeniería",
            "descripcion": "Enfriamiento con retroalimentación: x = 0.35 + 0.55 * e^(-x)",
            "g": lambda x: 0.35 + 0.55 * np.exp(-x),
            "x0": 0.8,
        },
        {
            "area": "Economía",
            "descripcion": "Precio de equilibrio con ajuste de demanda: x = 1.5 - 0.4 ln(1 + x)",
            "g": lambda x: 1.5 - 0.4 * np.log(1 + x),
            "x0": 0.6,
        },
        {
            "area": "Ciencia de datos",
            "descripcion": "Umbral de normalización: x = 0.8 / (1 + e^(-2x))",
            "g": lambda x: 0.8 / (1 + np.exp(-2 * x)),
            "x0": 0.3,
        },
        {
            "area": "Redes neuronales",
            "descripcion": "Actualización de activación: x = tanh(0.9x + 0.2)",
            "g": lambda x: np.tanh(0.9 * x + 0.2),
            "x0": 0.1,
        },
        {
            "area": "Optimización en machine learning",
            "descripcion": "Ajuste de learning rate: x = 0.03 + 0.5(x - 0.03)^2",
            "g": lambda x: 0.03 + 0.5 * (x - 0.03) ** 2,
            "x0": 0.2,
        },
    ]

    for i, ejemplo in enumerate(ejemplos, start=1):
        print("\n" + "=" * 72)
        print(f"Ejemplo {i} - {ejemplo['area']}")
        print(ejemplo["descripcion"])
        print("=" * 72)

        punto_fijo(
            ejemplo["g"],
            ejemplo["x0"],
            tolerancia=1e-6,
            max_iter=50,
            mostrar_grafico=True,
            titulo_grafico=f"{ejemplo['area']}, {ejemplo['descripcion']} - Metodo de Punto Fijo",
        )


if __name__ == "__main__":
    ejecutar_ejemplos_aplicados()
