import matplotlib.pyplot as plt
import numpy as np


def aitken_punto_fijo(g, x0, tol=1e-8, max_iter=50, graficar=True, titulo_grafico="Convergencia con Aitken"):

    print("Método de Punto Fijo con Aceleración de Aitken")
    print(f"x0 = {x0}")
    print("-"*50)

    historial = [x0]

    for i in range(max_iter):

        x1 = g(x0)
        x2 = g(x1)

        print(f"\nIteración {i+1}")
        print(f"x0 = {x0}")
        print(f"x1 = {x1}")
        print(f"x2 = {x2}")

        denominador = x2 - 2*x1 + x0

        if abs(denominador) < 1e-14:
            print("⚠️ Denominador cercano a cero, no se puede aplicar Aitken")
            return None

        x_aitken = x0 - ((x1 - x0)**2) / denominador

        error = abs(x_aitken - x0)

        print(f"x_aitken = {x_aitken}")
        print(f"error = {error}")

        historial.append(x_aitken)

        if error < tol:
            print("\n✅ Convergencia alcanzada")
            print(f"Solución ≈ {x_aitken}")
            print(f"Iteraciones = {i+1}")

            if graficar:
                graficar_convergencia(historial, titulo_grafico)

            return x_aitken

        x0 = x_aitken

    print("\n⚠️ Se alcanzó el máximo de iteraciones")

    if graficar:
        graficar_convergencia(historial, titulo_grafico)

    return x0


def graficar_convergencia(historial, titulo_grafico):

    xs = range(len(historial))

    plt.figure(figsize=(7, 4))
    plt.plot(xs, historial, marker='o')
    plt.title(titulo_grafico)
    plt.xlabel("Iteración")
    plt.ylabel("x")
    plt.grid()
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
    return np.e**-x


aitken_punto_fijo(g, 1, tol=1e-8, max_iter=50, graficar=True,
                  titulo_grafico="Ejemplo de Aitken - g(x) = e^(-x)")


def derivada(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)


d = derivada(g, 1.4)

print("f'(x) ≈", d)


def ejecutar_ejemplos_aplicados():
    ejemplos = [
        {
            "area": "Física",
            "descripcion": "Relajación térmica: x = e^(-x)",
            "g": lambda x: np.exp(-x),
            "x0": 1.0,
        },
        {
            "area": "Ingeniería",
            "descripcion": "Respuesta de control: x = 0.4 + 0.5e^(-x)",
            "g": lambda x: 0.4 + 0.5 * np.exp(-x),
            "x0": 0.6,
        },
        {
            "area": "Economía",
            "descripcion": "Ajuste de equilibrio: x = 1.2 - 0.3 ln(1 + x)",
            "g": lambda x: 1.2 - 0.3 * np.log(1 + x),
            "x0": 0.5,
        },
        {
            "area": "Ciencia de datos",
            "descripcion": "Normalización iterativa: x = 0.8 / (1 + e^(-x))",
            "g": lambda x: 0.8 / (1 + np.exp(-x)),
            "x0": 0.3,
        },
        {
            "area": "Redes neuronales",
            "descripcion": "Activación estable: x = tanh(0.7x + 0.2)",
            "g": lambda x: np.tanh(0.7 * x + 0.2),
            "x0": 0.2,
        },
        {
            "area": "Optimización en machine learning",
            "descripcion": "Learning rate estable: x = 0.03 + 0.2(x-0.03)^2",
            "g": lambda x: 0.03 + 0.2 * (x - 0.03) ** 2,
            "x0": 0.1,
        },
    ]

    for i, ejemplo in enumerate(ejemplos, start=1):
        print("\n" + "=" * 72)
        print(f"Ejemplo {i} - {ejemplo['area']}")
        print(ejemplo["descripcion"])
        print("=" * 72)

        aitken_punto_fijo(
            ejemplo["g"],
            ejemplo["x0"],
            tol=1e-8,
            max_iter=50,
            graficar=True,
            titulo_grafico=f"{ejemplo['area']}, {ejemplo['descripcion']} - Convergencia con Aitken",
        )


if __name__ == "__main__":
    ejecutar_ejemplos_aplicados()
