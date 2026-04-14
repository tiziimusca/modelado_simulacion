import matplotlib.pyplot as plt
import numpy as np


def newton_raphson(f, x0, tol=1e-8, max_iter=50, graficar=True, titulo_grafico="Progreso Newton-Raphson"):

    print("Método de Newton-Raphson")
    print(f"x0 = {x0}")
    print("-"*50)

    historial = [x0]

    for i in range(max_iter):

        fx = f(x0)
        dfx = derivada(f, x0)

        print(f"\nIteración {i+1}")
        print(f"x = {x0}")
        print(f"f(x) = {fx}")
        print(f"f'(x) = {dfx}")

        if abs(dfx) < 1e-12:
            print("❌ Derivada cercana a cero. El método falla.")
            return None

        x1 = x0 - fx/dfx
        error = abs(x1 - x0)

        print(f"x nuevo = {x1}")
        print(f"error = {error}")

        historial.append(x1)

        if error < tol:
            print("\n✅ Convergencia alcanzada")
            print(f"Raíz ≈ {x1}")
            print(f"Iteraciones = {i+1}")

            if graficar:
                graficar_newton(f, historial, titulo_grafico)

            return x1

        x0 = x1

    print("\n⚠️ Se alcanzó el máximo de iteraciones")

    if graficar:
        graficar_newton(f, historial, titulo_grafico)

    return x0


def graficar_newton(f, historial, titulo_grafico):

    xmin = min(historial) - 1
    xmax = max(historial) + 1

    x = np.linspace(xmin, xmax, 400)
    y = f(x)

    plt.figure(figsize=(8, 5))
    plt.axhline(0)
    plt.plot(x, y, label="f(x)")

    for i, xi in enumerate(historial):
        plt.scatter(xi, f(xi))
        plt.text(xi, f(xi), str(i))

    plt.title(titulo_grafico)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.show()


def derivada(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)


def g(x):
    # return (x + 2)**(1/3)
    # return x**3 - x - 2
    # return np.cos(x) + x
    # return np.exp(x) - 2 - x
    # return x**2 - 2
    # return np.sin(x) - 0.5
    # return 0.4*np.exp(x**2)
    # return (x-1)**2
    return np.sin(x)/x


newton_raphson(g, 2.5, tol=1e-8, max_iter=50, graficar=True,
               titulo_grafico="Ejemplo de Newton-Raphson")


def ejecutar_ejemplos_aplicados():
    ejemplos = [
        {
            "area": "Física",
            "descripcion": "Equilibrio oscilatorio: cos(x) - x = 0",
            "f": lambda x: np.cos(x) - x,
            "x0": 0.7,
        },
        {
            "area": "Ingeniería",
            "descripcion": "Transferencia térmica: e^(-x) - x = 0",
            "f": lambda x: np.exp(-x) - x,
            "x0": 0.5,
        },
        {
            "area": "Economía",
            "descripcion": "Equilibrio de mercado: x^2 - 3 = 0",
            "f": lambda x: x**2 - 3,
            "x0": 1.8,
        },
        {
            "area": "Ciencia de datos",
            "descripcion": "Umbral de probabilidad: 1/(1+e^-x) - 0.8 = 0",
            "f": lambda x: 1 / (1 + np.exp(-x)) - 0.8,
            "x0": 1.0,
        },
        {
            "area": "Redes neuronales",
            "descripcion": "Punto fijo de activación: tanh(x) - 0.5 = 0",
            "f": lambda x: np.tanh(x) - 0.5,
            "x0": 0.6,
        },
        {
            "area": "Optimización en machine learning",
            "descripcion": "Óptimo de loss cuadrática: 2(x-0.03) = 0",
            "f": lambda x: 2 * (x - 0.03),
            "x0": 0.2,
        },
    ]

    for i, ejemplo in enumerate(ejemplos, start=1):
        print("\n" + "=" * 72)
        print(f"Ejemplo {i} - {ejemplo['area']}")
        print(ejemplo["descripcion"])
        print("=" * 72)

        newton_raphson(
            ejemplo["f"],
            ejemplo["x0"],
            tol=1e-8,
            max_iter=50,
            graficar=True,
            titulo_grafico=f"{ejemplo['area']}, {ejemplo['descripcion']} - Newton-Raphson",
        )


if __name__ == "__main__":
    ejecutar_ejemplos_aplicados()
