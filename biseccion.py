import matplotlib.pyplot as plt
import numpy as np


def biseccion(f, a, b, tolerancia=1e-6, max_iter=50, mostrar_grafico=True, titulo_grafico="Progreso del metodo de biseccion"):

    if f(a) * f(b) >= 0:
        print("❌ Error: la función no cambia de signo en el intervalo.")
        print(f"f({a}) = {f(a)}, f({b}) = {f(b)}")
        return None

    print("Inicio del método de bisección")
    print(f"Intervalo inicial: [{a}, {b}]")
    print("-" * 50)

    historial = []

    for i in range(1, max_iter + 1):

        m = (a + b) / 2
        fm = f(m)

        historial.append(m)

        print(f"Iteración {i}")
        print(f"a = {a:.6f}, b = {b:.6f}, m = {m:.6f}")
        print(f"f(m) = {fm:.6f}")

        # condición de convergencia
        if abs(fm) < tolerancia or abs(b - a) < tolerancia:
            print("\n✅ Convergencia alcanzada")
            print(f"Raíz aproximada: {m}")
            print(f"Iteraciones: {i}")

            if mostrar_grafico:
                graficar_progreso(f, a, b, historial, titulo_grafico)

            return m

        if f(a) * fm < 0:
            print("La raíz está en el subintervalo [a, m]\n")
            b = m
        else:
            print("La raíz está en el subintervalo [m, b]\n")
            a = m

    print("⚠️ Se alcanzó el número máximo de iteraciones")
    print(f"Aproximación final: {(a+b)/2}")

    if mostrar_grafico:
        graficar_progreso(f, a, b, historial, titulo_grafico)

    return (a + b) / 2


def graficar_progreso(f, a, b, historial, titulo_grafico):

    x = np.linspace(a-1, b+1, 400)
    y = f(x)

    plt.figure(figsize=(8, 5))
    plt.axhline(0)

    plt.plot(x, y, label="f(x)")

    valores = [f(m) for m in historial]
    plt.scatter(historial, valores)

    for i, m in enumerate(historial):
        plt.text(m, valores[i], str(i+1))

    plt.title(titulo_grafico)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()

    plt.show()


def f(x):
    return x**3 - x - 2
    # return np.cos(x) + x
    # return np.exp(x) - 2 - x
    # return x**2 - 2
    # return np.sin(x) - 0.5


biseccion(f, 1, 2, tolerancia=1e-6, max_iter=50,
          mostrar_grafico=True, titulo_grafico="Ejemplo de bisección")


def ejecutar_ejemplos_area():
    ejemplos = [
        {
            "area": "Física",
            "descripcion": "Velocidad terminal: 0.25*v^2 - 9.81 = 0",
            "funcion": lambda x: 0.25 * x**2 - 9.81,
            "intervalo": (0, 10),
        },
        {
            "area": "Ingeniería",
            "descripcion": "Enfriamiento: 200*e^(-0.1t) - 40 = 0",
            "funcion": lambda x: 200 * np.exp(-0.1 * x) - 40,
            "intervalo": (0, 30),
        },
        {
            "area": "Economía",
            "descripcion": "Equilibrio oferta-demanda: (120-2p)-(20+0.5p)=0",
            "funcion": lambda x: (120 - 2 * x) - (20 + 0.5 * x),
            "intervalo": (0, 80),
        },
        {
            "area": "Ciencia de datos",
            "descripcion": "Umbral sigmoide: 1/(1+e^-x) - 0.8 = 0",
            "funcion": lambda x: 1 / (1 + np.exp(-x)) - 0.8,
            "intervalo": (0, 5),
        },
        {
            "area": "Redes neuronales",
            "descripcion": "Activación tanh: tanh(x) - 0.5 = 0",
            "funcion": lambda x: np.tanh(x) - 0.5,
            "intervalo": (0, 2),
        },
        {
            "area": "Optimización en machine learning",
            "descripcion": "Gradiente nulo: 2*(lr-0.03)=0",
            "funcion": lambda x: 2 * (x - 0.03),
            "intervalo": (0.001, 0.1),
        },
    ]

    for i, ejemplo in enumerate(ejemplos, start=1):
        print("\n" + "=" * 70)
        print(f"Ejemplo {i} - {ejemplo['area']}")
        print(ejemplo["descripcion"])
        print("=" * 70)

        a, b = ejemplo["intervalo"]
        raiz = biseccion(
            ejemplo["funcion"],
            a,
            b,
            tolerancia=1e-6,
            max_iter=60,
            mostrar_grafico=True,
            titulo_grafico=f"{ejemplo['area']}, {ejemplo['descripcion']} - Metodo de biseccion",
        )

        if raiz is not None:
            print(f"Raíz aproximada ({ejemplo['area']}): {raiz:.8f}")


ejecutar_ejemplos_area()
