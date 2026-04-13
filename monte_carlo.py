import math
import random
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

# --- Configuración para Integración 1D (Función f(x)) ---


def f(x):
    return np.e**(-x**2)


a = 0
b = 1
tamanos_muestra_1d = [10000]  # Para ver convergencia

# Valores Z para los Intervalos de Confianza
Z_90 = 1.645
Z_95 = 1.960
Z_99 = 2.576
Z_997 = 2.968
Z_999 = 3.291

# Mostrar gráficos al final
MOSTRAR_GRAFICOS = True


# --- Configuración para Integración 2D (Función f(x, y)) ---
def g(x, y):
    return np.e**(x + y)


ax = 0
bx = 2
ay = 1
by = 3

tamanos_muestra_2d = [10000]

# =========================================================
# UTILIDADES
# =========================================================


def evaluar_lista(func, xs):
    return np.array([func(x) for x in xs], dtype=float)


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


def referencia_numerica(func, a, b, n=4000):
    h = (b - a) / n
    xs = np.linspace(a, b, n + 1)
    ys = evaluar_lista(func, xs)
    return (h / 2) * (ys[0] + 2 * np.sum(ys[1:-1]) + ys[-1])


# =========================================================
# 2. MONTE CARLO: INTEGRACIÓN 1D + INCERTIDUMBRE
# =========================================================

def montecarlo_1d(func, a, b, n):
    xs = np.random.uniform(a, b, n)
    ys = evaluar_lista(func, xs)

    f_promedio = np.mean(ys)
    volumen = b - a
    area_estimada = volumen * f_promedio

    # Desviación estándar de la muestra (sigma)
    sigma = np.std(ys, ddof=1) if n > 1 else 0
    varianza = sigma**2
    # Error estándar
    ee = sigma / math.sqrt(n) if n > 0 else 0

    # IMPORTANTE: Aquí se escala el error por el ancho del intervalo (b-a)
    ee_area = volumen * ee

    ic_90 = (area_estimada - Z_90 * ee_area, area_estimada + Z_90 * ee_area)
    ic_95 = (area_estimada - Z_95 * ee_area, area_estimada + Z_95 * ee_area)
    ic_99 = (area_estimada - Z_99 * ee_area, area_estimada + Z_99 * ee_area)
    ic_997 = (area_estimada - Z_997 * ee_area, area_estimada + Z_997 * ee_area)
    ic_999 = (area_estimada - Z_999 * ee_area, area_estimada + Z_999 * ee_area)

    return {
        "n": n,
        "area": area_estimada,
        "f_promedio": f_promedio,
        "sigma": sigma,
        "varianza": varianza,
        "ee_area": ee_area,
        "ic_90": ic_90,
        "ic_95": ic_95,
        "ic_99": ic_99,
        "ic_997": ic_997,
        "ic_999": ic_999,
        "xs": xs,
        "ys": ys
    }


def montecarlo_2d(func, ax, bx, ay, by, n):
    xs = np.random.uniform(ax, bx, n)
    ys = np.random.uniform(ay, by, n)
    zs = np.array([func(x, y) for x, y in zip(xs, ys)], dtype=float)

    area = (bx - ax) * (by - ay)
    f_promedio = np.mean(zs)
    integral_estimada = area * f_promedio

    # Desviación estándar
    sigma = np.std(zs, ddof=1) if n > 1 else 0
    varianza = sigma**2
    # Error estándar
    ee = sigma / math.sqrt(n) if n > 0 else 0
    ee_integral = area * ee

    ic_90 = (integral_estimada - Z_90 * ee_integral,
             integral_estimada + Z_90 * ee_integral)

    ic_95 = (integral_estimada - Z_95 * ee_integral,
             integral_estimada + Z_95 * ee_integral)

    ic_99 = (integral_estimada - Z_99 * ee_integral,
             integral_estimada + Z_99 * ee_integral)

    ic_997 = (integral_estimada - Z_997 * ee_integral,
              integral_estimada + Z_997 * ee_integral)

    ic_999 = (integral_estimada - Z_999 * ee_integral,
              integral_estimada + Z_999 * ee_integral)

    return {
        "n": n,
        "integral": integral_estimada,
        "f_promedio": f_promedio,
        "sigma": sigma,
        "varianza": varianza,
        "ee_integral": ee_integral,
        "ic_90": ic_90,
        "ic_95": ic_95,
        "ic_99": ic_99,
        "ic_997": ic_997,
        "ic_999": ic_999,
        "xs": xs,
        "ys": ys,
        "zs": zs
    }

# =========================================================
# GRÁFICOS
# =========================================================


def graficar_integral_1d(func, a, b, resultado):
    xs_plot = np.linspace(a, b, 500)
    ys_plot = evaluar_lista(func, xs_plot)

    plt.figure(figsize=(9, 5))
    plt.fill_between(xs_plot, ys_plot, color="skyblue",
                     alpha=0.3, label="Área Real $\int f(x)$")
    plt.plot(xs_plot, ys_plot, linewidth=2, color="blue", label="f(x)")

    puntos_a_graficar = min(resultado["n"], 1000)
    plt.scatter(resultado["xs"][:puntos_a_graficar], resultado["ys"][:puntos_a_graficar],
                color="orange", s=15, alpha=0.7, label=f"Muestras (Mostrando {puntos_a_graficar})")

    plt.axhline(y=resultado["f_promedio"], color="red", linestyle="--",
                label=f"f(x) prom: {resultado['f_promedio']:.4f}")
    plt.title(f"Montecarlo 1D: Integración $f(x)$ (n={resultado['n']})")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()


def graficar_integral_2d(resultado):
    puntos = min(resultado["n"], 4000)

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(
        resultado["xs"][:puntos],
        resultado["ys"][:puntos],
        c=resultado["zs"][:puntos],
        cmap="viridis",
        s=10,
        alpha=0.7
    )
    plt.colorbar(sc, label="f(x, y)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Monte Carlo 2D (n={resultado['n']})")
    plt.grid(True)


def _ic_normal(media, sigma_muestral, n, z=Z_95):
    ee = sigma_muestral / math.sqrt(n) if n > 0 else 0.0
    return (media - z * ee, media + z * ee)


def ejemplos_montecarlo_aplicados(n=8000, semilla=123):
    rng = np.random.default_rng(semilla)
    filas = []
    detalles = {}

    # 1) Fisica: periodo de pendulo con incertidumbre en longitud y gravedad.
    L = np.clip(rng.normal(1.0, 0.05, n), 0.1, None)
    g_local = np.clip(rng.normal(9.81, 0.03, n), 9.0, None)
    T = 2 * np.pi * np.sqrt(L / g_local)
    media_T = float(np.mean(T))
    sigma_T = float(np.std(T, ddof=1))
    ic_T = _ic_normal(media_T, sigma_T, n)
    filas.append([
        "Fisica",
        "Periodo de pendulo",
        media_T,
        f"[{ic_T[0]:.4f}, {ic_T[1]:.4f}] s"
    ])
    detalles["fisica"] = {
        "T": T,
        "media": media_T,
        "ic": ic_T
    }

    # 2) Ingenieria: riesgo de falla cuando la carga supera la resistencia.
    resistencia = rng.normal(1000.0, 80.0, n)
    carga = rng.normal(900.0, 120.0, n)
    falla = carga > resistencia
    p_falla = float(np.mean(falla))
    sigma_p = math.sqrt(max(p_falla * (1 - p_falla), 0.0))
    ic_p = _ic_normal(p_falla, sigma_p, n)
    filas.append([
        "Ingenieria",
        "Probabilidad de falla",
        p_falla,
        f"[{max(ic_p[0], 0.0):.4f}, {min(ic_p[1], 1.0):.4f}]"
    ])
    detalles["ingenieria"] = {
        "p_falla": p_falla,
        "ic": (max(ic_p[0], 0.0), min(ic_p[1], 1.0))
    }

    # 3) Economia: precio de opcion call europea por Monte Carlo.
    S0, K, r, sigma, tau = 100.0, 105.0, 0.05, 0.20, 1.0
    z = rng.normal(0.0, 1.0, n)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * tau + sigma * math.sqrt(tau) * z)
    payoff = np.maximum(ST - K, 0.0)
    precio_call = float(np.exp(-r * tau) * np.mean(payoff))
    sigma_call = float(np.std(np.exp(-r * tau) * payoff, ddof=1))
    ic_call = _ic_normal(precio_call, sigma_call, n)
    filas.append([
        "Economia",
        "Precio call europea",
        precio_call,
        f"[{ic_call[0]:.4f}, {ic_call[1]:.4f}]"
    ])
    detalles["economia"] = {
        "ST": ST,
        "K": K,
        "precio": precio_call,
        "ic": ic_call
    }

    # 4) Ciencia de datos: bootstrap (MC) para incertidumbre de la media.
    datos = rng.normal(50.0, 10.0, 300)
    medias_boot = np.empty(n)
    for i in range(n):
        muestra = rng.choice(datos, size=datos.size, replace=True)
        medias_boot[i] = np.mean(muestra)
    media_boot = float(np.mean(medias_boot))
    ic_boot = (float(np.percentile(medias_boot, 2.5)),
               float(np.percentile(medias_boot, 97.5)))
    filas.append([
        "Ciencia de datos",
        "Media con bootstrap",
        media_boot,
        f"[{ic_boot[0]:.4f}, {ic_boot[1]:.4f}]"
    ])
    detalles["ciencia_datos"] = {
        "medias_boot": medias_boot,
        "media": media_boot,
        "ic": ic_boot
    }

    # 5) Redes neuronales: Monte Carlo Dropout para incertidumbre predictiva.
    x = 2.0
    w = rng.normal(1.2, 0.08, n)
    b0 = rng.normal(0.3, 0.05, n)
    mascara = rng.binomial(1, 0.8, n)
    pred = (w * x + b0) * mascara / 0.8
    pred_media = float(np.mean(pred))
    pred_sigma = float(np.std(pred, ddof=1))
    ic_pred = _ic_normal(pred_media, pred_sigma, n)
    filas.append([
        "Redes neuronales",
        "Prediccion con dropout",
        pred_media,
        f"[{ic_pred[0]:.4f}, {ic_pred[1]:.4f}]"
    ])
    detalles["redes"] = {
        "pred": pred,
        "media": pred_media,
        "ic": ic_pred
    }

    # 6) Optimizacion en ML: random search (MC) de hiperparametros.
    lr = 10 ** rng.uniform(-4, -1, n)
    wd = 10 ** rng.uniform(-6, -2, n)
    ruido = rng.normal(0.0, 0.01, n)
    val_loss = (lr - 0.03)**2 + 0.5 * (wd - 0.001)**2 + ruido
    idx_mejor = int(np.argmin(val_loss))
    filas.append([
        "Opt. en ML",
        "Mejor val_loss (random search)",
        float(val_loss[idx_mejor]),
        f"lr={lr[idx_mejor]:.5f}, wd={wd[idx_mejor]:.6f}"
    ])
    detalles["opt_ml"] = {
        "lr": lr,
        "wd": wd,
        "val_loss": val_loss,
        "idx_mejor": idx_mejor
    }

    return {"filas": filas, "detalles": detalles}


def graficar_ejemplos_aplicados(ejemplos):
    d = ejemplos["detalles"]
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    axs = axs.ravel()

    axs[0].hist(d["fisica"]["T"], bins=35, color="#1f77b4", alpha=0.8)
    axs[0].axvline(d["fisica"]["media"], color="red",
                   linestyle="--", linewidth=2)
    axs[0].set_title("Fisica: Periodo de pendulo")
    axs[0].set_xlabel("T (s)")
    axs[0].set_ylabel("Frecuencia")

    p_falla = d["ingenieria"]["p_falla"]
    axs[1].bar(["Seguro", "Falla"], [1 - p_falla, p_falla],
               color=["#2ca02c", "#d62728"])
    axs[1].set_ylim(0, 1)
    axs[1].set_title("Ingenieria: Riesgo de falla")
    axs[1].set_ylabel("Probabilidad")

    st = d["economia"]["ST"]
    axs[2].hist(st[:3000], bins=40, color="#ff7f0e", alpha=0.8)
    axs[2].axvline(d["economia"]["K"], color="black",
                   linestyle="--", linewidth=2)
    axs[2].set_title("Economia: Distribucion de S_T")
    axs[2].set_xlabel("Precio final")
    axs[2].set_ylabel("Frecuencia")

    medias_boot = d["ciencia_datos"]["medias_boot"]
    axs[3].hist(medias_boot, bins=35, color="#9467bd", alpha=0.8)
    axs[3].axvline(d["ciencia_datos"]["ic"][0], color="black", linestyle=":")
    axs[3].axvline(d["ciencia_datos"]["ic"][1], color="black", linestyle=":")
    axs[3].set_title("Ciencia de datos: Bootstrap")
    axs[3].set_xlabel("Media estimada")
    axs[3].set_ylabel("Frecuencia")

    axs[4].hist(d["redes"]["pred"], bins=35, color="#17becf", alpha=0.8)
    axs[4].axvline(d["redes"]["media"], color="red",
                   linestyle="--", linewidth=2)
    axs[4].set_title("Redes neuronales: MC Dropout")
    axs[4].set_xlabel("Prediccion")
    axs[4].set_ylabel("Frecuencia")

    lr = d["opt_ml"]["lr"]
    val_loss = d["opt_ml"]["val_loss"]
    idx_mejor = d["opt_ml"]["idx_mejor"]
    axs[5].scatter(lr[:3000], val_loss[:3000], s=8, alpha=0.4, color="#8c564b")
    axs[5].scatter(lr[idx_mejor], val_loss[idx_mejor],
                   s=90, color="red", label="Mejor")
    axs[5].set_xscale("log")
    axs[5].set_title("Optimizacion ML: Random search")
    axs[5].set_xlabel("Learning rate (log)")
    axs[5].set_ylabel("Validation loss")
    axs[5].legend()

    fig.suptitle("Ejemplos aplicados con Monte Carlo", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

# =========================================================
# EJECUCIÓN CENTRALIZADA
# =========================================================


random.seed(0)
np.random.seed(0)

# --- 2. EJECUCIÓN INTEGRAL 1D ---
ref_1d = referencia_numerica(f, a, b)
resultados_1d = [montecarlo_1d(f, a, b, n) for n in tamanos_muestra_1d]
res_1d_final = resultados_1d[-1]

imprimir_tabla(
    "RESUMEN FINAL - INTEGRAL 1D",
    ["Concepto", "Valor"],
    [
        ["Tamaño de muestra (n)", res_1d_final["n"]],
        ["Valor Promedio f(x)", res_1d_final["f_promedio"]],
        ["Área Aproximada (Î)", res_1d_final["area"]],
        ["Referencia Numérica", ref_1d],
        ["Error Real Absoluto", abs(ref_1d - res_1d_final["area"])],
        ["Desviación Estándar (σ)", res_1d_final["sigma"]],
        ["Varianza", res_1d_final["varianza"]],
        ["Error Estándar (EE)", res_1d_final["ee_area"]],
    ],
    anchos=[35, 25],
    decimales=8
)

# --- 3. TABLAS DE ESTADÍSTICA E INCERTIDUMBRE ---
imprimir_tabla(
    "TABLA DE VALORES CRÍTICOS (z)",
    ["Nivel de Confianza", "Valor z"],
    [
        ["90%", Z_90],
        ["95%", Z_95],
        ["99%", Z_99],
        ["99.7%", Z_997],
        ["99.9%", Z_999]
    ],
    anchos=[25, 15],
    decimales=3
)

imprimir_tabla(
    "INTERVALOS DE CONFIANZA",
    ["Nivel de Confianza", "Rango Probable de la Integral"],
    [
        ["90% (z=1.645)",
         f"[{res_1d_final['ic_90'][0]:.6f} , {res_1d_final['ic_90'][1]:.6f}]"],
        ["95% (z=1.960)",
         f"[{res_1d_final['ic_95'][0]:.6f} , {res_1d_final['ic_95'][1]:.6f}]"],
        ["99% (z=2.576)",
         f"[{res_1d_final['ic_99'][0]:.6f} , {res_1d_final['ic_99'][1]:.6f}]"],
        ["99.7% (z=2.968)",
         f"[{res_1d_final['ic_997'][0]:.6f} , {res_1d_final['ic_997'][1]:.6f}]"],
        ["99.9% (z=3.291)",
         f"[{res_1d_final['ic_999'][0]:.6f} , {res_1d_final['ic_999'][1]:.6f}]"]
    ],
    anchos=[25, 40],
    decimales=8
)


# --- EJECUCIÓN INTEGRAL 2D ---
resultados_2d = [montecarlo_2d(g, ax, bx, ay, by, n)
                 for n in tamanos_muestra_2d]
res_2d_final = resultados_2d[-1]

imprimir_tabla(
    "RESUMEN FINAL - INTEGRAL 2D",
    ["Concepto", "Valor"],
    [
        ["Tamaño de muestra (n)", res_2d_final["n"]],
        ["Valor Promedio f(x,y)", res_2d_final["f_promedio"]],
        ["Integral Aproximada (Î)", res_2d_final["integral"]],
        ["Desviación Estándar (σ)", res_2d_final["sigma"]],
        ["Varianza", res_2d_final["varianza"]],
        ["Error Estándar (EE)", res_2d_final["ee_integral"]],
    ],
    anchos=[40, 30],
    decimales=8
)

imprimir_tabla(
    "INTERVALOS DE CONFIANZA - 2D",
    ["Nivel de Confianza", "Rango Probable"],
    [
        ["90%", f"[{res_2d_final['ic_90'][0]:.6f} , {res_2d_final['ic_90'][1]:.6f}]"],
        ["95%", f"[{res_2d_final['ic_95'][0]:.6f} , {res_2d_final['ic_95'][1]:.6f}]"],
        ["99%", f"[{res_2d_final['ic_99'][0]:.6f} , {res_2d_final['ic_99'][1]:.6f}]"],
        ["99.7%",
            f"[{res_2d_final['ic_997'][0]:.6f} , {res_2d_final['ic_997'][1]:.6f}]"],
        ["99.9%",
            f"[{res_2d_final['ic_999'][0]:.6f} , {res_2d_final['ic_999'][1]:.6f}]"]
    ],
    anchos=[25, 45]
)

ejemplos_area = ejemplos_montecarlo_aplicados(n=8000, semilla=2026)

imprimir_tabla(
    "EJEMPLOS PEQUENOS DE MONTE CARLO POR AREA",
    ["Area", "Ejemplo", "Estimacion", "Detalle (IC 95% u optimo)"],
    ejemplos_area["filas"],
    anchos=[22, 35, 20, 35],
    decimales=6
)

# =========================================================
# MOSTRAR GRÁFICOS
# =========================================================

if MOSTRAR_GRAFICOS:
    graficar_integral_1d(f, a, b, res_1d_final)
    plt.show()

if MOSTRAR_GRAFICOS:
    graficar_integral_2d(res_2d_final)
    plt.show()

if MOSTRAR_GRAFICOS:
    graficar_ejemplos_aplicados(ejemplos_area)
    plt.show()
