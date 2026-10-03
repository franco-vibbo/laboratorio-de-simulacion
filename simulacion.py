"""Motor de simulación Monte Carlo para el laboratorio de Econometría.

Cada módulo genera `reps` muestras de tamaño `n`, estima por MCO y devuelve
los resultados de todas las réplicas (vectorizado con NumPy, sin bucles).
Para sumar un módulo nuevo: agregar una función que devuelva el mismo formato.
"""
import numpy as np

BETA0, BETA1 = 1.0, 2.0   # valores verdaderos, conocidos en la simulación
Z95 = 1.96


def _mco_simple(x, y):
    """MCO de y sobre x, fila por fila (cada fila es una réplica)."""
    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    sxx = (xc ** 2).sum(axis=1)
    b1 = (xc * yc).sum(axis=1) / sxx
    b0 = y.mean(axis=1) - b1 * x.mean(axis=1)
    e = y - b0[:, None] - b1[:, None] * x
    return b1, b0, e, sxx, xc


def simular_heterocedasticidad(n, reps, gamma=0.0, sigma=0.6, semilla=11):
    """y = b0 + b1*x + u, con u = sigma * exp(gamma*x) * eps.

    gamma = 0  -> homocedasticidad (caso base).
    gamma > 0  -> la varianza del error crece con x (forma de embudo).
    """
    rng = np.random.default_rng(semilla)
    x = rng.standard_normal((reps, n))
    eps = rng.standard_normal((reps, n))
    y = BETA0 + BETA1 * x + sigma * np.exp(gamma * x) * eps

    b1, b0, e, sxx, xc = _mco_simple(x, y)
    se_clasico = np.sqrt((e ** 2).sum(axis=1) / (n - 2) / sxx)
    se_robusto = np.sqrt((xc ** 2 * e ** 2).sum(axis=1) * n / (n - 2)) / sxx  # HC1
    return {
        "b1": b1, "se_clasico": se_clasico, "se_robusto": se_robusto,
        "x0": x[0], "y0": y[0], "b0_0": b0[0], "b1_0": b1[0],  # réplica 1, para el gráfico
    }


def resumen(res):
    """Sesgo, dispersión y cobertura del IC al 95% (clásico y robusto)."""
    b1 = res["b1"]
    cob = lambda se: float(np.mean(np.abs(b1 - BETA1) <= Z95 * se))
    return {
        "sesgo": float(b1.mean() - BETA1),
        "dispersion": float(b1.std()),
        "cobertura_clasico": cob(res["se_clasico"]),
        "cobertura_robusto": cob(res["se_robusto"]),
    }
