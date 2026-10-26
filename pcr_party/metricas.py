"""Métricas por paciente, bootstrap pareado, calibración y umbrales."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_curve

EPS = 1e-6


def auc(y, p) -> float:
    """ROC-AUC por rangos (Mann-Whitney); NaN si falta una clase."""
    y = np.asarray(y).astype(bool)
    p = np.asarray(p, dtype=float)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    rangos = rankdata(p)
    return float((rangos[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bce(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def spearman(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    validos = ~(np.isnan(x) | np.isnan(y))
    return float(spearmanr(x[validos], y[validos]).statistic) if validos.sum() > 2 else float("nan")


def por_paciente(cortes: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """Media de las columnas de predicción entre los cortes de cada paciente."""
    return cortes.groupby("patient_id", sort=True)[columnas].mean()


def auc_media_folds(tabla: pd.DataFrame, columna: str) -> tuple[float, float, list[float]]:
    valores = [auc(g["pCR"], g[columna]) for _, g in tabla.groupby("fold", sort=True)]
    return float(np.mean(valores)), float(np.std(valores, ddof=1)), valores


def bootstrap_delta(tabla: pd.DataFrame, a: str, b: str, n: int = 2000,
                    semilla: int = 20261026) -> dict:
    """Δ de la media de AUC por fold (a − b), remuestreo pareado por fold × clase."""
    rng = np.random.default_rng(semilla)
    y = tabla["pCR"].to_numpy()
    pa = tabla[a].to_numpy()
    pb = tabla[b].to_numpy()
    estratos = [
        [np.flatnonzero((tabla["fold"].to_numpy() == f) & (y == c)) for c in (0, 1)]
        for f in sorted(tabla["fold"].unique())
    ]
    deltas = np.empty(n)
    for r in range(n):
        da, db = [], []
        for negativos, positivos in estratos:
            idx = np.concatenate((rng.choice(negativos, len(negativos)),
                                  rng.choice(positivos, len(positivos))))
            da.append(auc(y[idx], pa[idx]))
            db.append(auc(y[idx], pb[idx]))
        deltas[r] = np.mean(da) - np.mean(db)
    puntual = auc_media_folds(tabla, a)[0] - auc_media_folds(tabla, b)[0]
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return {"delta": float(puntual), "ic95": [float(lo), float(hi)],
            "p_delta_menor_igual_0": float(np.mean(deltas <= 0)), "remuestreos": n}


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def ajustar_platt(y, p) -> dict:
    """Regresión logística sobre logit(p); pendiente negativa → identidad."""
    modelo = LogisticRegression(C=1.0).fit(logit(p).reshape(-1, 1), np.asarray(y))
    a, b = float(modelo.coef_[0, 0]), float(modelo.intercept_[0])
    if a <= 0:
        return {"a": 1.0, "b": 0.0, "nota": "pendiente no positiva; identidad"}
    return {"a": a, "b": b}


def aplicar_platt(calibracion: dict, p):
    return 1 / (1 + np.exp(-(calibracion["a"] * logit(p) + calibracion["b"])))


def umbral_youden(y, p) -> float:
    fpr, tpr, umbrales = roc_curve(y, p)
    finitos = np.isfinite(umbrales)
    j = np.where(finitos, tpr - fpr, -np.inf)
    return float(umbrales[int(np.argmax(j))])


def umbral_macro_f1(y, p) -> float:
    y = np.asarray(y)
    candidatos = np.unique(np.asarray(p))
    mejor, umbral = -1.0, 0.5
    for t in candidatos:
        f = binarias(y, p, t)["macro_f1"]
        if f > mejor:
            mejor, umbral = f, float(t)
    return umbral


def binarias(y, p, umbral: float) -> dict:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= umbral).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    def f1(tp_, fp_, fn_):
        return 2 * tp_ / (2 * tp_ + fp_ + fn_) if tp_ else 0.0

    return {
        "umbral": float(umbral), "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "sensibilidad": tp / (tp + fn) if tp + fn else float("nan"),
        "especificidad": tn / (tn + fp) if tn + fp else float("nan"),
        "precision_pcr": tp / (tp + fp) if tp + fp else float("nan"),
        "exactitud": (tp + tn) / len(y),
        "f1_pcr": f1(tp, fp, fn),
        "macro_f1": (f1(tp, fp, fn) + f1(tn, fn, fp)) / 2,
    }


def resumen_probabilidades(y, p) -> dict:
    return {"auc": auc(y, p), "ap": float(average_precision_score(y, p)),
            "bce": bce(y, p), "brier": float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))}
