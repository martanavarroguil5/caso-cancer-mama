"""Resúmenes ligeros para la web: métricas, curvas y bandas, calculados solo con train.

Escribe ``app/recursos/*.json`` (versionados) para que la web no dependa de
``runs/`` (ignorado en Git). Nunca lee test.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

from . import datos, metricas

RECURSOS = datos.RAIZ / "app" / "recursos"
RUNS = datos.RAIZ / "runs"


def _wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan")] * 2
    p = k / n
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    margen = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(centro - margen), float(centro + margen)]


def bandas(y, p, nombres=("baja", "intermedia", "alta")) -> dict:
    """Tres bandas de igual tamaño (terciles de las OOF) con la pCR observada en cada una."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    cortes = [float(c) for c in np.quantile(p, [1 / 3, 2 / 3])]
    grupo = np.searchsorted(cortes, p, side="right")
    filas = []
    for i, nombre in enumerate(nombres):
        m = grupo == i
        k, n = int(y[m].sum()), int(m.sum())
        filas.append({"banda": nombre, "n": n, "pcr": k, "tasa_pcr": k / n if n else float("nan"),
                      "ic95": _wilson(k, n), "p_min": float(p[m].min()), "p_max": float(p[m].max())})
    return {"cortes": cortes, "regla": "terciles de las probabilidades OOF (solo train)",
            "prevalencia": float(y.mean()), "bandas": filas}


def _curva(x, y, n=200):
    """Submuestrea una curva a ~n puntos conservando extremos."""
    idx = np.unique(np.linspace(0, len(x) - 1, min(n, len(x))).round().astype(int))
    return [[float(x[i]), float(y[i])] for i in idx]


def _historias(cv: str, brazo: str) -> pd.DataFrame:
    return pd.concat(pd.read_csv(f).assign(red=str(f.parent))
                     for f in sorted((RUNS / cv).glob(f"fold*/{brazo}/semilla*/historia.csv")))


def _por_epoca(h: pd.DataFrame, columnas: list[str], minimo: int = 5) -> list[dict]:
    filas = []
    for epoca, g in h.groupby("epoca"):
        if len(g) < minimo:
            continue
        fila = {"epoca": int(epoca), "n": int(len(g))}
        for c in columnas:
            fila[c] = float(g[c].mean())
            fila[c + "_p25"], fila[c + "_p75"] = (float(v) for v in g[c].quantile([0.25, 0.75]))
        filas.append(fila)
    return filas


def modelo_imagen(cv: str = "cv_presupuesto", brazo: str = "ponderada_aux0",
                  paquete: str = "party_final_v2", control: str = "cv_party") -> dict:
    info = json.loads((datos.RAIZ / "modelos" / paquete / "paquete.json").read_text())
    politica = info["politica"]
    oof = pd.read_csv(RUNS / cv / f"oof_{brazo}.csv")
    y, p = oof["pCR"].to_numpy(), oof["p"].to_numpy()
    pc = metricas.aplicar_platt(politica["calibracion_platt"], p)
    media, de, folds = metricas.auc_media_folds(oof, "p")
    rng = np.random.default_rng(20261027)
    boot = [metricas.auc(y[i], p[i]) for i in (
        np.concatenate([rng.choice(np.flatnonzero(y == c), (y == c).sum()) for c in (0, 1)])
        for _ in range(2000))]
    fpr, tpr, _ = roc_curve(y, p)
    fpr_c, tpr_c, _ = roc_curve(y, oof["p_cohorte"])
    prec, rec, _ = precision_recall_curve(y, p)
    bordes = np.quantile(pc, np.linspace(0, 1, 11))
    decil = np.clip(np.searchsorted(bordes, pc, side="right") - 1, 0, 9)
    sub = np.select([oof["HR"].isna() | oof["HER2"].isna(), oof["HER2"] == 1, oof["HR"] == 1],
                    ["desconocido", "HER2+", "HR+/HER2−"], "Triple negativo")
    h = _historias(cv, brazo)
    hc = _historias(control, brazo)
    return {
        "paquete": paquete, "cv": cv, "brazo": brazo, "epocas": info["epocas"], "semillas": info["semillas"],
        "parametros": info["parametros_entrenables"], "n_pacientes": int(len(oof)), "n_pcr": int(y.sum()),
        "metricas": {
            "auc_media_folds": media, "auc_de_folds": de, "auc_por_fold": folds,
            "auc_agrupada": metricas.auc(y, p), "auc_ic95": [float(v) for v in np.quantile(boot, [0.025, 0.975])],
            "ap": float(average_precision_score(y, p)), "prevalencia": float(y.mean()),
            "auc_cohorte_media_folds": metricas.auc_media_folds(oof, "p_cohorte")[0],
            "ap_cohorte": float(average_precision_score(y, oof["p_cohorte"])),
            "auc_por_cohorte": {c: metricas.auc(g["pCR"], g["p"]) for c, g in oof.groupby("cohorte")},
            "auc_por_subtipo": {s: metricas.auc(g["pCR"], g["p"]) for s, g in oof.assign(s=sub).groupby("s")
                                if s != "desconocido"},
        },
        "curvas": _por_epoca(h, ["bce_pcr_fit_aumentada", "fit_bce_obj", "sel_bce_obj", "fit_auc", "sel_auc"]),
        "curvas_control": _por_epoca(hc, ["fit_auc", "sel_auc"]),
        "roc": _curva(fpr, tpr), "roc_cohorte": _curva(fpr_c, tpr_c),
        "pr": _curva(rec[::-1], prec[::-1]),
        "calibracion": [{"predicha": float(pc[decil == d].mean()), "observada": float(y[decil == d].mean()),
                         "n": int((decil == d).sum())} for d in range(10)],
        "umbrales": {"youden": politica["oof_con_umbral"], "macro_f1": politica["oof_con_umbral_macro_f1"]},
        "calibracion_platt": politica["calibracion_platt"],
        "bandas": bandas(y, pc),
    }


def modelo_fusion(interna: str = "cv_interna_fusion", paquete: str = "party_fusion_v1") -> dict | None:
    """Resumen de la CV anidada de la fusión; None si aún no existe."""
    evaluacion = RUNS / interna / "evaluacion"
    if not (evaluacion / "resumen.json").exists():
        return None
    resumen = json.loads((evaluacion / "resumen.json").read_text())
    oof = pd.read_csv(evaluacion / "oof_fusion.csv")
    alias = resumen["alias"]
    curvas, bandas_sub = {}, {}
    for nombre in ("imagen", "subtipo", "completa", "completa+imagen"):
        t = oof.dropna(subset=[alias[nombre]])
        fpr, tpr, _ = roc_curve(t["pCR"], t[alias[nombre]])
        curvas[nombre] = _curva(fpr, tpr)
    for clave in resumen["brazos"]:
        if clave.endswith("|normal") and clave.split("|")[0].endswith("imagen"):
            t = oof.dropna(subset=[clave])
            bandas_sub[clave.split("|")[0]] = bandas(t["pCR"], t[clave])
    coef = pd.read_csv(evaluacion / "coeficientes_completa_imagen.csv")
    return {"resumen": resumen, "roc": curvas, "bandas": bandas_sub,
            "coeficientes_por_fold": coef.to_dict(orient="records"),
            "paquete_disponible": (datos.RAIZ / "modelos" / paquete / "paquete.json").exists()}


def escribir() -> list[Path]:
    RECURSOS.mkdir(parents=True, exist_ok=True)
    rutas = [RECURSOS / "modelo_imagen.json"]
    rutas[0].write_text(json.dumps(modelo_imagen(), indent=1, ensure_ascii=False))
    fusion = modelo_fusion()
    if fusion is not None:
        rutas.append(RECURSOS / "modelo_fusion.json")
        rutas[1].write_text(json.dumps(fusion, indent=1, ensure_ascii=False))
    return rutas
