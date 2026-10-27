"""Lógica de la web sin Streamlit: paquetes, lectura de PNG, predicción y arquitectura."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pcr_party import datos, metricas
from pcr_party.entrenar import predecir
from pcr_party.inferencia import Paquete
from pcr_party.modelo import PartyNet

MODELOS = datos.RAIZ / "modelos"
PAQUETE_IMAGEN = MODELOS / "party_final_v2"
PAQUETE_FUSION = MODELOS / "party_fusion_v1"
FASE = re.compile(r"^(?P<base>.+?)[_\-\s]?(?P<fase>PRE|EARLY|LATE)\.png$", re.IGNORECASE)


@dataclass
class Prediccion:
    p_cortes: np.ndarray          # [redes, cortes] probabilidad cruda
    nombres: list[str]

    @property
    def p_cruda(self) -> float:
        return float(self.p_cortes.mean())

    @property
    def por_corte(self) -> np.ndarray:
        return self.p_cortes.mean(0)

    @property
    def por_red(self) -> np.ndarray:
        return self.p_cortes.mean(1)


def cargar_imagen() -> Paquete:
    return Paquete(PAQUETE_IMAGEN, "cpu")


def cargar_fusion():
    if not (PAQUETE_FUSION / "paquete.json").exists():
        return None
    from pcr_party.fusion import PaqueteFusion
    return PaqueteFusion(PAQUETE_FUSION, "cpu")


def agrupar_subidas(archivos) -> tuple[dict[str, dict[str, object]], list[str]]:
    """Agrupa PNG subidos por corte según el sufijo _PRE/_EARLY/_LATE del nombre."""
    cortes: dict[str, dict[str, object]] = {}
    errores = []
    for a in archivos:
        m = FASE.match(a.name)
        if not m:
            errores.append(f"{a.name}: el nombre debe terminar en _PRE.png, _EARLY.png o _LATE.png")
            continue
        cortes.setdefault(m["base"], {})[m["fase"].upper()] = a
    incompletos = [b for b, f in cortes.items() if len(f) != 3]
    errores += [f"{b}: faltan fases ({', '.join(sorted({'PRE', 'EARLY', 'LATE'} - set(cortes[b]))) })"
                for b in incompletos]
    return {b: f for b, f in sorted(cortes.items()) if len(f) == 3}, errores


def leer_cortes(grupos: dict[str, dict[str, object]]) -> np.ndarray:
    """uint8 [N,3,256,256]; valida cada PNG con la misma regla que el entrenamiento."""
    return np.stack([np.stack([datos.leer_png(f[fase]) for fase in datos.FASES]) for f in grupos.values()])


def predecir_cortes(paquete: Paquete, pixeles: np.ndarray, nombres: list[str]) -> Prediccion:
    tensor = torch.from_numpy(np.ascontiguousarray(pixeles))
    filas = np.arange(len(pixeles))
    p = np.stack([predecir(m, tensor, filas, tta=paquete.tta)[0] for m in paquete.modelos])
    return Prediccion(p, nombres)


def calibrar(paquete: Paquete, p_cruda: float) -> float:
    return float(metricas.aplicar_platt(paquete.info["politica"]["calibracion_platt"], p_cruda))


# ---------------------------------------------------------------- ejemplos de train

def ejemplos(n: int = 40) -> pd.DataFrame:
    """Pacientes de train (el modelo final las vio) para probar la interfaz."""
    from pcr_party.fusion import tabla_clinica

    indice = datos.cargar_indice("train")
    clin = tabla_clinica("train")
    pac = indice.groupby("patient_id").agg(cohorte=("cohorte", "first"), cortes=("sample_id", "size"))
    pac = pac.join(clin[["edad", "log_volumen", "HR", "HER2", "pCR"]])
    return pac.groupby(["cohorte", "pCR"]).sample(n=max(1, n // 6), random_state=7).sort_index()


def cortes_de_ejemplo(paciente: str) -> tuple[np.ndarray, list[str]]:
    indice = datos.cargar_indice("train")
    filas = indice[indice["patient_id"] == paciente]
    pix = np.stack([np.stack([datos.leer_png(datos.DATOS / r) for r in f[list(datos.COLUMNAS_FASE)]])
                    for _, f in filas.iterrows()])
    return pix, [f"z{int(z):03d}" for z in filas["slice_index"]]


# ---------------------------------------------------------------- mapas para mostrar

def mapa_realce(corte: np.ndarray) -> np.ndarray:
    """EARLY − PRE en color (rojo = realce) como RGB uint8."""
    from matplotlib import colormaps
    d = corte[1].astype(float) - corte[0].astype(float)
    escala = max(np.percentile(np.abs(d), 99.5), 1.0)
    rgb = colormaps["RdBu_r"]((np.clip(d / escala, -1, 1) + 1) / 2)[..., :3]
    return (rgb * 255).astype(np.uint8)


# ---------------------------------------------------------------- arquitectura

def capas_red() -> list[dict]:
    """Capas de PartyNet para la vista 3D, con parámetros contados del propio modelo."""
    cfg = json.loads((PAQUETE_IMAGEN / "paquete.json").read_text())["config_modelo"]
    m = PartyNet(**cfg)
    contar = lambda mod: sum(p.numel() for p in mod.parameters())
    anchos, lado, escalas = cfg["anchos"], cfg["lado"], cfg["escalas"]
    colores = ["#c9d1dc", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#1baf7a", "#eb6834", "#eda100"]
    caja = lambda c, s: {"d": max(10, 7 * c ** 0.5), "h": 26 + 1.15 * s, "w": 26 + 1.15 * s}
    capas = [
        {"nombre": "Entrada", "forma": "3 fases × 256 × 256", **caja(3, 128),
         "detalle": "PRE, EARLY y LATE del mismo corte, en gris y divididas entre 255", "params": 0},
        {"nombre": "Canales cinéticos", "forma": f"6 canales × {len(escalas)} vistas × {lado} × {lado}", **caja(6, 104),
         "detalle": "Se añaden EARLY−PRE, LATE−PRE y EARLY−LATE; recortes centrales de 256, 128 y 64 px reescalados a 128",
         "params": 0},
    ]
    entrada, s = 6, lado
    for i, (c, bloque) in enumerate(zip(anchos, m.backbone), 1):
        s //= 2
        capas.append({"nombre": f"Bloque conv {i}", "forma": f"{c} × {s} × {s}", **caja(c, s * 1.6),
                      "detalle": f"2 convoluciones 3×3 ({entrada}→{c}) con BatchNorm y ReLU, y MaxPool 2×2",
                      "params": f"{contar(bloque):,}".replace(",", ".")})
        entrada = c
    capas += [
        {"nombre": "Pooling global", "forma": f"{anchos[-1]}×2×{len(escalas)} = {anchos[-1] * 2 * len(escalas)}",
         **caja(anchos[-1] * 2 * len(escalas) / 6, 6),
         "detalle": "Media y máximo de cada mapa en las tres vistas (pesos compartidos); Dropout2d 0,10", "params": 0},
        {"nombre": "Capa densa", "forma": f"{cfg['oculto']} neuronas", **caja(cfg["oculto"], 4),
         "detalle": f"Dropout {cfg['dropout']}, lineal {anchos[-1] * 2 * len(escalas)}→{cfg['oculto']} y ReLU",
         "params": f"{contar(m.tronco):,}".replace(",", ".")},
        {"nombre": "Salida", "forma": "1 logit de pCR", **caja(1, 2),
         "detalle": "Sigmoide → probabilidad del corte; la paciente es la media de sus cortes y de 5 redes",
         "params": f"{contar(m.cabeza_pcr):,}".replace(",", ".")},
    ]
    for c, color in zip(capas, colores):
        c["color"] = color
    return capas
