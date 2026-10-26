"""Inferencia con un paquete congelado y evaluación única sobre test."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import datos, metricas
from .entrenar import cargar_modelo, predecir
from .final import sha256


class Paquete:
    def __init__(self, ruta: Path, dispositivo: str | None = None):
        self.ruta = Path(ruta)
        self.info = json.loads((self.ruta / "paquete.json").read_text())
        for relativa, esperado in self.info["sha256"].items():
            if sha256(self.ruta / relativa) != esperado:
                raise RuntimeError(f"SHA-256 distinto en {relativa}")
        self.dispositivo = torch.device(dispositivo or ("cuda" if torch.cuda.is_available() else "cpu"))
        receta = {"modelo": self.info["config_modelo"]}
        self.modelos = [
            cargar_modelo(receta, torch.load(self.ruta / relativa, map_location="cpu",
                                             weights_only=True), self.dispositivo)
            for relativa in sorted(self.info["sha256"])
        ]
        self.tta = self.info["tta"] is not None

    def probabilidad_cortes(self, pixeles: np.ndarray) -> np.ndarray:
        """uint8 [N,3,256,256] → probabilidad cruda (sin calibrar) de pCR por corte."""
        if pixeles.dtype != np.uint8 or pixeles.ndim != 4 or pixeles.shape[1:] != (3, 256, 256):
            raise ValueError("Se espera uint8 [N,3,256,256]")
        tensor = torch.from_numpy(np.ascontiguousarray(pixeles))
        filas = np.arange(len(pixeles))
        return np.mean([predecir(m, tensor, filas, tta=self.tta)[0] for m in self.modelos], axis=0)

    def decidir(self, p_cruda: float) -> dict:
        politica = self.info["politica"]
        calibrada = float(metricas.aplicar_platt(politica["calibracion_platt"], p_cruda))
        return {"probabilidad_pcr": calibrada, "probabilidad_cruda": float(p_cruda),
                "umbral": politica["umbral"],
                "clase": "pCR" if calibrada >= politica["umbral"] else "no pCR",
                "modelo": self.info["nombre"]}


def leer_tripleta(pre: Path, early: Path, late: Path) -> np.ndarray:
    rutas = [Path(pre), Path(early), Path(late)]
    if len({r.resolve() for r in rutas}) != 3:
        raise ValueError("Las tres fases deben ser archivos distintos")
    return np.stack([datos.leer_png(r) for r in rutas])[None]


def evaluar_test(ruta_paquete: Path, forzar: bool = False) -> dict:
    """Evaluación final única. Escribe la marca y se niega a repetir sin --forzar."""
    paquete = Paquete(ruta_paquete)
    marca = paquete.ruta / "evaluacion_test.json"
    if marca.exists() and not forzar:
        raise RuntimeError(f"{marca} ya existe: el test de este paquete ya se evaluó")
    indice = datos.cargar_indice("test")
    pixeles = datos.cargar_imagenes(indice, cache=None)
    cortes = indice[["patient_id", "pCR", "cohorte"]].copy()
    cortes["p"] = paquete.probabilidad_cortes(pixeles)
    pac = cortes.groupby("patient_id").agg(pCR=("pCR", "first"), cohorte=("cohorte", "first"),
                                            p=("p", "mean"))
    politica = paquete.info["politica"]
    pac["p_calibrada"] = metricas.aplicar_platt(politica["calibracion_platt"], pac["p"])
    rng = np.random.default_rng(20261026)
    boot = []
    y, p = pac["pCR"].to_numpy(), pac["p"].to_numpy()
    for _ in range(5000):
        idx = np.concatenate([rng.choice(np.flatnonzero(y == c), (y == c).sum()) for c in (0, 1)])
        boot.append(metricas.auc(y[idx], p[idx]))
    resultado = {
        "paquete": paquete.info["nombre"],
        "pacientes": len(pac), "pcr": int(pac["pCR"].sum()), "cortes": len(cortes),
        "auc": metricas.auc(y, p),
        "auc_ic95": [float(v) for v in np.quantile(boot, [0.025, 0.975])],
        "probabilidades_calibradas": metricas.resumen_probabilidades(pac["pCR"], pac["p_calibrada"]),
        "umbral_congelado": metricas.binarias(pac["pCR"], pac["p_calibrada"], politica["umbral"]),
        "auc_por_cohorte": {c: metricas.auc(g["pCR"], g["p"]) for c, g in pac.groupby("cohorte")},
    }
    pac.to_csv(paquete.ruta / "predicciones_test.csv")
    marca.write_text(json.dumps(resultado, indent=2, ensure_ascii=False))
    return resultado
