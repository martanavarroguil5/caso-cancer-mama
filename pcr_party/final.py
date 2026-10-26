"""Modelo final: todas las pacientes de train, épocas fijadas por la CV.

La política (Platt + umbral) se ajusta con las OOF de la CV del mismo brazo,
nunca con test. El paquete resultante es inmutable y lleva SHA-256 de todo.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import datos, metricas
from .cv import Registro, entorno, nombre_brazo, pixeles_en_dispositivo
from .entrenar import entrenar
from .modelo import numero_parametros, PartyNet

MODELOS = datos.RAIZ / "modelos"


def sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def politica_desde_oof(oof: pd.DataFrame) -> dict:
    calibracion = metricas.ajustar_platt(oof["pCR"], oof["p"])
    calibrada = metricas.aplicar_platt(calibracion, oof["p"])
    youden = metricas.umbral_youden(oof["pCR"], calibrada)
    macro = metricas.umbral_macro_f1(oof["pCR"], calibrada)
    return {
        "calibracion_platt": calibracion,
        "umbral": youden,
        "regla_umbral": "Youden máximo sobre OOF calibradas de la CV (solo train)",
        "umbral_macro_f1_alternativo": macro,
        "oof_con_umbral": metricas.binarias(oof["pCR"], calibrada, youden),
        "oof_con_umbral_macro_f1": metricas.binarias(oof["pCR"], calibrada, macro),
        "oof_probabilidades_calibradas": metricas.resumen_probabilidades(oof["pCR"], calibrada),
    }


def entrenar_final(cv: Path, perdida: str, lambda_aux: float, semillas: list[int],
                   nombre: str, epocas: int | None = None) -> Path:
    resumen = json.loads((cv / "resumen.json").read_text())
    receta = resumen["plan"]["receta"]
    brazo = nombre_brazo(perdida, lambda_aux)
    if brazo not in resumen["brazos"]:
        raise ValueError(f"La CV {cv} no tiene el brazo {brazo}")
    if resumen["brazos"][brazo]["semillas_por_fold"] < len(resumen["plan"]["semillas"]):
        raise RuntimeError("La CV de ese brazo no está completa")
    epocas = epocas or max(1, int(round(resumen["brazos"][brazo]["epoca_elegida_mediana"])))
    salida = MODELOS / nombre
    if salida.exists():
        raise FileExistsError(f"{salida} ya existe; los paquetes no se sobrescriben")
    (salida / "pesos").mkdir(parents=True)
    registro = Registro(salida / "registro.log")

    oof = pd.read_csv(cv / f"oof_{brazo}.csv")
    politica = politica_desde_oof(oof)
    registro(f"Brazo {brazo}: {epocas} épocas, semillas {semillas}")

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    indice = datos.cargar_indice("train")
    pixeles = pixeles_en_dispositivo(indice, dispositivo)
    todas = np.arange(len(indice))
    tiempos = []
    for semilla in semillas:
        resultado = entrenar(receta, pixeles, indice, todas, None, semilla, perdida,
                             lambda_aux, epocas=epocas, registro=registro)
        torch.save(resultado.estado, salida / "pesos" / f"semilla{semilla}.pt")
        pd.DataFrame(resultado.historia).to_csv(salida / f"historia_semilla{semilla}.csv", index=False)
        tiempos.append({"semilla": semilla, "segundos": resultado.segundos,
                        "memoria_max_mb": resultado.memoria_max_mb})

    pesos = sorted((salida / "pesos").glob("*.pt"))
    paquete = {
        "nombre": nombre,
        "arquitectura": "PartyNet",
        "config_modelo": receta["modelo"],
        "parametros_entrenables": numero_parametros(PartyNet(**receta["modelo"])),
        "entrada": "PNG gris uint8 256×256 por fase, apilado PRE/EARLY/LATE y dividido entre 255",
        "salida": "un logit de pCR por corte; paciente = media de probabilidades de sus cortes",
        "tta": "media de probabilidades de 4 reflexiones" if receta["tta"] else None,
        "ensemble": "media de probabilidades de todas las semillas",
        "perdida": perdida,
        "pos_weight": "N0/N1 de pacientes de train" if perdida == "ponderada" else 1.0,
        "lambda_aux": lambda_aux,
        "auxiliares_solo_entrenamiento": ["HR", "HER2", "log_volumen"],
        "epocas": epocas,
        "semillas": semillas,
        "receta": receta,
        "politica": politica,
        "cv_origen": str(cv.relative_to(datos.RAIZ)),
        "cv_brazo": resumen["brazos"][brazo],
        "entrenamiento": tiempos,
        "entorno": entorno(),
        "n_pacientes_train": int(indice["patient_id"].nunique()),
        "test_evaluado": False,
        "sha256": {str(p.relative_to(salida)): sha256(p) for p in pesos},
    }
    (salida / "paquete.json").write_text(json.dumps(paquete, indent=2, ensure_ascii=False))
    registro(json.dumps(politica["oof_con_umbral"], ensure_ascii=False))
    return salida
