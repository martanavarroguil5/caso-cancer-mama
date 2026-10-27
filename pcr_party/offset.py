"""Piloto: PartyNet entrenada sobre la clínica (docs/PROTOCOLO_OFFSET.md).

El logit de cada corte es o_clínico + f(x). El offset es fijo (no se entrena),
así que la red solo puede mejorar la predicción a través de la imagen. Misma
receta, roles y semillas que ``cv_presupuesto`` para comparar de forma pareada.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import datos, fusion, metricas
from .cv import RUNS, Registro, entorno, pixeles_en_dispositivo
from .entrenar import cargar_modelo, entrenar, predecir

BRAZOS = {"offset": 1, "offset_subtipo": 3}
SUBTIPOS = ("HR+/HER2−", "HER2+", "TN")
COMPLETA = tuple(fusion.GRUPOS)
FUSION_V1 = "subtipo+edad+volumen+imagen|normal"
CLINICA_FUSION = "subtipo+edad+volumen|normal"


def grupo_subtipo(clinica: pd.DataFrame) -> pd.Series:
    """0 HR+/HER2−, 1 HER2+, 2 triple negativo; −1 si falta HR o HER2."""
    hr, her2 = clinica["HR"], clinica["HER2"]
    g = np.select([her2 == 1, (hr == 1) & (her2 == 0), (hr == 0) & (her2 == 0)], [1, 0, 2], -1)
    return pd.Series(g, index=clinica.index).where(hr.notna() & her2.notna(), -1).astype(int)


def _logit_clinico(ajuste: pd.DataFrame, prueba: pd.DataFrame) -> pd.Series:
    """Logística clínica completa por patrón (sin imagen); sin datos → prevalencia del ajuste."""
    submodelos = fusion.ajustar_submodelos(ajuste.assign(imagen=0.0), ponderada=False)
    p, _ = fusion.predecir_patron(submodelos, prueba, COMPLETA, imagen=False)
    p = np.where(np.isnan(p), ajuste["pCR"].mean(), p)
    return pd.Series(metricas.logit(p), index=prueba.index)


def desplazamientos(clinica: pd.DataFrame, exterior: int) -> pd.Series:
    """Offset por paciente para el fold exterior ``exterior``, sin fugas.

    Desarrollo (folds ≠ k): ajuste cruzado con los folds de ``samples.csv``; la
    paciente del fold j recibe la logística ajustada con los folds ∉ {k, j}. La
    partición no depende de ninguna etiqueta. Exterior: logística ajustada con
    todo el desarrollo.
    """
    desarrollo = clinica[clinica["fold"] != exterior]
    partes = [_logit_clinico(desarrollo, clinica[clinica["fold"] == exterior])]
    for j in sorted(desarrollo["fold"].unique()):
        partes.append(_logit_clinico(desarrollo[desarrollo["fold"] != j], desarrollo[desarrollo["fold"] == j]))
    return pd.concat(partes).reindex(clinica.index)


def _clinica_con_fold(indice: pd.DataFrame) -> pd.DataFrame:
    clinica = fusion.tabla_clinica("train")
    folds = datos.tabla_pacientes(indice).set_index("patient_id")["fold"]
    if set(folds.index) != set(clinica.index):
        raise RuntimeError("samples.csv y patients.csv no cubren las mismas pacientes")
    return clinica.join(folds)


def cv(nombre: str = "cv_offset", control: str = fusion.CV_EXTERIOR) -> Path:
    plan_control = json.loads((RUNS / control / "plan.json").read_text())
    receta = deepcopy(plan_control["receta"])
    folds, semillas = plan_control["folds"], plan_control["semillas"]
    salida = RUNS / nombre
    salida.mkdir(parents=True, exist_ok=True)
    plan = {"receta": receta, "folds": folds, "semillas": semillas, "brazos": BRAZOS,
            "perdida": "normal", "control": control, "protocolo": "docs/PROTOCOLO_OFFSET.md",
            "entorno": entorno()}
    (salida / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    registro = Registro(salida / "registro.log")

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    indice = datos.cargar_indice("train")
    pacientes = datos.tabla_pacientes(indice)
    clinica = _clinica_con_fold(indice)
    grupo = grupo_subtipo(clinica).reindex(indice["patient_id"]).to_numpy()
    pixeles = pixeles_en_dispositivo(indice, dispositivo)
    registro(f"{len(pacientes)} pacientes, {len(indice)} cortes, dispositivo {dispositivo}")

    offsets = {k: desplazamientos(clinica, k) for k in folds}
    pd.DataFrame(offsets).to_csv(salida / "desplazamientos.csv")
    for semilla in semillas:
        for fold in folds:
            roles = datos.roles_fold(pacientes, fold, receta["fraccion_seleccion"])
            previos = pd.read_csv(RUNS / control / f"roles_fold{fold}.csv")
            if not roles[["patient_id", "rol"]].reset_index(drop=True).equals(previos[["patient_id", "rol"]]):
                raise RuntimeError(f"Los roles del fold {fold} no coinciden con {control}")
            rol = indice["patient_id"].map(roles.set_index("patient_id")["rol"]).to_numpy()
            fit, seleccion = np.flatnonzero(rol == "fit"), np.flatnonzero(rol == "seleccion")
            exterior = np.flatnonzero(rol == "exterior")
            o = offsets[fold].reindex(indice["patient_id"]).to_numpy(np.float32)
            for brazo, cabezas in BRAZOS.items():
                carpeta = salida / f"fold{fold}" / brazo / f"semilla{semilla}"
                if (carpeta / "hecho.json").exists():
                    continue
                carpeta.mkdir(parents=True, exist_ok=True)
                registro(f"== fold {fold} · {brazo} · semilla {semilla}")
                receta_brazo = deepcopy(receta)
                receta_brazo["modelo"]["cabezas"] = cabezas
                g = grupo if cabezas > 1 else None
                resultado = entrenar(receta_brazo, pixeles, indice, fit, seleccion, semilla, "normal",
                                     0.0, registro=registro, desplazamiento=o, grupo=g)
                torch.save(resultado.estado, carpeta / "modelo.pt")
                pd.DataFrame(resultado.historia).to_csv(carpeta / "historia.csv", index=False)
                modelo = cargar_modelo(receta_brazo, resultado.estado, dispositivo)
                filas = np.concatenate((exterior, seleccion))
                p, _ = predecir(modelo, pixeles, filas, tta=receta["tta"], desplazamiento=o, grupo=g)
                p_img, _ = predecir(modelo, pixeles, filas, tta=receta["tta"], grupo=g)
                cortes = indice.iloc[filas][["sample_id", "patient_id"]].copy()
                cortes["rol"], cortes["p"], cortes["p_imagen"] = rol[filas], p, p_img
                cortes["desplazamiento"] = o[filas]
                cortes.to_csv(carpeta / "predicciones.csv", index=False)
                (carpeta / "hecho.json").write_text(json.dumps({
                    "epoca_elegida": resultado.epoca, "segundos": resultado.segundos,
                    "memoria_max_mb": resultado.memoria_max_mb}, indent=2))
                registro(f"   {resultado.segundos / 60:.1f} min")
    return salida


def oof(salida: Path, brazo: str, folds: list[int]) -> pd.DataFrame:
    """Por paciente exterior: media de cortes y luego de semillas de p, p_imagen y offset."""
    tablas = []
    for fold in folds:
        semillas = []
        for ruta in sorted((salida / f"fold{fold}" / brazo).glob("semilla*/predicciones.csv")):
            if not (ruta.parent / "hecho.json").exists():
                raise RuntimeError(f"{ruta.parent} sin terminar")
            c = pd.read_csv(ruta)
            semillas.append(metricas.por_paciente(c[c["rol"] == "exterior"], ["p", "p_imagen", "desplazamiento"]))
        tablas.append((sum(semillas) / len(semillas)).assign(fold=fold, n_semillas=len(semillas)))
    return pd.concat(tablas)


def evaluar(nombre: str = "cv_offset", interna: str = "cv_interna_fusion") -> dict:
    salida = RUNS / nombre
    plan = json.loads((salida / "plan.json").read_text())
    fus = pd.read_csv(RUNS / interna / "evaluacion" / "oof_fusion.csv").set_index("patient_id")
    tabla = fus[["pCR", "fold", "cohorte", "HR", "HER2"]].copy()
    tabla["fusion_v1"] = fus[FUSION_V1]
    tabla["clinica_fusion"] = fus[CLINICA_FUSION]
    tabla["cnn_actual"] = fus["p_cnn"]
    for brazo in BRAZOS:
        o = oof(salida, brazo, plan["folds"])
        if (o["fold"] != tabla.loc[o.index, "fold"]).any() or len(o) != len(tabla):
            raise RuntimeError("Las OOF del piloto y de la fusión no cuadran")
        tabla[brazo] = o["p"]
        tabla[f"{brazo}|imagen"] = o["p_imagen"]
        tabla[f"{brazo}|clinica"] = 1 / (1 + np.exp(-o["desplazamiento"]))
    if not np.allclose(tabla["offset|clinica"], tabla["offset_subtipo|clinica"]):
        raise RuntimeError("Los dos brazos deberían compartir el offset")
    tabla["clinica_offset"] = tabla["offset|clinica"]

    columnas = ["cnn_actual", "clinica_offset", "clinica_fusion", "fusion_v1", *BRAZOS,
                *(f"{b}|imagen" for b in BRAZOS)]
    aucs = {}
    for c in columnas:
        media, de, por_fold = metricas.auc_media_folds(tabla, c)
        aucs[c] = {"media": media, "de": de, "por_fold": por_fold,
                   **{k: v for k, v in metricas.resumen_probabilidades(tabla["pCR"], tabla[c]).items()
                      if k in ("auc", "ap", "brier")}}
    comp = {}
    for brazo in BRAZOS:
        comp[f"{brazo} − fusion_v1"] = metricas.bootstrap_delta(tabla, brazo, "fusion_v1")
        comp[f"{brazo} − clinica_offset"] = metricas.bootstrap_delta(tabla, brazo, "clinica_offset")
    comp["fusion_v1 − clinica_fusion"] = metricas.bootstrap_delta(tabla, "fusion_v1", "clinica_fusion")
    comp["clinica_offset − clinica_fusion"] = metricas.bootstrap_delta(tabla, "clinica_offset", "clinica_fusion")
    comp["offset_subtipo − offset"] = metricas.bootstrap_delta(tabla, "offset_subtipo", "offset")

    sub = grupo_subtipo(tabla).map(dict(enumerate(SUBTIPOS))).fillna("desconocido")
    por_grupo = {}
    for etiqueta, g in [*((f"subtipo {s}", t) for s, t in tabla.groupby(sub)),
                        *((f"cohorte {c}", t) for c, t in tabla.groupby("cohorte"))]:
        if g["pCR"].nunique() < 2:
            continue
        por_grupo[etiqueta] = {"n": len(g), **{c: metricas.auc(g["pCR"], g[c]) for c in
                                                ("clinica_offset", "fusion_v1", *BRAZOS)}}

    decision = {}
    for brazo in BRAZOS:
        d_fus = comp[f"{brazo} − fusion_v1"]["delta"]
        ic = comp[f"{brazo} − clinica_offset"]["ic95"]
        decision[brazo] = {"delta_frente_fusion": d_fus, "ic_frente_clinica": ic,
                           "cumple": bool(d_fus >= 0.010 and ic[0] > 0)}
    candidatos = [b for b in BRAZOS if decision[b]["cumple"]]
    elegido = max(candidatos, key=lambda b: decision[b]["delta_frente_fusion"]) if candidatos else None
    r = {"aucs": aucs, "comparaciones": comp, "por_grupo": por_grupo, "decision": decision,
         "adoptado": elegido or "ninguno (sigue party_fusion_v1)"}
    ev = salida / "evaluacion"
    ev.mkdir(exist_ok=True)
    tabla.to_csv(ev / "oof_offset.csv")
    (ev / "resumen.json").write_text(json.dumps(r, indent=2, ensure_ascii=False))
    (ev / "RESUMEN.md").write_text(_markdown(r))
    return r


def _markdown(r: dict) -> str:
    f = lambda x: f"{x:.3f}"
    lineas = ["# Piloto offset: CNN entrenada sobre la clínica", "",
              "Protocolo: `docs/PROTOCOLO_OFFSET.md`. OOF de train, 5 folds × 3 semillas.", "",
              "| Modelo | AUC media ± DE | AUC agrupada | AP | Brier |", "|---|---|---|---|---|"]
    for c, a in r["aucs"].items():
        lineas.append(f"| {c} | {f(a['media'])} ± {f(a['de'])} | {f(a['auc'])} | {f(a['ap'])} | {f(a['brier'])} |")
    lineas += ["", "| Comparación | Δ AUC | IC95 % | P(Δ ≤ 0) |", "|---|---|---|---|"]
    for c, d in r["comparaciones"].items():
        lineas.append(f"| {c} | {d['delta']:+.4f} | [{d['ic95'][0]:+.4f}; {d['ic95'][1]:+.4f}] | "
                      f"{d['p_delta_menor_igual_0']:.3f} |")
    lineas += ["", "| Grupo | n | Clínica | Fusión v1 | Offset | Offset subtipo |", "|---|---|---|---|---|---|"]
    for g, v in r["por_grupo"].items():
        lineas.append(f"| {g} | {v['n']} | {f(v['clinica_offset'])} | {f(v['fusion_v1'])} | "
                      f"{f(v['offset'])} | {f(v['offset_subtipo'])} |")
    lineas += ["", "## Criterio de adopción", ""]
    for b, d in r["decision"].items():
        lineas.append(f"- `{b}`: Δ frente a la fusión {d['delta_frente_fusion']:+.4f} (≥ +0,010), IC frente a "
                      f"su clínica [{d['ic_frente_clinica'][0]:+.4f}; {d['ic_frente_clinica'][1]:+.4f}] "
                      f"(excluye 0) → {'cumple' if d['cumple'] else 'no cumple'}")
    lineas += ["", f"**Adoptado:** {r['adoptado']}", ""]
    return "\n".join(lineas)
