"""Validación cruzada por paciente de PartyNet y su resumen.

Cada brazo es (pérdida, λ auxiliar). Para un mismo fold y semilla, los brazos
comparten roles, inicialización, orden de lotes y aumentos: solo cambia la
pérdida. El fold exterior no interviene en ninguna decisión de entrenamiento.
"""
from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import datos, metricas
from .entrenar import RECETA, cargar_modelo, entrenar, predecir

RUNS = datos.RAIZ / "runs"


def nombre_brazo(perdida: str, lambda_aux: float) -> str:
    return f"{perdida}_aux{lambda_aux:g}"


def entorno() -> dict:
    cuda = torch.cuda.is_available()
    return {"python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if cuda else None,
            "host": platform.node()}


def pixeles_en_dispositivo(indice: pd.DataFrame, dispositivo: torch.device) -> torch.Tensor:
    """La caché uint8 completa (≈2,2 GB en train) cabe en la memoria de la RTX 3090."""
    return torch.from_numpy(datos.cargar_imagenes(indice)).to(dispositivo)


class Registro:
    def __init__(self, ruta: Path):
        self.ruta = ruta

    def __call__(self, mensaje: str) -> None:
        print(mensaje, flush=True)
        with open(self.ruta, "a", encoding="utf-8") as f:
            f.write(mensaje + "\n")


def validacion_cruzada(nombre: str, folds: list[int], semillas: list[int],
                       perdidas: list[str], lambdas: list[float],
                       receta: dict = RECETA, control: str | None = None) -> Path:
    """``control``: otra CV con los mismos roles y semillas para comparar brazo a brazo."""
    salida = RUNS / nombre
    salida.mkdir(parents=True, exist_ok=True)
    plan = {"receta": receta, "folds": folds, "semillas": semillas, "perdidas": perdidas,
            "lambdas": lambdas, "control": control, "entorno": entorno()}
    ruta_plan = salida / "plan.json"
    if ruta_plan.exists():
        previo = json.loads(ruta_plan.read_text())
        if previo["receta"] != json.loads(json.dumps(receta)):
            raise RuntimeError(f"{ruta_plan} tiene otra receta; usa otro --nombre")
    ruta_plan.write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    registro = Registro(salida / "registro.log")

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    indice = datos.cargar_indice("train")
    pacientes = datos.tabla_pacientes(indice)
    pixeles = pixeles_en_dispositivo(indice, dispositivo)
    registro(f"{len(pacientes)} pacientes, {len(indice)} cortes, dispositivo {dispositivo}")

    # Semilla por fuera: tras la primera semilla ya hay una lectura completa de los cinco folds.
    for semilla in semillas:
        for fold in folds:
            roles = datos.roles_fold(pacientes, fold, receta["fraccion_seleccion"])
            roles.to_csv(salida / f"roles_fold{fold}.csv", index=False)
            rol_corte = indice["patient_id"].map(roles.set_index("patient_id")["rol"]).to_numpy()
            fit = np.flatnonzero(rol_corte == "fit")
            seleccion = np.flatnonzero(rol_corte == "seleccion")
            exterior = np.flatnonzero(rol_corte == "exterior")
            for perdida in perdidas:
                for lambda_aux in lambdas:
                    carpeta = salida / f"fold{fold}" / nombre_brazo(perdida, lambda_aux) / f"semilla{semilla}"
                    if (carpeta / "hecho.json").exists():
                        continue
                    carpeta.mkdir(parents=True, exist_ok=True)
                    registro(f"== fold {fold} · {nombre_brazo(perdida, lambda_aux)} · semilla {semilla}")
                    resultado = entrenar(receta, pixeles, indice, fit, seleccion, semilla,
                                         perdida, lambda_aux, registro=registro)
                    torch.save(resultado.estado, carpeta / "modelo.pt")
                    pd.DataFrame(resultado.historia).to_csv(carpeta / "historia.csv", index=False)
                    modelo = cargar_modelo(receta, resultado.estado, dispositivo)
                    filas = np.concatenate((exterior, seleccion))
                    p, aux = predecir(modelo, pixeles, filas, tta=receta["tta"])
                    cortes = indice.iloc[filas][["sample_id", "patient_id"]].copy()
                    cortes["rol"] = rol_corte[filas]
                    cortes["p"] = p
                    cortes[["a_hr", "a_her2", "a_vol"]] = aux
                    cortes.to_csv(carpeta / "predicciones.csv", index=False)
                    (carpeta / "hecho.json").write_text(json.dumps({
                        "epoca_elegida": resultado.epoca,
                        "epocas_ejecutadas": len(resultado.historia),
                        "segundos": resultado.segundos,
                        "segundos_por_epoca": resultado.segundos / len(resultado.historia),
                        "memoria_max_mb": resultado.memoria_max_mb,
                        "n_fit": int(len(set(indice["patient_id"].iloc[fit]))),
                        "n_seleccion": int(len(set(indice["patient_id"].iloc[seleccion]))),
                        "n_exterior": int(len(set(indice["patient_id"].iloc[exterior]))),
                    }, indent=2))
                    registro(f"   época elegida {resultado.epoca}, {resultado.segundos/60:.1f} min")
        resumir(salida)
    return salida


def _predicciones_brazo(salida: Path, brazo: str, pacientes: pd.DataFrame) -> pd.DataFrame | None:
    """OOF por paciente: media de cortes, luego media de semillas, solo pacientes exteriores."""
    tablas = []
    for fold in sorted(pacientes["fold"].unique()):
        semillas = []
        for ruta in sorted((salida / f"fold{fold}" / brazo).glob("semilla*/predicciones.csv")):
            if not (ruta.parent / "hecho.json").exists():
                continue
            cortes = pd.read_csv(ruta)
            cortes = cortes[cortes["rol"] == "exterior"]
            semillas.append(metricas.por_paciente(cortes, ["p", "a_hr", "a_her2", "a_vol"]))
        if not semillas:
            return None
        media = sum(semillas) / len(semillas)
        media["n_semillas"] = len(semillas)
        tablas.append(media)
    oof = pd.concat(tablas).join(pacientes.set_index("patient_id"), how="left")
    return oof.reset_index()


def _referencia_cohorte(salida: Path, pacientes: pd.DataFrame) -> pd.Series:
    """Puntuación = prevalencia de pCR de la cohorte en fit de cada fold (sin imagen)."""
    puntuaciones = {}
    for fold in sorted(pacientes["fold"].unique()):
        roles = pd.read_csv(salida / f"roles_fold{fold}.csv")
        prevalencia = roles[roles["rol"] == "fit"].groupby("cohorte")["pCR"].mean()
        exterior = roles[roles["rol"] == "exterior"]
        puntuaciones.update(dict(zip(exterior["patient_id"], exterior["cohorte"].map(prevalencia))))
    return pd.Series(puntuaciones, name="p_cohorte")


def resumir(salida: Path) -> dict:
    plan = json.loads((salida / "plan.json").read_text())
    pacientes = datos.tabla_pacientes(datos.cargar_indice("train"))
    folds_listos = [f for f in sorted(pacientes["fold"].unique())
                    if (salida / f"roles_fold{f}.csv").exists()]
    if len(folds_listos) < pacientes["fold"].nunique():
        return {}
    cohorte = _referencia_cohorte(salida, pacientes)
    resumen = {"plan": plan, "brazos": {}}
    columnas = {}
    for perdida in plan["perdidas"]:
        for lambda_aux in plan["lambdas"]:
            brazo = nombre_brazo(perdida, lambda_aux)
            oof = _predicciones_brazo(salida, brazo, pacientes)
            if oof is None:
                continue
            oof["p_cohorte"] = oof["patient_id"].map(cohorte)
            oof.to_csv(salida / f"oof_{brazo}.csv", index=False)
            columnas[brazo] = oof.set_index("patient_id")["p"]
            media, de, por_fold = metricas.auc_media_folds(oof, "p")
            epocas = [json.loads(r.read_text())["epoca_elegida"]
                      for r in (salida).glob(f"fold*/{brazo}/semilla*/hecho.json")]
            curvas = diagnostico_curvas(salida, brazo)
            hechos = [json.loads(r.read_text())
                      for r in (salida).glob(f"fold*/{brazo}/semilla*/hecho.json")]
            dentro = {c: metricas.auc(g["pCR"], g["p"]) for c, g in oof.groupby("cohorte")}
            hr, her2 = oof["HR"].notna(), oof["HER2"].notna()
            resumen["brazos"][brazo] = {
                "auc_media_folds": media, "auc_de_folds": de, "auc_por_fold": por_fold,
                "auc_agrupada": metricas.auc(oof["pCR"], oof["p"]),
                "auc_dentro_cohorte": dentro,
                "auc_dentro_cohorte_media": float(np.mean(list(dentro.values()))),
                "auc_por_subtipo": {s: metricas.auc(g["pCR"], g["p"]) for s, g in
                                    oof.assign(sub=np.select(
                                        [oof["HR"].isna() | oof["HER2"].isna(),
                                         oof["HER2"] == 1, oof["HR"] == 1],
                                        ["desconocido", "HER2+", "HR+/HER2-"], "TN")).groupby("sub")},
                "aux_auc_hr": metricas.auc(oof.loc[hr, "HR"], oof.loc[hr, "a_hr"]),
                "aux_auc_her2": metricas.auc(oof.loc[her2, "HER2"], oof.loc[her2, "a_her2"]),
                "aux_rho_volumen": metricas.spearman(oof["tum_vol"], oof["a_vol"]),
                "epoca_elegida_mediana": float(np.median(epocas)),
                "epocas_elegidas": sorted(epocas),
                "semillas_por_fold": int(oof["n_semillas"].min()),
                "minutos_totales": float(sum(h["segundos"] for h in hechos) / 60),
                "segundos_por_epoca_mediana": float(np.median([h["segundos_por_epoca"] for h in hechos])),
                "memoria_max_mb": float(max(h["memoria_max_mb"] for h in hechos)),
                "frente_a_cohorte": metricas.bootstrap_delta(oof, "p", "p_cohorte"),
                **curvas,
            }
    if not columnas:
        return {}
    base = pacientes.set_index("patient_id")[["pCR", "fold"]]
    base["p_cohorte"] = cohorte
    resumen["referencia_cohorte"] = dict(zip(
        ("auc_media_folds", "auc_de_folds"),
        metricas.auc_media_folds(base.dropna(), "p_cohorte")[:2]))
    comparaciones = {}
    for perdida in plan["perdidas"]:
        con = [nombre_brazo(perdida, l) for l in plan["lambdas"] if l > 0]
        sin = nombre_brazo(perdida, 0.0)
        for brazo in con:
            if brazo in columnas and sin in columnas:
                tabla = base.join(columnas[brazo].rename("a")).join(columnas[sin].rename("b")).dropna()
                comparaciones[f"{brazo} − {sin}"] = metricas.bootstrap_delta(tabla, "a", "b")
    for lambda_aux in plan["lambdas"]:
        a, b = nombre_brazo("ponderada", lambda_aux), nombre_brazo("normal", lambda_aux)
        if a in columnas and b in columnas:
            tabla = base.join(columnas[a].rename("a")).join(columnas[b].rename("b")).dropna()
            comparaciones[f"{a} − {b}"] = metricas.bootstrap_delta(tabla, "a", "b")
    if plan.get("control"):
        for brazo in columnas:
            ruta = RUNS / plan["control"] / f"oof_{brazo}.csv"
            if ruta.exists():
                control = pd.read_csv(ruta).set_index("patient_id")["p"]
                tabla = base.join(columnas[brazo].rename("a")).join(control.rename("b")).dropna()
                comparaciones[f"{brazo} − control {plan['control']}"] = \
                    metricas.bootstrap_delta(tabla, "a", "b")
    resumen["comparaciones"] = comparaciones
    (salida / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    (salida / "RESUMEN.md").write_text(_markdown(resumen), encoding="utf-8")
    _graficar(salida, list(columnas))
    print(_markdown(resumen), flush=True)
    return resumen


def diagnostico_curvas(salida: Path, brazo: str, hasta: int = 14) -> dict:
    """Época del pico de AUC de selección, brechas y tendencia conjunta 1–14.

    Tendencia conjunta (criterio del profesor, ensayo 27): pendiente negativa de
    la BCE con el pos_weight de entrenamiento en fit y en selección entre las
    épocas 1 y ``hasta``, y media de las tres últimas menor que la de 1–3.
    """
    picos, brecha_elegida, brecha_final, cumple, evaluables = [], [], [], 0, 0
    for ruta in sorted(salida.glob(f"fold*/{brazo}/semilla*/historia.csv")):
        historia = pd.read_csv(ruta)
        elegida = json.loads((ruta.parent / "hecho.json").read_text())["epoca_elegida"]
        picos.append(int(historia["sel_auc"].idxmax()) + 1)
        fila = historia.iloc[elegida - 1]
        brecha_elegida.append(fila["fit_auc"] - fila["sel_auc"])
        brecha_final.append(historia["fit_auc"].iloc[-1] - historia["sel_auc"].iloc[-1])
        columnas = [c for c in ("fit_bce_obj", "sel_bce_obj") if c in historia]
        if len(historia) >= hasta and len(columnas) == 2:
            evaluables += 1
            tramo = historia.iloc[:hasta]
            epocas = np.arange(1, hasta + 1)
            cumple += all(np.polyfit(epocas, tramo[c], 1)[0] < 0 and
                          tramo[c].iloc[-3:].mean() < tramo[c].iloc[:3].mean() for c in columnas)
    return {"epoca_pico_sel_mediana": float(np.median(picos)),
            "brecha_auc_elegida_mediana": float(np.median(brecha_elegida)),
            "brecha_auc_final_mediana": float(np.median(brecha_final)),
            "tendencia_conjunta_1_14": f"{cumple}/{evaluables}"}


def _markdown(resumen: dict) -> str:
    lineas = ["# Resumen de la validación cruzada de PartyNet", "",
              "AUC por paciente (media de probabilidades de cortes y de semillas). "
              "El exterior de cada fold no se usó para entrenar ni para parar.", "",
              "| Brazo | Media AUC ± DE (5 folds) | AUC agrupada | AUC dentro de cohorte | "
              "AUC HR / HER2 aux | Época elegida | Pico sel. | Brecha elegida / final | "
              "Tendencia 1–14 | Semillas |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for brazo, r in resumen["brazos"].items():
        lineas.append(
            f"| {brazo} | {r['auc_media_folds']:.4f} ± {r['auc_de_folds']:.4f} | "
            f"{r['auc_agrupada']:.4f} | {r['auc_dentro_cohorte_media']:.4f} | "
            f"{r['aux_auc_hr']:.3f} / {r['aux_auc_her2']:.3f} | "
            f"{r['epoca_elegida_mediana']:g} | {r['epoca_pico_sel_mediana']:g} | "
            f"{r['brecha_auc_elegida_mediana']:.3f} / {r['brecha_auc_final_mediana']:.3f} | "
            f"{r['tendencia_conjunta_1_14']} | {r['semillas_por_fold']} |")
    ref = resumen.get("referencia_cohorte", {})
    if ref:
        lineas += ["", f"Referencia solo cohorte: {ref['auc_media_folds']:.4f} ± {ref['auc_de_folds']:.4f}. "
                   "Referencias históricas (otro protocolo): V007 0,5766; control 27 0,5563."]
    lineas += ["", "| Comparación (Δ media AUC por fold) | Δ | IC95 % pareado | P(Δ≤0) |",
               "|---|---:|---|---:|"]
    for nombre, c in resumen["comparaciones"].items():
        lineas.append(f"| {nombre} | {c['delta']:+.4f} | [{c['ic95'][0]:+.4f}; {c['ic95'][1]:+.4f}] | "
                      f"{c['p_delta_menor_igual_0']:.3f} |")
    for brazo, r in resumen["brazos"].items():
        c = r["frente_a_cohorte"]
        lineas.append(f"| {brazo} − solo cohorte | {c['delta']:+.4f} | "
                      f"[{c['ic95'][0]:+.4f}; {c['ic95'][1]:+.4f}] | {c['p_delta_menor_igual_0']:.3f} |")
    return "\n".join(lineas) + "\n"


def _graficar(salida: Path, brazos: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figura, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
    for brazo in brazos:
        historias = [pd.read_csv(r) for r in salida.glob(f"fold*/{brazo}/semilla*/historia.csv")]
        if not historias:
            continue
        largo = min(len(h) for h in historias)
        for eje, metrica in zip(ejes, ("auc", "bce")):
            for prefijo, estilo in (("fit", "--"), ("sel", "-")):
                valores = np.median([h[f"{prefijo}_{metrica}"].to_numpy()[:largo] for h in historias], axis=0)
                eje.plot(np.arange(1, largo + 1), valores, estilo, label=f"{brazo} · {prefijo}")
    for eje, titulo in zip(ejes, ("AUC por paciente (mediana)", "BCE por paciente (mediana)")):
        eje.set_title(titulo)
        eje.set_xlabel("época (épocas comunes a todas las ejecuciones)")
        eje.grid(alpha=0.3)
    ejes[0].legend(fontsize=7)
    figura.tight_layout()
    figura.savefig(salida / "curvas.png", dpi=130)
    plt.close(figura)
