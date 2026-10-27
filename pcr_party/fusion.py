"""Fusión tardía: puntuación de la CNN congelada + clínica en una regresión logística.

Protocolo en docs/PROTOCOLO_FUSION.md. La CNN no cambia; la clínica entra después.
Con datos clínicos incompletos se usa un submodelo por patrón de variables
disponibles (Fletcher Mercaldo y Blume, 2020), nunca imputación.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from . import datos, metricas
from .cv import RUNS, Registro, entorno, pixeles_en_dispositivo
from .entrenar import cargar_modelo, entrenar, predecir
from .final import MODELOS, sha256

GRUPOS = {"subtipo": ("HR", "HER2"), "edad": ("edad",), "volumen": ("log_volumen",)}
CV_EXTERIOR = "cv_presupuesto"
BRAZO_IMAGEN = "ponderada_aux0"


# ---------------------------------------------------------------- datos clínicos

def tabla_clinica(split: str = "train", raiz: Path = datos.DATOS) -> pd.DataFrame:
    """Una fila por paciente del split: pCR, cohorte y variables clínicas permitidas."""
    if split not in ("train", "test"):
        raise ValueError("Solo existen los splits públicos train y test")
    p = pd.read_csv(raiz / "metadata/patients.csv")
    p = p[p["split"] == split]
    return pd.DataFrame({
        "patient_id": p["pid"].to_numpy(), "pCR": p["pCR"].astype(int).to_numpy(),
        "cohorte": p["dataset"].to_numpy(), "edad": p["age"].to_numpy(dtype=float),
        "log_volumen": np.log1p(p["tum_vol"].to_numpy(dtype=float)),
        "HR": p["HR"].to_numpy(dtype=float), "HER2": p["HER2"].to_numpy(dtype=float),
    }).set_index("patient_id").sort_index()


def combinaciones() -> list[tuple[str, ...]]:
    """Las 8 combinaciones de grupos, de menos a más variables."""
    nombres = list(GRUPOS)
    return [c for n in range(len(nombres) + 1) for c in combinations(nombres, n)]


def nombre_submodelo(grupos: tuple[str, ...], imagen: bool) -> str:
    base = "+".join(grupos) if grupos else ""
    if imagen:
        return f"{base}+imagen" if base else "imagen"
    return base


def variables(grupos: tuple[str, ...], imagen: bool) -> list[str]:
    columnas = [v for g in grupos for v in GRUPOS[g]]
    return columnas + (["imagen"] if imagen else [])


def grupos_disponibles(fila: pd.Series | dict) -> tuple[str, ...]:
    """Grupos con todas sus variables observadas, en el orden canónico."""
    return tuple(g for g, vs in GRUPOS.items()
                 if all(fila.get(v) is not None and not pd.isna(fila.get(v)) for v in vs))


# ---------------------------------------------------------------- logística

def ajustar_logistica(tabla: pd.DataFrame, columnas: list[str], ponderada: bool,
                      interaccion: bool = False) -> dict:
    """L2 (C=1) sobre variables estandarizadas con las filas completas de ``tabla``.

    ``interaccion`` añade imagen × HER2 e imagen × (HR=0) (solo exploratorio).
    """
    filas = tabla.dropna(subset=columnas)
    X = filas[columnas].to_numpy(dtype=float)
    media, escala = X.mean(0), X.std(0)
    escala[escala == 0] = 1.0
    Z = (X - media) / escala
    nombres = list(columnas)
    if interaccion:
        i = columnas.index("imagen")
        Z = np.column_stack([Z, Z[:, i] * filas["HER2"].to_numpy(), Z[:, i] * (1 - filas["HR"].to_numpy())])
        nombres += ["imagen×HER2", "imagen×HR−"]
    modelo = LogisticRegression(C=1.0, max_iter=2000,
                                class_weight="balanced" if ponderada else None)
    modelo.fit(Z, filas["pCR"].to_numpy())
    return {"columnas": list(columnas), "media": media.tolist(), "escala": escala.tolist(),
            "coeficientes": dict(zip(nombres, modelo.coef_[0].tolist())),
            "intercepto": float(modelo.intercept_[0]), "interaccion": interaccion,
            "n_ajuste": int(len(filas)), "pcr_ajuste": int(filas["pCR"].sum())}


def aplicar_logistica(params: dict, tabla: pd.DataFrame) -> np.ndarray:
    X = tabla[params["columnas"]].to_numpy(dtype=float)
    Z = (X - np.asarray(params["media"])) / np.asarray(params["escala"])
    coef = params["coeficientes"]
    z = params["intercepto"] + Z @ np.array([coef[c] for c in params["columnas"]])
    if params["interaccion"]:
        i = params["columnas"].index("imagen")
        z = z + coef["imagen×HER2"] * Z[:, i] * tabla["HER2"].to_numpy() \
              + coef["imagen×HR−"] * Z[:, i] * (1 - tabla["HR"].to_numpy())
    return 1 / (1 + np.exp(-z))


def ajustar_submodelos(tabla: pd.DataFrame, ponderada: bool) -> dict:
    """Todos los submodelos (con y sin imagen) para todas las combinaciones de grupos."""
    submodelos = {}
    for grupos in combinaciones():
        for imagen in (False, True):
            if not grupos and not imagen:
                continue
            submodelos[nombre_submodelo(grupos, imagen)] = ajustar_logistica(
                tabla, variables(grupos, imagen), ponderada)
    return submodelos


def predecir_patron(submodelos: dict, tabla: pd.DataFrame, grupos: tuple[str, ...],
                    imagen: bool) -> tuple[np.ndarray, list[str]]:
    """Brazo ``grupos`` (+imagen): cada paciente usa el submodelo de lo que tiene observado.

    Sin ningún dato aplicable en un brazo solo clínico, la predicción queda en NaN.
    """
    usados = [nombre_submodelo(tuple(g for g in grupos_disponibles(f) if g in grupos), imagen)
              for f in tabla.to_dict("records")]
    p = np.full(len(tabla), np.nan)
    for nombre in set(usados) - {""}:
        filas = np.flatnonzero(np.array(usados) == nombre)
        p[filas] = aplicar_logistica(submodelos[nombre], tabla.iloc[filas])
    return p, usados


# ---------------------------------------------------------------- CV interna de imagen

def roles_internos(pacientes: pd.DataFrame, exterior: int, interior: int,
                   fraccion: float) -> pd.DataFrame:
    """Fold ``exterior`` fuera por completo; se predice ``interior`` con el resto."""
    if exterior == interior:
        raise ValueError("El fold interior debe ser distinto del exterior")
    roles = pacientes[["patient_id", "pCR", "cohorte", "fold"]].copy()
    desarrollo = roles[~roles["fold"].isin([exterior, interior])]
    seleccion = datos.separar_seleccion(desarrollo, fraccion, semilla=3000 + 10 * exterior + interior)
    roles["rol"] = np.select(
        [roles["fold"] == exterior, roles["fold"] == interior, roles["patient_id"].isin(seleccion)],
        ["fuera", "exterior", "seleccion"], "fit")
    return roles


def cv_interna(nombre: str = "cv_interna_fusion", cv_exterior: str = CV_EXTERIOR,
               brazo: str = BRAZO_IMAGEN) -> Path:
    """Las 60 redes de la CV interna (5 exteriores × 4 interiores × 3 semillas). Reanudable."""
    plan_ext = json.loads((RUNS / cv_exterior / "plan.json").read_text())
    receta, semillas = plan_ext["receta"], plan_ext["semillas"]
    perdida = brazo.split("_aux")[0]
    lambda_aux = float(brazo.split("_aux")[1])
    salida = RUNS / nombre
    salida.mkdir(parents=True, exist_ok=True)
    plan = {"cv_exterior": cv_exterior, "brazo": brazo, "receta": receta, "semillas": semillas,
            "entorno": entorno()}
    ruta_plan = salida / "plan.json"
    if ruta_plan.exists() and json.loads(ruta_plan.read_text())["receta"] != receta:
        raise RuntimeError(f"{ruta_plan} tiene otra receta")
    ruta_plan.write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    registro = Registro(salida / "registro.log")

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    indice = datos.cargar_indice("train")
    pacientes = datos.tabla_pacientes(indice)
    pixeles = pixeles_en_dispositivo(indice, dispositivo)
    folds = sorted(pacientes["fold"].unique())
    for semilla in semillas:
        for exterior in folds:
            for interior in folds:
                if interior == exterior:
                    continue
                carpeta = salida / f"exterior{exterior}" / f"interior{interior}" / f"semilla{semilla}"
                if (carpeta / "hecho.json").exists():
                    continue
                carpeta.mkdir(parents=True, exist_ok=True)
                roles = roles_internos(pacientes, exterior, interior, receta["fraccion_seleccion"])
                rol = indice["patient_id"].map(roles.set_index("patient_id")["rol"]).to_numpy()
                fit, sel = np.flatnonzero(rol == "fit"), np.flatnonzero(rol == "seleccion")
                filas = np.flatnonzero(rol == "exterior")
                registro(f"== exterior {exterior} · interior {interior} · semilla {semilla}")
                r = entrenar(receta, pixeles, indice, fit, sel, semilla, perdida, lambda_aux,
                             registro=registro)
                modelo = cargar_modelo(receta, r.estado, dispositivo)
                cortes = indice.iloc[filas][["sample_id", "patient_id"]].copy()
                cortes["p"] = predecir(modelo, pixeles, filas, tta=receta["tta"])[0]
                cortes.to_csv(carpeta / "predicciones.csv", index=False)
                pd.DataFrame(r.historia).to_csv(carpeta / "historia.csv", index=False)
                (carpeta / "hecho.json").write_text(json.dumps({
                    "epoca_elegida": r.epoca, "segundos": r.segundos,
                    "n_fit": int(len(set(indice["patient_id"].iloc[fit]))),
                    "n_seleccion": int(len(set(indice["patient_id"].iloc[sel]))),
                    "n_predichas": int(len(set(indice["patient_id"].iloc[filas]))),
                }, indent=2))
                registro(f"   {r.segundos:.0f} s")
    return salida


def imagen_interna(salida: Path, exterior: int, n_semillas: int) -> pd.Series:
    """Probabilidad de imagen de las pacientes de folds ≠ exterior, sin haber visto ``exterior``."""
    partes = []
    for carpeta in sorted((salida / f"exterior{exterior}").glob("interior*")):
        semillas = [pd.read_csv(r) for r in sorted(carpeta.glob("semilla*/predicciones.csv"))
                    if (r.parent / "hecho.json").exists()]
        if len(semillas) < n_semillas:
            raise RuntimeError(f"{carpeta} tiene {len(semillas)}/{n_semillas} semillas")
        partes.append(sum(metricas.por_paciente(s, ["p"])["p"] for s in semillas) / len(semillas))
    return pd.concat(partes).rename("p_imagen")


# ---------------------------------------------------------------- evaluación anidada

def evaluar(interna: str = "cv_interna_fusion", cv_exterior: str = CV_EXTERIOR,
            brazo: str = BRAZO_IMAGEN) -> dict:
    salida_int = RUNS / interna
    plan = json.loads((salida_int / "plan.json").read_text())
    salida = salida_int / "evaluacion"
    salida.mkdir(exist_ok=True)
    clinica = tabla_clinica("train")
    oof_img = pd.read_csv(RUNS / cv_exterior / f"oof_{brazo}.csv").set_index("patient_id")
    clinica = clinica.join(oof_img[["fold"]], how="inner")
    clinica["imagen_oof"] = metricas.logit(oof_img["p"].reindex(clinica.index))
    if len(clinica) != len(oof_img):
        raise RuntimeError("La OOF de imagen y patients.csv no cubren las mismas pacientes")

    brazos = [(g, i) for g in combinaciones() for i in (False, True) if g or i]
    nombres = [nombre_submodelo(g, i) for g, i in brazos]
    pred = {f"{n}|{p}": pd.Series(np.nan, index=clinica.index) for n in nombres
            for p in ("normal", "ponderada")}
    pred["completa+imagen|normal|interaccion"] = pd.Series(np.nan, index=clinica.index)
    pred["completa+imagen|normal|ingenuo"] = pd.Series(np.nan, index=clinica.index)
    coeficientes = []
    completa = tuple(GRUPOS)
    for k in sorted(clinica["fold"].unique()):
        ajuste = clinica[clinica["fold"] != k].copy()
        prueba = clinica[clinica["fold"] == k].copy()
        ajuste["imagen"] = metricas.logit(imagen_interna(salida_int, k, len(plan["semillas"]))
                                          .reindex(ajuste.index))
        prueba["imagen"] = prueba["imagen_oof"]
        if ajuste["imagen"].isna().any():
            raise RuntimeError(f"Faltan puntuaciones internas en el exterior {k}")
        for perdida in ("normal", "ponderada"):
            submodelos = ajustar_submodelos(ajuste, perdida == "ponderada")
            for (grupos, imagen), nombre in zip(brazos, nombres):
                p, _ = predecir_patron(submodelos, prueba, grupos, imagen)
                pred[f"{nombre}|{perdida}"].loc[prueba.index] = p
            if perdida == "normal":
                c = submodelos["subtipo+edad+volumen+imagen"]["coeficientes"]
                coeficientes.append({"fold": int(k), **c})
        inter = ajustar_logistica(ajuste, variables(completa, True), False, interaccion=True)
        pred["completa+imagen|normal|interaccion"].loc[prueba.index] = aplicar_logistica(
            inter, prueba.fillna({"HR": 0, "HER2": 0}))
        ingenuo = ajuste.assign(imagen=ajuste["imagen_oof"])
        sub_ing = ajustar_submodelos(ingenuo, False)
        pred["completa+imagen|normal|ingenuo"].loc[prueba.index] = predecir_patron(
            sub_ing, prueba, completa, True)[0]

    tabla = clinica[["pCR", "fold", "cohorte", "HR", "HER2"]].copy()
    for clave, serie in pred.items():
        tabla[clave] = serie
    tabla["p_cnn"] = oof_img["p"].reindex(tabla.index)
    tabla.to_csv(salida / "oof_fusion.csv")

    def fila(clave):
        datos_ = tabla.dropna(subset=[clave])
        media, de, folds = metricas.auc_media_folds(datos_, clave)
        return {"auc_media_folds": media, "auc_de_folds": de, "auc_por_fold": folds,
                "auc_agrupada": metricas.auc(datos_["pCR"], datos_[clave]),
                "n": int(len(datos_)), **metricas.resumen_probabilidades(datos_["pCR"], datos_[clave]),
                "auc_dentro_cohorte": {c: metricas.auc(g["pCR"], g[clave]) for c, g in datos_.groupby("cohorte")}}

    alias = {"imagen": "imagen|normal", "subtipo": "subtipo|normal",
             "subtipo+imagen": "subtipo+imagen|normal",
             "completa": "subtipo+edad+volumen|normal",
             "completa+imagen": "subtipo+edad+volumen+imagen|normal",
             "completa+imagen ponderada": "subtipo+edad+volumen+imagen|ponderada",
             "completa ponderada": "subtipo+edad+volumen|ponderada",
             "completa+imagen×subtipo": "completa+imagen|normal|interaccion",
             "completa+imagen ingenuo": "completa+imagen|normal|ingenuo"}
    resumen = {"brazos": {clave: fila(clave) for clave in pred},
               "alias": alias,
               "cnn_sin_recalibrar": fila("p_cnn")}

    def delta(a, b):
        t = tabla.dropna(subset=[alias[a], alias[b]])
        return metricas.bootstrap_delta(t, alias[a], alias[b])

    resumen["comparaciones"] = {
        "principal: completa+imagen − completa": delta("completa+imagen", "completa"),
        "subtipo+imagen − subtipo": delta("subtipo+imagen", "subtipo"),
        "completa+imagen − imagen": delta("completa+imagen", "imagen"),
        "completa+imagen ponderada − normal": delta("completa+imagen ponderada", "completa+imagen"),
        "completa ponderada − normal": delta("completa ponderada", "completa"),
        "exploratoria: completa+imagen×subtipo − completa+imagen":
            delta("completa+imagen×subtipo", "completa+imagen"),
        "diagnóstico: ingenuo − anidado": delta("completa+imagen ingenuo", "completa+imagen"),
    }
    coef = pd.DataFrame(coeficientes).set_index("fold")
    coef.to_csv(salida / "coeficientes_completa_imagen.csv")
    resumen["coeficientes_completa_imagen"] = {
        c: {"media": float(coef[c].mean()), "min": float(coef[c].min()), "max": float(coef[c].max())}
        for c in coef.columns}
    sub = np.select([tabla["HR"].isna() | tabla["HER2"].isna(), tabla["HER2"] == 1, tabla["HR"] == 1],
                    ["desconocido", "HER2+", "HR+/HER2−"], "TN")
    resumen["auc_por_subtipo"] = {
        s: {a: metricas.auc(g["pCR"], g[alias[a]]) for a in ("imagen", "completa", "completa+imagen")}
        for s, g in tabla.assign(sub=sub).groupby("sub") if s != "desconocido"}
    (salida / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    (salida / "RESUMEN.md").write_text(_markdown(resumen), encoding="utf-8")
    _graficar(tabla, alias, salida)
    print(_markdown(resumen), flush=True)
    return resumen


def _markdown(r: dict) -> str:
    a = r["alias"]
    lineas = ["# Fusión tardía imagen + clínica: CV anidada", "",
              "AUC por paciente en las 1.097 pacientes de train; ningún coeficiente vio su fold.", "",
              "| Brazo | Media AUC ± DE (5 folds) | AUC agrupada | AP | Brier |", "|---|---:|---:|---:|---:|"]
    for nombre in ("imagen", "subtipo", "subtipo+imagen", "completa", "completa+imagen",
                   "completa ponderada", "completa+imagen ponderada", "completa+imagen×subtipo",
                   "completa+imagen ingenuo"):
        b = r["brazos"][a[nombre]]
        lineas.append(f"| {nombre} | {b['auc_media_folds']:.4f} ± {b['auc_de_folds']:.4f} | "
                      f"{b['auc_agrupada']:.4f} | {b['ap']:.4f} | {b['brier']:.4f} |")
    lineas += ["", "Todas las combinaciones (pérdida normal):", "",
               "| Submodelo | Media AUC | AP |", "|---|---:|---:|"]
    for clave, b in r["brazos"].items():
        if clave.endswith("|normal"):
            lineas.append(f"| {clave.split('|')[0]} | {b['auc_media_folds']:.4f} | {b['ap']:.4f} |")
    lineas += ["", "| Comparación (Δ media AUC por fold) | Δ | IC95 % pareado | P(Δ≤0) |",
               "|---|---:|---|---:|"]
    for nombre, c in r["comparaciones"].items():
        lineas.append(f"| {nombre} | {c['delta']:+.4f} | [{c['ic95'][0]:+.4f}; {c['ic95'][1]:+.4f}] | "
                      f"{c['p_delta_menor_igual_0']:.3f} |")
    lineas += ["", "Coeficientes estandarizados de completa+imagen (media y rango en los 5 folds):", ""]
    for c, v in r["coeficientes_completa_imagen"].items():
        lineas.append(f"- {c}: {v['media']:+.3f} [{v['min']:+.3f}; {v['max']:+.3f}]")
    lineas += ["", "| Subtipo | Imagen | Completa | Completa+imagen |", "|---|---:|---:|---:|"]
    for s, v in r["auc_por_subtipo"].items():
        lineas.append(f"| {s} | {v['imagen']:.3f} | {v['completa']:.3f} | {v['completa+imagen']:.3f} |")
    return "\n".join(lineas) + "\n"


def _graficar(tabla: pd.DataFrame, alias: dict, salida: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fondo = "#fcfcfb"
    plt.rcParams.update({"figure.facecolor": fondo, "axes.facecolor": fondo, "savefig.facecolor": fondo,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#ecebe7", "lines.linewidth": 2})
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    for nombre, color in (("imagen", "#2a78d6"), ("subtipo", "#eda100"),
                          ("completa", "#8a8984"), ("completa+imagen", "#eb6834")):
        t = tabla.dropna(subset=[alias[nombre]])
        fpr, tpr, _ = roc_curve(t["pCR"], t[alias[nombre]])
        media = metricas.auc_media_folds(t, alias[nombre])[0]
        ax.plot(fpr, tpr, color=color, label=f"{nombre} (AUC media {media:.3f})")
    ax.plot([0, 1], [0, 1], ":", color="#8a8984", lw=1)
    ax.set(xlabel="1 − especificidad", ylabel="sensibilidad", xlim=(0, 1), ylim=(0, 1), aspect="equal",
           title="ROC OOF anidada (1.097 pacientes de train)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(salida / "roc_fusion.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- paquete final

def paquete_final(nombre: str = "party_fusion_v1", imagen: str = "party_final_v2",
                  interna: str = "cv_interna_fusion", cv_exterior: str = CV_EXTERIOR,
                  brazo: str = BRAZO_IMAGEN) -> Path:
    salida = MODELOS / nombre
    if salida.exists():
        raise FileExistsError(f"{salida} ya existe; los paquetes no se sobrescriben")
    ruta_imagen = MODELOS / imagen
    info_imagen = json.loads((ruta_imagen / "paquete.json").read_text())
    if info_imagen["cv_origen"] != f"runs/{cv_exterior}":
        raise RuntimeError("El paquete de imagen no procede de la CV exterior indicada")
    evaluacion = RUNS / interna / "evaluacion"
    oof = pd.read_csv(evaluacion / "oof_fusion.csv").set_index("patient_id")
    resumen = json.loads((evaluacion / "resumen.json").read_text())

    clinica = tabla_clinica("train")
    oof_img = pd.read_csv(RUNS / cv_exterior / f"oof_{brazo}.csv").set_index("patient_id")
    clinica["imagen"] = metricas.logit(oof_img["p"].reindex(clinica.index))
    submodelos = ajustar_submodelos(clinica, ponderada=False)
    for clave, params in submodelos.items():
        columna = f"{clave}|normal"
        t = oof.dropna(subset=[columna])
        youden = metricas.umbral_youden(t["pCR"], t[columna])
        macro = metricas.umbral_macro_f1(t["pCR"], t[columna])
        params["politica"] = {
            "umbral": youden, "regla_umbral": "Youden máximo sobre OOF anidadas (solo train)",
            "umbral_macro_f1_alternativo": macro,
            "oof_con_umbral": metricas.binarias(t["pCR"], t[columna], youden),
            "oof_con_umbral_macro_f1": metricas.binarias(t["pCR"], t[columna], macro),
            "oof_auc_media_folds": resumen["brazos"][columna]["auc_media_folds"]}
    salida.mkdir(parents=True)
    paquete = {
        "nombre": nombre, "tipo": "fusion",
        "descripcion": "Regresión logística por patrón de datos clínicos sobre la puntuación de la CNN",
        "paquete_imagen": imagen,
        "sha256_paquete_imagen": sha256(ruta_imagen / "paquete.json"),
        "grupos": {g: list(v) for g, v in GRUPOS.items()},
        "variable_imagen": "logit de la probabilidad cruda de paciente de la CNN (media de cortes y semillas)",
        "volumen": "log(1 + tum_vol) en las unidades de patients.csv",
        "perdida": "normal (comparada con ponderada en la CV anidada)",
        "submodelos": submodelos,
        "cv_anidada": str(evaluacion.relative_to(datos.RAIZ)),
        "comparaciones": resumen["comparaciones"],
        "n_pacientes_train": int(len(clinica)), "test_evaluado": False, "entorno": entorno(),
    }
    (salida / "paquete.json").write_text(json.dumps(paquete, indent=2, ensure_ascii=False))
    return salida


class PaqueteFusion:
    def __init__(self, ruta: Path, dispositivo: str | None = None):
        from .inferencia import Paquete

        self.ruta = Path(ruta)
        self.info = json.loads((self.ruta / "paquete.json").read_text())
        ruta_imagen = self.ruta.parent / self.info["paquete_imagen"]
        if sha256(ruta_imagen / "paquete.json") != self.info["sha256_paquete_imagen"]:
            raise RuntimeError("El paquete de imagen no coincide con el que se usó al ajustar")
        self.imagen = Paquete(ruta_imagen, dispositivo)

    def decidir(self, p_imagen: float, clinica: dict) -> dict:
        """``clinica``: edad, volumen (tum_vol), HR, HER2; los ausentes como None."""
        fila = {"edad": clinica.get("edad"), "HR": clinica.get("HR"), "HER2": clinica.get("HER2"),
                "log_volumen": None if clinica.get("volumen") is None else float(np.log1p(clinica["volumen"])),
                "imagen": float(metricas.logit(p_imagen))}
        grupos = grupos_disponibles(fila)
        if not grupos:
            salida = self.imagen.decidir(p_imagen)
            salida["submodelo"] = "imagen sola (paquete de imagen)"
            return salida
        nombre = nombre_submodelo(grupos, True)
        params = self.info["submodelos"][nombre]
        p = float(aplicar_logistica(params, pd.DataFrame([fila]))[0])
        umbral = params["politica"]["umbral"]
        return {"probabilidad_pcr": p, "umbral": umbral, "clase": "pCR" if p >= umbral else "no pCR",
                "submodelo": nombre, "probabilidad_solo_imagen": self.imagen.decidir(p_imagen)["probabilidad_pcr"],
                "modelo": self.info["nombre"]}


def evaluar_test(ruta: Path, forzar: bool = False) -> dict:
    """Evaluación única del paquete de fusión sobre test, con la clínica de patients.csv."""
    paquete = PaqueteFusion(ruta)
    marca = paquete.ruta / "evaluacion_test.json"
    if marca.exists() and not forzar:
        raise RuntimeError(f"{marca} ya existe: el test de este paquete ya se evaluó")
    indice = datos.cargar_indice("test")
    cortes = indice[["patient_id"]].copy()
    cortes["p"] = paquete.imagen.probabilidad_cortes(datos.cargar_imagenes(indice, cache=None))
    clinica = tabla_clinica("test").join(cortes.groupby("patient_id")["p"].mean().rename("p_imagen"))
    filas = []
    for pid, f in clinica.iterrows():
        d = paquete.decidir(f["p_imagen"], {
            "edad": None if pd.isna(f["edad"]) else f["edad"],
            "volumen": None if pd.isna(f["log_volumen"]) else float(np.expm1(f["log_volumen"])),
            "HR": None if pd.isna(f["HR"]) else f["HR"], "HER2": None if pd.isna(f["HER2"]) else f["HER2"]})
        filas.append({"patient_id": pid, "pCR": f["pCR"], "cohorte": f["cohorte"], "p": d["probabilidad_pcr"],
                      "umbral": d["umbral"], "submodelo": d["submodelo"], "p_imagen": f["p_imagen"]})
    t = pd.DataFrame(filas).set_index("patient_id")
    pred = (t["p"] >= t["umbral"]).astype(int)
    resultado = {
        "paquete": paquete.info["nombre"], "pacientes": len(t), "pcr": int(t["pCR"].sum()),
        "auc": metricas.auc(t["pCR"], t["p"]), "auc_solo_imagen": metricas.auc(t["pCR"], t["p_imagen"]),
        "submodelos_usados": t["submodelo"].value_counts().to_dict(),
        "TP": int(((pred == 1) & (t["pCR"] == 1)).sum()), "FP": int(((pred == 1) & (t["pCR"] == 0)).sum()),
        "TN": int(((pred == 0) & (t["pCR"] == 0)).sum()), "FN": int(((pred == 0) & (t["pCR"] == 1)).sum()),
        "auc_por_cohorte": {c: metricas.auc(g["pCR"], g["p"]) for c, g in t.groupby("cohorte")},
    }
    t.to_csv(paquete.ruta / "predicciones_test.csv")
    marca.write_text(json.dumps(resultado, indent=2, ensure_ascii=False))
    return resultado
