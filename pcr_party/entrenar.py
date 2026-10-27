"""Bucle de entrenamiento de PartyNet, común a la CV y al modelo final."""
from __future__ import annotations

import math
import time
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import metricas
from .modelo import PartyNet

RECETA = {
    "modelo": {"anchos": [24, 48, 96, 160], "oculto": 64, "dropout": 0.35,
               "dropout_espacial": 0.10, "escalas": [256, 128, 64], "lado": 128},
    "optimizador": "AdamW",
    "lr": 3e-4,
    "weight_decay": 0.01,
    "lote": 64,
    "epocas_max": 40,
    "warmup_epocas": 1,
    "paciencia": 8,
    "epocas_min": 6,
    "ventana_auc": 3,
    "ema": 0.995,
    "clip_grad": 1.0,
    "afin": {"escala": 0.05, "traslacion": 0.03},
    "fraccion_seleccion": 0.15,
    "lambda_aux": 1.0,
    "tta": True,
    "pacientes_diagnostico_fit": 150,
}

# Receta lenta: retrasa la memorización. Una época ve unos pocos cortes
# aleatorios de cada paciente (no los ~10 casi idénticos), con LR menor y
# regularización más fuerte. Las claves nuevas tienen valor neutro en RECETA.
RECETA_LENTA = deepcopy(RECETA)
RECETA_LENTA.update({
    "modelo": dict(RECETA["modelo"], dropout=0.5),
    # Piloto fold 0, semilla 42, solo selección: 1e-4 → pico en la época 21;
    # 2e-4 → pico en la 11, dentro del objetivo 10–15.
    "lr": 2e-4,
    "weight_decay": 0.05,
    "epocas_max": 60,
    "warmup_epocas": 2,
    "paciencia": 10,
    "epocas_min": 14,
    "ema": 0.98,
    "afin": {"escala": 0.10, "traslacion": 0.08, "rotacion_grados": 15},
    "cutout": {"probabilidad": 0.5, "lado": 0.25},
    "label_smoothing": 0.1,
    "cortes_por_paciente": 3,
    "lambda_aux": 0.0,
})
# Receta media: la regularización del control (que discrimina mejor) con el
# muestreo de la lenta. La CV lenta mostró curvas sanas pero infraajuste
# (AUC ≈ cohorte); aquí solo se cambia qué ve cada época, no cuánto se penaliza.
RECETA_MEDIA = deepcopy(RECETA)
RECETA_MEDIA.update({
    "epocas_max": 60,
    "warmup_epocas": 2,
    "paciencia": 10,
    "epocas_min": 14,
    "ema": 0.98,
    "afin": {"escala": 0.05, "traslacion": 0.03, "rotacion_grados": 10},
    "cortes_por_paciente": 3,
    "lambda_aux": 0.0,
})
# Receta de presupuesto: la media con un calendario que llega a LR 0 en la
# época 16. En la media, el coseno de 60 épocas mantenía el LR alto cuando la
# red empezaba a memorizar; aquí converge dentro del presupuesto y el mejor
# punto debería caer al final, hacia la época 12–16.
RECETA_PRESUPUESTO = deepcopy(RECETA_MEDIA)
RECETA_PRESUPUESTO.update({
    # Pilotos fold 0, semilla 42, normal, solo selección: con 16 épocas la BCE
    # de selección toca mínimo en la 11 (lineal y coseno) y la AUC tiene su
    # meseta en la 9–13. Presupuesto 12 y se conserva la última época, sin
    # depender del ruido de la AUC de 131 pacientes.
    "epocas_max": 12,
    "epocas_min": 12,
    "paciencia": 12,
    "calendario": "lineal",
    "checkpoint": "ultima",
})
RECETAS = {"control": RECETA, "lenta": RECETA_LENTA, "media": RECETA_MEDIA,
           "presupuesto": RECETA_PRESUPUESTO}


@dataclass
class Resultado:
    estado: dict
    epoca: int
    historia: list[dict] = field(default_factory=list)
    segundos: float = 0.0
    memoria_max_mb: float = 0.0


class EMA:
    """Media exponencial de pesos y buffers; los enteros se copian."""

    def __init__(self, modelo: torch.nn.Module, decay: float):
        self.modelo = deepcopy(modelo).eval()
        for p in self.modelo.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.pasos = 0

    @torch.no_grad()
    def actualizar(self, modelo: torch.nn.Module) -> None:
        self.pasos += 1
        decay = min(self.decay, (1 + self.pasos) / (10 + self.pasos))
        for media, actual in zip(self.modelo.state_dict().values(), modelo.state_dict().values()):
            if media.dtype.is_floating_point:
                media.lerp_(actual, 1 - decay)
            else:
                media.copy_(actual)


def aumentar(x: torch.Tensor, generador: torch.Generator, afin: dict | None,
             cutout: dict | None = None) -> torch.Tensor:
    """Grupo diédrico, afín (escala, traslación, giro) y cutout, idénticos en las tres fases."""
    n, dispositivo = len(x), x.device
    giro = torch.randint(0, 4, (n,), generator=generador, device=dispositivo)
    volteo = torch.rand(n, generator=generador, device=dispositivo) < 0.5
    x = torch.where(volteo[:, None, None, None], x.flip(-1), x)
    girada = x.clone()
    for k in (1, 2, 3):
        mascara = giro == k
        if mascara.any():
            girada[mascara] = torch.rot90(x[mascara], k, dims=(-2, -1))
    x = girada
    if afin:
        escala = 1 + (torch.rand(n, generator=generador, device=dispositivo) * 2 - 1) * afin["escala"]
        desplazamiento = (torch.rand(n, 2, generator=generador, device=dispositivo) * 2 - 1) \
            * afin["traslacion"] * 2
        angulo = torch.zeros(n, device=dispositivo)
        if afin.get("rotacion_grados", 0):
            # Solo se consume azar si hay giro: el control conserva su secuencia exacta.
            angulo = (torch.rand(n, generator=generador, device=dispositivo) * 2 - 1) \
                * math.radians(afin["rotacion_grados"])
        coseno, seno = torch.cos(angulo) / escala, torch.sin(angulo) / escala
        theta = torch.zeros(n, 2, 3, device=dispositivo)
        theta[:, 0, 0], theta[:, 0, 1] = coseno, -seno
        theta[:, 1, 0], theta[:, 1, 1] = seno, coseno
        theta[:, :, 2] = desplazamiento
        malla = F.affine_grid(theta, list(x.shape), align_corners=False)
        x = F.grid_sample(x, malla, mode="bilinear", padding_mode="border", align_corners=False)
    if cutout:
        alto, ancho = x.shape[-2:]
        lado = int(round(cutout["lado"] * alto))
        activo = torch.rand(n, generator=generador, device=dispositivo) < cutout["probabilidad"]
        fila = torch.randint(0, alto - lado + 1, (n,), generator=generador, device=dispositivo)
        columna = torch.randint(0, ancho - lado + 1, (n,), generator=generador, device=dispositivo)
        filas = torch.arange(alto, device=dispositivo)[None, :]
        columnas = torch.arange(ancho, device=dispositivo)[None, :]
        dentro_fila = (filas >= fila[:, None]) & (filas < fila[:, None] + lado)
        dentro_columna = (columnas >= columna[:, None]) & (columnas < columna[:, None] + lado)
        parche = dentro_fila[:, :, None] & dentro_columna[:, None, :] & activo[:, None, None]
        x = x.masked_fill(parche[:, None], 0.0)
    return x


def muestrear_cortes(filas_fit: np.ndarray, pacientes: np.ndarray, k: int | None,
                     rng: np.random.Generator) -> np.ndarray:
    """Hasta ``k`` cortes aleatorios distintos por paciente; ``None`` = todos."""
    if k is None:
        return filas_fit
    tabla = pd.DataFrame({"paciente": pacientes, "fila": filas_fit,
                          "azar": rng.random(len(filas_fit))})
    return tabla.sort_values(["paciente", "azar"]).groupby("paciente").head(k)["fila"].to_numpy(copy=True)


def objetivos_auxiliares(indice: pd.DataFrame, fit_pacientes: set[str]) -> np.ndarray:
    """[N,3]: HR, HER2 y log-volumen estandarizado con pacientes de fit. NaN = ausente."""
    log_volumen = np.log1p(indice["tum_vol"].to_numpy(float))
    fit = indice[indice["patient_id"].isin(fit_pacientes)].drop_duplicates("patient_id")
    referencia = np.log1p(fit["tum_vol"].to_numpy(float))
    media, desviacion = np.nanmean(referencia), np.nanstd(referencia)
    return np.stack((indice["HR"].to_numpy(float), indice["HER2"].to_numpy(float),
                     (log_volumen - media) / desviacion), axis=1).astype(np.float32)


def logit_pcr(salida: torch.Tensor, desplazamiento: torch.Tensor | None = None,
              grupo: torch.Tensor | None = None) -> torch.Tensor:
    """[N,K] → [N]: cabeza del grupo de cada corte (media si grupo < 0) más el offset."""
    if salida.shape[1] == 1:
        logit = salida.squeeze(1)
    else:
        if grupo is None:
            raise ValueError("Con varias cabezas hay que indicar el grupo de cada corte")
        elegida = salida.gather(1, grupo.clamp(min=0)[:, None]).squeeze(1)
        logit = torch.where(grupo >= 0, elegida, salida.mean(1))
    return logit if desplazamiento is None else logit + desplazamiento


def _por_fila(valores: np.ndarray | None, dtype, dispositivo) -> torch.Tensor | None:
    return None if valores is None else torch.as_tensor(np.asarray(valores, dtype=dtype), device=dispositivo)


@torch.no_grad()
def predecir(modelo: PartyNet, pixeles: torch.Tensor, filas: np.ndarray,
             tta: bool = True, lote: int = 256, desplazamiento: np.ndarray | None = None,
             grupo: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Probabilidad de pCR y salidas auxiliares por corte, en FP32 y modo eval.

    Con TTA se promedian las probabilidades de las cuatro reflexiones.
    ``desplazamiento`` (offset clínico) y ``grupo`` van alineados con las filas
    de ``pixeles``, no con ``filas``.
    """
    modelo.eval()
    dispositivo = next(modelo.parameters()).device
    offset_t = _por_fila(desplazamiento, np.float32, dispositivo)
    grupo_t = _por_fila(grupo, np.int64, dispositivo)
    probabilidades, auxiliares = [], []
    for inicio in range(0, len(filas), lote):
        indices = torch.as_tensor(filas[inicio:inicio + lote], device=pixeles.device)
        x = pixeles[indices].to(dispositivo).float().div_(255)
        i = indices.to(dispositivo)
        o = None if offset_t is None else offset_t[i]
        g = None if grupo_t is None else grupo_t[i]
        variantes = [x, x.flip(-1), x.flip(-2), x.flip(-2).flip(-1)] if tta else [x]
        suma_p, suma_a = 0, 0
        for variante in variantes:
            salida, aux = modelo.forward_todo(variante)
            suma_p = suma_p + torch.sigmoid(logit_pcr(salida, o, g))
            suma_a = suma_a + aux
        probabilidades.append((suma_p / len(variantes)).cpu())
        auxiliares.append((suma_a / len(variantes)).cpu())
    return torch.cat(probabilidades).numpy(), torch.cat(auxiliares).numpy()


def _evaluar(modelo, pixeles, indice, filas, prefijo: str, pos_weight: float = 1.0,
             desplazamiento=None, grupo=None) -> dict:
    p, aux = predecir(modelo, pixeles, filas, tta=False, desplazamiento=desplazamiento, grupo=grupo)
    cortes = indice.iloc[filas][["patient_id", "pCR", "HR", "HER2", "tum_vol"]].copy()
    cortes["p"] = p
    cortes[["a_hr", "a_her2", "a_vol"]] = aux
    pac = cortes.groupby("patient_id").agg(
        pCR=("pCR", "first"), HR=("HR", "first"), HER2=("HER2", "first"),
        tum_vol=("tum_vol", "first"), p=("p", "mean"), a_hr=("a_hr", "mean"),
        a_her2=("a_her2", "mean"), a_vol=("a_vol", "mean"))
    hr = pac["HR"].notna()
    her2 = pac["HER2"].notna()
    probabilidad = np.clip(pac["p"].to_numpy(), metricas.EPS, 1 - metricas.EPS)
    objetivo = pac["pCR"].to_numpy()
    return {
        f"{prefijo}_bce": metricas.bce(pac["pCR"], pac["p"]),
        # BCE con el mismo pos_weight que optimiza el entrenamiento (diagnóstico de curvas).
        f"{prefijo}_bce_obj": float(-np.mean(pos_weight * objetivo * np.log(probabilidad)
                                             + (1 - objetivo) * np.log(1 - probabilidad))),
        f"{prefijo}_auc": metricas.auc(pac["pCR"], pac["p"]),
        f"{prefijo}_auc_hr": metricas.auc(pac.loc[hr, "HR"], pac.loc[hr, "a_hr"]),
        f"{prefijo}_auc_her2": metricas.auc(pac.loc[her2, "HER2"], pac.loc[her2, "a_her2"]),
        f"{prefijo}_rho_vol": metricas.spearman(pac["tum_vol"], pac["a_vol"]),
    }


def entrenar(receta: dict, pixeles: torch.Tensor, indice: pd.DataFrame,
             fit: np.ndarray, seleccion: np.ndarray | None, semilla: int,
             perdida: str, lambda_aux: float, epocas: int | None = None,
             registro=print, desplazamiento: np.ndarray | None = None,
             grupo: np.ndarray | None = None) -> Resultado:
    """Entrena una red con BCEWithLogitsLoss por corte y peso 1/n_cortes por paciente.

    ``desplazamiento`` (una por fila de ``indice``) es un offset fijo que se
    suma al logit: la red aprende solo el residuo (p. ej. de la clínica).
    ``grupo`` elige la cabeza de cada corte cuando ``receta["modelo"]`` tiene
    varias (−1 = grupo desconocido, media de las cabezas).

    ``seleccion`` decide la parada temprana con la AUC de paciente de los pesos
    EMA, suavizada con una media móvil de ``ventana_auc`` épocas. No se usa la
    BCE de selección: con ``pos_weight`` las probabilidades se descalibran a
    propósito y la BCE empeora aunque el orden mejore. Sin selección (modelo
    final) se entrena exactamente ``epocas`` épocas con el mismo calendario de
    LR que en la CV.
    """
    if perdida not in ("normal", "ponderada"):
        raise ValueError("La pérdida debe ser normal o ponderada")
    if seleccion is None and epocas is None:
        raise ValueError("Sin selección hay que fijar el número de épocas")
    dispositivo = pixeles.device
    cuda = dispositivo.type == "cuda"
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    if cuda:
        torch.cuda.reset_peak_memory_stats(dispositivo)
    modelo = PartyNet(**receta["modelo"]).to(dispositivo)

    fit_pacientes = set(indice["patient_id"].iloc[fit])
    tabla_fit = indice.iloc[fit].drop_duplicates("patient_id")
    n1 = int(tabla_fit["pCR"].sum())
    n0 = len(tabla_fit) - n1
    pos_weight = n0 / n1 if perdida == "ponderada" else 1.0
    if desplazamiento is not None and perdida != "normal":
        raise ValueError("Con offset la base ya está calibrada: usa la pérdida normal")
    with torch.no_grad():
        # Con offset, la prevalencia ya está en el desplazamiento.
        modelo.sesgo.fill_(0.0 if desplazamiento is not None else math.log(n1 * pos_weight / n0))
    offset_t = _por_fila(desplazamiento, np.float32, dispositivo)
    grupo_t = _por_fila(grupo, np.int64, dispositivo)

    y = torch.tensor(indice["pCR"].to_numpy(np.float32), device=dispositivo)
    aux = torch.tensor(objetivos_auxiliares(indice, fit_pacientes), device=dispositivo)
    k = receta.get("cortes_por_paciente")
    n_cortes = indice.groupby("patient_id")["patient_id"].transform("size").to_numpy()
    if k is not None:
        n_cortes = np.minimum(n_cortes, k)
    peso = torch.tensor((1.0 / n_cortes).astype(np.float32), device=dispositivo)
    suavizado = receta.get("label_smoothing", 0.0)
    y_objetivo = y * (1 - suavizado) + suavizado / 2
    peso_positivo = torch.tensor(pos_weight, device=dispositivo)

    optimizador = torch.optim.AdamW(modelo.parameters(), lr=receta["lr"],
                                    weight_decay=receta["weight_decay"])
    pacientes_fit = indice["patient_id"].to_numpy()[fit]
    rng_muestreo = np.random.default_rng(semilla + 7)
    cortes_por_epoca = len(muestrear_cortes(fit, pacientes_fit, k, np.random.default_rng(0)))
    pasos_epoca = math.ceil(cortes_por_epoca / receta["lote"])
    total = receta["epocas_max"] * pasos_epoca
    calentamiento = receta["warmup_epocas"] * pasos_epoca

    # «lineal»: decaimiento lineal hasta 0 al agotar el presupuesto de épocas
    # (Li, Yumer y Ramanan, Budgeted Training, ICLR 2020).
    lineal = receta.get("calendario", "coseno") == "lineal"

    def factor(paso: int) -> float:
        if paso < calentamiento:
            return (paso + 1) / calentamiento
        progreso = (paso - calentamiento) / max(1, total - calentamiento)
        return max(0.0, 1 - progreso) if lineal else 0.5 * (1 + math.cos(math.pi * progreso))

    planificador = torch.optim.lr_scheduler.LambdaLR(optimizador, factor)
    ema = EMA(modelo, receta["ema"])
    generador = torch.Generator(device=dispositivo)
    generador.manual_seed(semilla)

    # Subconjunto fijo de fit, sin aumentos, para medir la brecha de sobreajuste.
    rng = np.random.default_rng(semilla)
    ids_diag = rng.permutation(sorted(fit_pacientes))[:receta["pacientes_diagnostico_fit"]]
    diag = np.flatnonzero(indice["patient_id"].isin(set(ids_diag)).to_numpy())

    n_epocas = receta["epocas_max"] if epocas is None else epocas
    historia: list[dict] = []
    mejor: dict | None = None
    sin_mejora = 0
    inicio_total = time.time()
    for epoca in range(1, n_epocas + 1):
        modelo.train()
        inicio = time.time()
        acumulada, vistos = 0.0, 0
        filas_epoca = torch.as_tensor(muestrear_cortes(fit, pacientes_fit, k, rng_muestreo),
                                      device=dispositivo)
        orden = filas_epoca[torch.randperm(len(filas_epoca), generator=generador, device=dispositivo)]
        for i in range(0, len(orden), receta["lote"]):
            lote = orden[i:i + receta["lote"]]
            if len(lote) < 2:
                continue
            x = aumentar(pixeles[lote].float().div_(255), generador, receta["afin"],
                         receta.get("cutout"))
            with torch.autocast(dispositivo.type, dtype=torch.bfloat16, enabled=cuda):
                salida, salida_aux = modelo.forward_todo(x)
            logit = logit_pcr(salida.float(), None if offset_t is None else offset_t[lote],
                              None if grupo_t is None else grupo_t[lote])
            salida_aux = salida_aux.float()
            w = peso[lote] / peso[lote].mean()
            perdida_pcr = (F.binary_cross_entropy_with_logits(
                logit, y_objetivo[lote], pos_weight=peso_positivo, reduction="none") * w).mean()
            total_lote = perdida_pcr
            if lambda_aux > 0:
                objetivo = aux[lote]
                terminos = []
                for j in range(3):
                    valido = ~torch.isnan(objetivo[:, j])
                    if not valido.any():
                        continue
                    if j < 2:
                        t = F.binary_cross_entropy_with_logits(
                            salida_aux[valido, j], objetivo[valido, j], reduction="none")
                    else:
                        t = F.smooth_l1_loss(salida_aux[valido, j], objetivo[valido, j],
                                             reduction="none")
                    terminos.append((t * w[valido]).sum() / w[valido].sum())
                if terminos:
                    total_lote = total_lote + lambda_aux * torch.stack(terminos).sum() / 3
            optimizador.zero_grad(set_to_none=True)
            total_lote.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), receta["clip_grad"])
            optimizador.step()
            planificador.step()
            ema.actualizar(modelo)
            acumulada += float(perdida_pcr.detach()) * len(lote)
            vistos += len(lote)

        fila = {"epoca": epoca, "bce_pcr_fit_aumentada": acumulada / max(1, vistos),
                "lr_final": planificador.get_last_lr()[0],
                "segundos_entrenamiento": time.time() - inicio}
        fila.update(_evaluar(ema.modelo, pixeles, indice, diag, "fit", pos_weight, desplazamiento, grupo))
        if seleccion is not None:
            fila.update(_evaluar(ema.modelo, pixeles, indice, seleccion, "sel", pos_weight,
                                 desplazamiento, grupo))
            recientes = [h["sel_auc"] for h in historia[-(receta["ventana_auc"] - 1):]] + [fila["sel_auc"]]
            fila["sel_auc_suavizada"] = float(np.mean(recientes))
            candidata = epoca >= min(receta["ventana_auc"], n_epocas)
            if candidata and (mejor is None or fila["sel_auc_suavizada"] > mejor["auc"]):
                mejor = {"auc": fila["sel_auc_suavizada"], "epoca": epoca,
                         "estado": deepcopy(ema.modelo.state_dict())}
                sin_mejora = 0
            else:
                sin_mejora += 1
        fila["segundos_epoca"] = time.time() - inicio
        historia.append(fila)
        registro(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in fila.items()))
        if seleccion is not None and epoca >= receta["epocas_min"] and \
                sin_mejora >= receta["paciencia"]:
            break

    if seleccion is None or receta.get("checkpoint") == "ultima":
        # «ultima»: con LR que llega a 0 el modelo converge al agotar el
        # presupuesto; la selección solo se registra para las curvas.
        estado, epoca_elegida = deepcopy(ema.modelo.state_dict()), len(historia)
    else:
        estado, epoca_elegida = mejor["estado"], mejor["epoca"]
    memoria = torch.cuda.max_memory_allocated(dispositivo) / 2**20 if cuda else 0.0
    return Resultado(estado={k: v.cpu() for k, v in estado.items()}, epoca=epoca_elegida,
                     historia=historia, segundos=time.time() - inicio_total,
                     memoria_max_mb=memoria)


def cargar_modelo(receta: dict, estado: dict, dispositivo) -> PartyNet:
    modelo = PartyNet(**receta["modelo"])
    modelo.load_state_dict(estado, strict=True)
    return modelo.to(dispositivo).eval()
