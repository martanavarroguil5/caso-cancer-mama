"""Lógica de la web sin Streamlit: paquetes, lectura de PNG, predicción, imágenes y recorrido por la red."""
from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

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


# ---------------------------------------------------------------- imágenes para la web

FONDO, REALCE, FRIO = np.array([7, 9, 12]), np.array([255, 106, 61]), np.array([63, 193, 220])


def url_imagen(arr: np.ndarray) -> str:
    """uint8 gris [H,W], RGB o RGBA [H,W,C] → data URL (WebP si es grande, PNG si es diminuta)."""
    imagen = Image.fromarray(np.ascontiguousarray(arr))
    buf = io.BytesIO()
    if min(arr.shape[:2]) >= 48:
        imagen.save(buf, "WEBP", quality=86, method=4)
        tipo = "webp"
    else:
        imagen.save(buf, "PNG", optimize=True)
        tipo = "png"
    return f"data:image/{tipo};base64,{base64.b64encode(buf.getvalue()).decode()}"


def divergente(d: np.ndarray, escala: float) -> np.ndarray:
    """Cian (negativo) ← negro → naranja (positivo), como RGB uint8."""
    t = np.clip(d / max(escala, 1e-6), -1, 1)[..., None]
    color = np.where(t > 0, REALCE, FRIO)
    return (FONDO + (color - FONDO) * np.abs(t)).astype(np.uint8)


def magma(a: np.ndarray) -> np.ndarray:
    from matplotlib import colormaps
    return (colormaps["magma"](np.clip(a, 0, 1))[..., :3] * 255).astype(np.uint8)


def realce_rgba(corte: np.ndarray, escala: float) -> np.ndarray:
    """EARLY − PRE positivo como capa RGBA caliente (transparente donde no capta)."""
    t = np.clip((corte[1].astype(float) - corte[0].astype(float)) / max(escala, 1.0), 0, 1)
    rampa = np.array([[122, 29, 14], [255, 106, 61], [255, 208, 138]], dtype=float)  # rojo → naranja → ámbar
    u = np.clip(t * 2, 0, 2)[..., None]
    rgb = np.where(u < 1, rampa[0] + (rampa[1] - rampa[0]) * u, rampa[1] + (rampa[2] - rampa[1]) * (u - 1))
    alfa = np.clip(t * 1.6, 0, 1) ** 0.8 * 255
    return np.concatenate([rgb, alfa[..., None]], axis=-1).astype(np.uint8)


def escala_realce(pixeles: np.ndarray) -> float:
    d = pixeles[:, 1].astype(float) - pixeles[:, 0].astype(float)
    return float(max(np.percentile(d[d > 0], 99.5) if (d > 0).any() else 1.0, 1.0))


def visor(pixeles: np.ndarray, nombres: list[str], p_cortes: list[float]) -> list[dict]:
    """Fases y mapa de realce de cada corte para el visor de la web."""
    escala = escala_realce(pixeles)
    return [{"nombre": n, "p": float(p), **{f: url_imagen(c[k]) for k, f in enumerate(datos.FASES)},
             "REALCE": url_imagen(realce_rgba(c, escala))}
            for c, n, p in zip(pixeles, nombres, p_cortes)]


def portada() -> dict:
    """Un corte de ejemplo con buen realce para la cabecera de inicio."""
    ej = ejemplos()
    pid = ej[(ej["pCR"] == 1) & (ej["cohorte"] == "spy2")].index[0]
    pix, nombres = cortes_de_ejemplo(pid)
    d = pix[:, 1].astype(float) - pix[:, 0].astype(float)
    i = int(np.argmax(np.percentile(d.reshape(len(d), -1), 99, axis=1)))
    escala = escala_realce(pix)
    return {"paciente": pid, "corte": nombres[i], "cohorte": ej.loc[pid, "cohorte"],
            **{f: url_imagen(pix[i, k]) for k, f in enumerate(datos.FASES)},
            "REALCE": url_imagen(realce_rgba(pix[i], escala))}


# ---------------------------------------------------------------- recorrido por la red

def _conectividad(bloque) -> torch.Tensor:
    """|pesos| efectivos entrada→salida de un bloque (dos conv con su BatchNorm): [C_sal, C_ent]."""
    capas = bloque.capas
    def efectiva(conv, bn):
        escala = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).abs()
        return conv.weight.abs().sum((2, 3)) * escala[:, None]
    return efectiva(capas[3], capas[4]) @ efectiva(capas[0], capas[1])


def _aristas(pesos: np.ndarray, por_destino: int = 4, signo: bool = False) -> list[list]:
    """pesos [destino, origen] → [[origen, destino, |w| normalizado, signo]] con los más fuertes."""
    maximo = float(np.abs(pesos).max()) or 1.0
    salida = []
    for j, fila in enumerate(pesos):
        for i in np.argsort(-np.abs(fila))[:por_destino]:
            w = abs(float(fila[i])) / maximo
            if w > 0.02:
                salida.append([int(i), j, round(w, 3), int(np.sign(fila[i])) if signo else 0])
    return salida


@torch.no_grad()
def recorrido(modelo: PartyNet, corte: np.ndarray, por_capa: int = 8, n_pool: int = 12,
              n_densa: int = 16) -> dict:
    """Propaga un corte por una red y devuelve mapas reales y pesos para dibujar la red.

    Se muestran los filtros que más contribuyen a la salida: de la capa densa hacia atrás,
    cada capa elige los canales con mayor activación × peso efectivo hacia los ya elegidos.
    """
    modelo.eval()
    x = torch.from_numpy(corte[None].astype(np.float32) / 255)
    vistas = modelo.vistas(x)[0]                                   # [S,3,L,L]
    h = modelo.canales(vistas)                                     # [S,6,L,L]
    mapas = [h]
    for bloque in modelo.backbone:
        h = bloque(h)
        mapas.append(h)
    s, c4 = h.shape[:2]
    z = torch.cat((h.mean((2, 3)), h.amax((2, 3))), 1).reshape(-1)  # mismo orden que representacion()
    lineal = modelo.tronco[1]
    oculta = torch.relu(lineal(z))
    logit = float(modelo._pcr(oculta[None])[0, 0])
    aporte = modelo.cabeza_pcr.weight[0] * oculta                  # [64] aporte de cada neurona al logit
    densa = torch.argsort(aporte.abs(), descending=True)[:n_densa]
    hacia_densa = lineal.weight[densa] * z[None]                   # [n_densa, 960]
    importancia = hacia_densa.abs().sum(0)

    # Bloque 4 y pooling: canales detrás de los rasgos agregados más influyentes.
    canales4, pool = [], []
    for f in torch.argsort(importancia, descending=True).tolist():
        c = f % c4
        if c not in canales4 and len(canales4) < por_capa:
            canales4.append(c)
        if c in canales4:
            pool.append(f)
        if len(pool) == n_pool:
            break
    elegidos = [canales4]
    actividad = [m.abs().mean((0, 2, 3)) for m in mapas]          # media en vistas y espacio
    conect = [_conectividad(b) for b in modelo.backbone]
    pares = []
    for b in range(len(modelo.backbone) - 1, -1, -1):              # bloque b+1 ← entrada del bloque b
        destino = elegidos[0]
        m = conect[b][destino] * actividad[b][None]                # [n_dest, C_ent]
        origen = list(range(m.shape[1])) if b == 0 else torch.argsort(m.sum(0), descending=True)[:por_capa].tolist()
        pares.insert(0, m[:, origen].numpy())
        elegidos.insert(0, origen)

    def imagenes(mapa: torch.Tensor, canal: int, diverge: bool) -> list[str]:
        a = mapa[:, canal].numpy()
        if diverge:
            escala = float(np.percentile(np.abs(a), 99.5)) or 1.0
            return [url_imagen(divergente(v, escala)) for v in a]
        maximo = float(a.max()) or 1.0
        return [url_imagen(magma(v / maximo)) for v in a]

    vistas_np = vistas.numpy()
    nombres_canal = ["PRE", "EARLY", "LATE", "EARLY − PRE", "LATE − PRE", "EARLY − LATE"]
    texto_canal = ["Antes del contraste", "Captación temprana", "Fase tardía",
                   "Realce temprano: dónde entra el contraste", "Realce tardío",
                   "Lavado: positivo si el contraste ya sale"]
    escalas = list(modelo.escalas)
    gris = lambda a: url_imagen((np.clip(a, 0, 1) * 255).astype(np.uint8))
    capas = [
        {"id": "fases", "titulo": "Fases", "forma": "3 × 256²",
         "nodos": [{"img": [url_imagen(corte[k][::2, ::2])], "tit": f, "txt": t}
                   for k, (f, t) in enumerate(zip(datos.FASES, texto_canal))]},
        {"id": "vistas", "titulo": "Vistas", "forma": f"3 × {modelo.lado}²",
         "nodos": [{"img": [gris(vistas_np[v, 1])], "vista": v, "tit": f"Vista central de {e} px",
                    "txt": f"Recorte de {e} × {e} reescalado a {modelo.lado}. Las tres vistas pasan por la misma red."}
                   for v, e in enumerate(escalas)]},
        {"id": "canales", "titulo": "Canales", "forma": f"6 × {modelo.lado}²",
         "nodos": [{"img": imagenes(mapas[0], k, k >= 3) if k >= 3 else [gris(vistas_np[v, k]) for v in range(s)],
                    "tit": nombres_canal[k], "txt": texto_canal[k]} for k in range(6)]},
    ]
    for b, canales in enumerate(elegidos[1:], 1):
        mapa = mapas[b]
        lado = mapa.shape[-1]
        capas.append({"id": f"b{b}", "titulo": f"Bloque {b}", "forma": f"{mapa.shape[1]} × {lado}²",
                      "mostrados": f"{len(canales)} de {mapa.shape[1]}", "pixelado": lado <= 32,
                      "nodos": [{"img": imagenes(mapa, c, False), "tit": f"Bloque {b} · filtro {c + 1} de {mapa.shape[1]}",
                                 "txt": f"Mapa de {lado} × {lado} tras 2 conv 3×3 + ReLU y MaxPool. "
                                        f"Activación media {float(actividad[b][c]):.2f}".replace(".", ",")}
                                for c in canales]})
    zp = z.numpy()
    imp_pool = importancia[pool].numpy()
    capas.append({"id": "pool", "titulo": "Pooling", "forma": f"{z.numel()} rasgos", "mostrados": f"{len(pool)} de {z.numel()}",
                  "nodos": [{"v": round(float(imp_pool[k] / imp_pool.max()), 3),
                             "tit": f"Vista {escalas[f // (2 * c4)]} px · {'media' if (f % (2 * c4)) < c4 else 'máximo'} · filtro {f % c4 + 1}",
                             "txt": f"Valor {zp[f]:.2f}: resume todo el mapa del filtro en un número".replace(".", ",")}
                            for k, f in enumerate(pool)]})
    ap = aporte[densa].numpy()
    capas.append({"id": "densa", "titulo": "Densa", "forma": f"{oculta.numel()} neuronas", "mostrados": f"{len(densa)} de {oculta.numel()}",
                  "nodos": [{"v": round(float(abs(a) / (np.abs(ap).max() or 1)), 3), "signo": int(np.sign(a)),
                             "tit": f"Neurona {int(n) + 1} de {oculta.numel()}",
                             "txt": f"Activación {float(oculta[n]):.2f} · aporta {a:+.2f} al logit".replace(".", ",")}
                            for n, a in zip(densa.tolist(), ap)]})
    p = 1 / (1 + np.exp(-logit))
    capas.append({"id": "salida", "titulo": "Salida", "forma": "1 logit",
                  "nodos": [{"p": round(float(p), 4), "tit": "Probabilidad de pCR de este corte",
                             "txt": f"logit {logit:+.2f} → σ = {p:.3f}. Red 1 de 5, sin reflexiones ni calibrar; "
                                    "la predicción final promedia cortes y redes.".replace(".", ",", 2)}]})

    canal_de = {c: k for k, c in enumerate(canales4)}
    aristas = [
        [[i, j, 0.5, 0, j] for i in range(3) for j in range(s)],
        [[v, k, 0.5, 0, v] for v in range(s) for k in range(6)],
        *[_aristas(par) for par in pares],
        [[canal_de[f % c4], k, round(float(imp_pool[k] / imp_pool.max()), 3), 0] for k, f in enumerate(pool)],
        _aristas(hacia_densa[:, pool].numpy(), por_destino=4, signo=True),
        [[k, 0, round(float(abs(a) / (np.abs(ap).max() or 1)), 3), int(np.sign(a))] for k, a in enumerate(ap)],
    ]
    return {"capas": capas, "aristas": aristas, "p": float(p), "logit": logit}


# ---------------------------------------------------------------- arquitectura

def capas_red() -> list[dict]:
    """Capas de PartyNet para la tabla, con parámetros contados del propio modelo."""
    cfg = json.loads((PAQUETE_IMAGEN / "paquete.json").read_text())["config_modelo"]
    m = PartyNet(**cfg)
    contar = lambda mod: sum(p.numel() for p in mod.parameters())
    anchos, lado, escalas = cfg["anchos"], cfg["lado"], cfg["escalas"]
    capas = [
        {"nombre": "Entrada", "forma": "3 fases × 256 × 256",
         "detalle": "PRE, EARLY y LATE del mismo corte, en gris y divididas entre 255", "params": 0},
        {"nombre": "Canales cinéticos", "forma": f"6 canales × {len(escalas)} vistas × {lado} × {lado}",
         "detalle": "Se añaden EARLY−PRE, LATE−PRE y EARLY−LATE; recortes centrales de 256, 128 y 64 px reescalados a 128",
         "params": 0},
    ]
    entrada, s = 6, lado
    for i, (c, bloque) in enumerate(zip(anchos, m.backbone), 1):
        s //= 2
        capas.append({"nombre": f"Bloque conv {i}", "forma": f"{c} × {s} × {s}",
                      "detalle": f"2 convoluciones 3×3 ({entrada}→{c}) con BatchNorm y ReLU, y MaxPool 2×2",
                      "params": f"{contar(bloque):,}".replace(",", ".")})
        entrada = c
    capas += [
        {"nombre": "Pooling global", "forma": f"{anchos[-1]}×2×{len(escalas)} = {anchos[-1] * 2 * len(escalas)}",
         "detalle": "Media y máximo de cada mapa en las tres vistas (pesos compartidos); Dropout2d 0,10", "params": 0},
        {"nombre": "Capa densa", "forma": f"{cfg['oculto']} neuronas",
         "detalle": f"Dropout {cfg['dropout']}, lineal {anchos[-1] * 2 * len(escalas)}→{cfg['oculto']} y ReLU",
         "params": f"{contar(m.tronco):,}".replace(",", ".")},
        {"nombre": "Salida", "forma": "1 logit de pCR",
         "detalle": "Sigmoide → probabilidad del corte; la paciente es la media de sus cortes y de 5 redes",
         "params": f"{contar(m.cabeza_pcr):,}".replace(",", ".")},
    ]
    return capas
