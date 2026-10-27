"""Piezas visuales compartidas: estilo «sala de lectura», cabeceras, lecturas, red, visor y gráficas."""
from __future__ import annotations

import html
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

RECURSOS = Path(__file__).resolve().parent / "recursos"

# Paleta de gráficas validada (CVD y contraste) sobre el panel oscuro #0e1217.
COBRE, ACERO, OCRE, GRIS = "#c46a46", "#3790b8", "#a8843a", "#8d949d"
TINTA, TINTA2, TENUE, LINEA, PANEL = "#ece8df", "#a9afb7", "#6b7480", "#1d252f", "#0d1116"
BANDAS = {"baja": "#4a3027", "intermedia": "#8a4e33", "alta": "#c97753"}
NOMBRE_BANDA = {"baja": "Probabilidad baja", "intermedia": "Zona intermedia", "alta": "Probabilidad alta"}
COHORTES = {"spy2": "I-SPY2", "duke": "Duke", "spy1": "I-SPY1"}
FUENTES = ("https://fonts.googleapis.com/css2?family=Geist:wght@300..700&family=Geist+Mono:wght@400..600"
           "&family=Instrument+Serif:ital@0;1&display=swap")


def recurso(nombre: str) -> dict | None:
    ruta = RECURSOS / f"{nombre}.json"
    return json.loads(ruta.read_text()) if ruta.exists() else None


def estilo() -> None:
    st.markdown(f"""
<style>
@import url('{FUENTES}');
:root{{--fondo:#0b0e12;--panel:#0d1116;--panel2:#121821;--linea:#1d252f;--linea2:#2a3440;
--tinta:#ece8df;--tinta2:#a9afb7;--tenue:#6b7480;--realce:#c97753;--realce-suave:rgba(201,119,83,.10);
--frio:#5a9ec4;--ambar:#b8955a;--serif:'Instrument Serif',Georgia,serif;--mono:'Geist Mono',ui-monospace,monospace;
--sans:'Geist',system-ui,sans-serif}}
.stApp{{isolation:isolate;background:
var(--fondo)}}
.stApp::before{{content:"";position:fixed;inset:0;z-index:-1;pointer-events:none;
  background-image:linear-gradient(rgba(236,232,223,.018) 1px,transparent 1px),linear-gradient(90deg,rgba(236,232,223,.018) 1px,transparent 1px);
  background-size:56px 56px;-webkit-mask-image:radial-gradient(ellipse 80% 70% at 50% 20%,#000 20%,transparent 85%);
  mask-image:radial-gradient(ellipse 80% 70% at 50% 20%,#000 20%,transparent 85%)}}
header[data-testid="stHeader"]{{background:rgba(11,14,18,.78);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
  border-bottom:1px solid var(--linea)}}
[data-testid="stAppDeployButton"],[data-testid="stDecoration"]{{display:none}}
.block-container{{max-width:1240px;padding-top:5.4rem;padding-bottom:3rem}}
::selection{{background:var(--realce);color:#0b0e12}}
code{{font-family:var(--mono);background:var(--panel2);color:var(--realce);border-radius:2px;padding:.05rem .35rem}}
#vg-tooltip-element{{background:#0d1116!important;color:var(--tinta)!important;border:1px solid var(--linea2)!important;
  border-radius:3px!important;font-family:var(--sans)!important;box-shadow:0 12px 30px rgba(0,0,0,.5)!important}}
#vg-tooltip-element td.key{{color:var(--tenue)!important}}
[data-testid="stExpander"] details{{background:var(--panel);border:1px solid var(--linea);border-radius:3px}}
[data-testid="stExpander"] summary:hover{{color:var(--realce)}}
[data-testid="stFileUploaderDropzone"]{{background:repeating-linear-gradient(135deg,rgba(236,232,223,.015) 0 10px,transparent 10px 20px),var(--panel);
  border:1px dashed var(--linea2)}}
[data-testid="stWidgetLabel"] p{{font-family:var(--mono);font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:var(--tenue)}}
[data-testid="stCaptionContainer"]{{color:var(--tenue)}}
/* tipografía propia */
.ojo{{display:flex;align-items:center;gap:.65rem;font:500 .7rem/1 var(--mono);letter-spacing:.2em;text-transform:uppercase;color:var(--tinta2);margin-bottom:1.1rem}}
.ojo::before{{content:"";width:18px;height:1px;background:var(--realce)}}
.titular{{font-family:var(--serif);font-weight:400;font-size:clamp(2.7rem,5.2vw,4.5rem);line-height:.98;letter-spacing:-.018em;color:var(--tinta);margin:0 0 1.1rem}}
.titular em,.h2 em{{color:var(--tinta2);font-style:italic}}
.entradilla{{font-size:1.08rem;line-height:1.65;color:var(--tinta2);max-width:720px;margin:0}}
.cabecera{{padding:1.2rem 0 2rem;border-bottom:1px solid var(--linea);margin-bottom:2.2rem;position:relative}}
.meta{{display:flex;flex-wrap:wrap;gap:0;margin-top:1.6rem;border:1px solid var(--linea);width:fit-content;max-width:100%}}
.meta div{{padding:.55rem .9rem;border-right:1px solid var(--linea);font:400 .72rem/1.35 var(--mono);color:var(--tenue);letter-spacing:.06em}}
.meta div:last-child{{border-right:0}}
.meta b{{display:block;color:var(--tinta);font-weight:500;font-size:.86rem;letter-spacing:0}}
.seccion{{display:flex;align-items:baseline;gap:1rem;margin:4.2rem 0 .5rem}}
.seccion .num{{font:500 .78rem var(--mono);color:var(--realce);letter-spacing:.08em}}
.seccion .h2{{font-family:var(--serif);font-size:2.15rem;line-height:1.05;color:var(--tinta);white-space:nowrap}}
.seccion .regla{{flex:1;height:1px;background:linear-gradient(90deg,var(--linea2),transparent);transform:translateY(-.4rem)}}
.seccion-texto{{color:var(--tinta2);max-width:820px;line-height:1.65;margin:.3rem 0 1.3rem}}
/* tarjetas con esquinas de visor */
.rejilla{{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:12px;margin:.4rem 0 1rem}}
.kpi{{--cc:var(--linea2);position:relative;padding:18px 18px 16px;background:linear-gradient(180deg,var(--panel2),var(--panel));
  border:1px solid var(--linea);border-radius:2px;transition:transform .2s ease,border-color .2s}}
.kpi::after{{content:"";position:absolute;inset:-1px;pointer-events:none;background:
  linear-gradient(var(--cc),var(--cc)) 0 0/14px 1px,linear-gradient(var(--cc),var(--cc)) 0 0/1px 14px,
  linear-gradient(var(--cc),var(--cc)) 100% 0/14px 1px,linear-gradient(var(--cc),var(--cc)) 100% 0/1px 14px,
  linear-gradient(var(--cc),var(--cc)) 0 100%/14px 1px,linear-gradient(var(--cc),var(--cc)) 0 100%/1px 14px,
  linear-gradient(var(--cc),var(--cc)) 100% 100%/14px 1px,linear-gradient(var(--cc),var(--cc)) 100% 100%/1px 14px;background-repeat:no-repeat}}
.kpi:hover{{--cc:var(--tinta2)}}
.kpi.acento{{border-top:1px solid var(--realce)}}
.kpi .t{{font:500 .66rem/1.3 var(--mono);letter-spacing:.16em;text-transform:uppercase;color:var(--tenue)}}
.kpi .v{{font-family:var(--serif);font-size:3.05rem;line-height:1;margin:.55rem 0 .45rem;color:var(--tinta);font-variant-numeric:tabular-nums}}
.kpi.acento .v{{color:var(--tinta)}}
.kpi .d{{font-size:.84rem;color:var(--tinta2);line-height:1.45}}
.kpi .barra{{height:2px;background:var(--linea);margin-top:14px;position:relative}}
.kpi .barra i{{position:absolute;left:0;top:0;bottom:0;background:var(--realce);animation:crece 1.2s cubic-bezier(.2,.8,.2,1) both}}
.kpi .barra span{{position:absolute;right:0;top:6px;font:400 .62rem var(--mono);color:var(--tenue)}}
@keyframes crece{{from{{width:0}}}}
.kpi .ayuda{{visibility:hidden;opacity:0;position:absolute;left:10px;right:10px;top:calc(100% + 8px);z-index:20;background:#0a0d11;
  border:1px solid var(--linea2);color:var(--tinta2);font-size:.82rem;line-height:1.5;padding:11px 13px;border-radius:2px;
  box-shadow:0 18px 40px rgba(0,0,0,.55);transition:opacity .15s}}
.kpi:hover .ayuda{{visibility:visible;opacity:1}}
/* notas */
.nota{{padding:15px 18px;border:1px solid var(--linea);background:var(--panel);font-size:.94rem;line-height:1.65;color:var(--tinta2);
  border-radius:2px;margin:.7rem 0 1rem}}
.nota b{{color:var(--tinta);font-weight:600}}
.nota.clave{{border-left:2px solid var(--realce)}}
.nota.info{{border-left:2px solid var(--frio)}}
.nota.atencion{{border-color:rgba(184,149,90,.3);background:linear-gradient(90deg,rgba(184,149,90,.08),transparent 60%),var(--panel);
  border-left:2px solid var(--ambar)}}
/* línea de pasos */
.hitos{{display:grid;grid-template-columns:repeat(var(--n),minmax(0,1fr));position:relative;margin:1.6rem 0 1.4rem;gap:0}}
.hitos::before{{content:"";position:absolute;left:15px;right:0;top:15px;height:1px;background:linear-gradient(90deg,var(--realce),var(--linea2) 45%,var(--linea))}}
.hito{{position:relative;padding-right:22px}}
.hito .n{{width:31px;height:31px;border-radius:50%;border:1px solid var(--linea2);background:var(--fondo);display:grid;place-items:center;
  font:500 .7rem var(--mono);color:var(--tinta2);position:relative;transition:all .2s}}
.hito:first-child .n{{border-color:var(--realce);color:var(--realce)}}
.hito:hover .n{{border-color:var(--realce);color:var(--realce)}}
.hito b{{display:block;margin-top:1rem;font-weight:500;color:var(--tinta);font-size:.98rem}}
.hito span{{display:block;margin-top:.3rem;font-size:.85rem;line-height:1.5;color:var(--tinta2)}}
@media (max-width:820px){{.hitos{{grid-template-columns:1fr 1fr;row-gap:1.4rem}}.hitos::before{{display:none}}}}
/* tablas */
table.limpia{{width:100%;border-collapse:collapse;font-size:.9rem;background:var(--panel);border:1px solid var(--linea)}}
table.limpia th{{text-align:left;font:500 .66rem var(--mono);letter-spacing:.14em;text-transform:uppercase;color:var(--tenue);
  background:var(--panel2);padding:11px 14px;border-bottom:1px solid var(--linea2)}}
table.limpia td{{padding:10px 14px;border-bottom:1px solid var(--linea);color:var(--tinta2);font-variant-numeric:tabular-nums}}
table.limpia td:first-child{{color:var(--tinta)}}
table.limpia tr:hover td{{background:rgba(201,119,83,.04)}}
/* lectura de probabilidad */
.lectura{{position:relative;background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--linea);padding:22px 24px 18px;border-radius:2px;height:100%}}
.lectura.vacia{{display:flex;flex-direction:column;justify-content:center;min-height:330px}}
.lectura .cab{{display:flex;justify-content:space-between;gap:1rem;align-items:center;font:500 .68rem var(--mono);letter-spacing:.16em;text-transform:uppercase;color:var(--tenue)}}
.lectura .cab .t{{color:var(--realce)}}
.lectura .cifra{{font-family:var(--serif);font-size:5.4rem;line-height:.95;margin:.9rem 0 .2rem;color:var(--tinta);font-variant-numeric:tabular-nums}}
.lectura .cifra small{{font-size:2rem;color:var(--tenue);margin-left:.2rem}}
.lectura .cifra.gris{{color:#3a4450}}
.lectura .que{{color:var(--tinta2);font-size:.92rem;line-height:1.55;min-height:2.9em}}
.lectura .que b{{color:var(--tinta)}}
.lectura .pie-l{{font:400 .72rem/1.5 var(--mono);color:var(--tenue);margin-top:1rem;border-top:1px solid var(--linea);padding-top:.8rem}}
.chip{{display:inline-block;font:500 .66rem var(--mono);letter-spacing:.12em;text-transform:uppercase;padding:.3rem .55rem;border-radius:2px;margin-right:.5rem;vertical-align:1px}}
.regla-p{{position:relative;height:70px;margin:1.6rem 0 .3rem}}
.regla-p .seg{{position:absolute;top:22px;height:12px;cursor:help;transition:filter .15s}}
.regla-p .seg:hover{{filter:brightness(1.3)}}
.regla-p .seg+.seg{{box-shadow:inset 2px 0 0 var(--panel)}}
.regla-p .seg .tip{{visibility:hidden;opacity:0;position:absolute;bottom:22px;left:50%;transform:translateX(-50%);width:240px;background:#0a0d11;
  border:1px solid var(--linea2);color:var(--tinta2);font-size:.8rem;line-height:1.5;padding:10px 12px;z-index:30;transition:opacity .15s;
  box-shadow:0 18px 40px rgba(0,0,0,.55)}}
.regla-p .seg .tip b{{color:var(--tinta)}}
.regla-p .seg:hover .tip{{visibility:visible;opacity:1}}
.regla-p .tick{{position:absolute;top:38px;width:1px;height:5px;background:var(--linea2)}}
.regla-p .tick.m{{height:9px;background:var(--tenue)}}
.regla-p .tick span{{position:absolute;top:11px;left:50%;transform:translateX(-50%);font:400 .62rem var(--mono);color:var(--tenue)}}
.regla-p .umbral{{position:absolute;top:14px;height:28px;border-left:1px dashed var(--tinta2);opacity:.7}}
.regla-p .umbral.izq span{{right:6px}}.regla-p .umbral.der span{{left:6px}}
.regla-p .umbral span{{position:absolute;top:-2px;font:400 .58rem var(--mono);letter-spacing:.1em;color:var(--tinta2);white-space:nowrap}}
.regla-p .aguja{{position:absolute;top:6px;width:2px;height:42px;margin-left:-1px;background:var(--tinta);box-shadow:0 0 0 3px rgba(11,14,18,.9);
  animation:aguja 1.3s cubic-bezier(.2,.9,.2,1) both}}
.regla-p .aguja::before{{content:"";position:absolute;top:-6px;left:-5px;border:6px solid transparent;border-top-color:var(--tinta)}}
@keyframes aguja{{from{{left:0}}}}
/* portada */
.portada{{display:grid;grid-template-columns:1.12fr .88fr;gap:3.2rem;align-items:center;padding:.6rem 0 2.6rem;border-bottom:1px solid var(--linea);margin-bottom:1rem}}
@media (max-width:900px){{.portada{{grid-template-columns:1fr}}}}
.cta{{display:flex;gap:.7rem;flex-wrap:wrap;margin-top:2rem}}
.cta a{{display:inline-flex;align-items:center;gap:.6rem;padding:.8rem 1.15rem;font:500 .74rem var(--mono);letter-spacing:.14em;text-transform:uppercase;
  text-decoration:none!important;border:1px solid var(--linea2);color:var(--tinta)!important;border-radius:2px;transition:all .2s}}
.cta a.prim{{background:var(--realce);border-color:var(--realce);color:#0b0e12!important}}
.cta a:hover{{transform:translateY(-2px);border-color:var(--realce)}}
.cta a .fl{{transition:transform .2s}}.cta a:hover .fl{{transform:translateX(4px)}}
.visor-p{{position:relative;aspect-ratio:1;background:#000;border:1px solid var(--linea2);overflow:hidden;}}
.visor-p img{{position:absolute;inset:0;width:100%;height:100%;object-fit:cover}}
.visor-p img:first-child{{filter:brightness(.72) contrast(1.05)}}
.visor-p .r{{opacity:.85;animation:barrido 9s cubic-bezier(.65,0,.35,1) infinite alternate}}
.visor-p .scan{{position:absolute;top:0;bottom:0;width:1px;background:rgba(236,232,223,.55);animation:scan 9s cubic-bezier(.65,0,.35,1) infinite alternate}}
@keyframes barrido{{from{{clip-path:inset(0 100% 0 0)}}to{{clip-path:inset(0 0 0 0)}}}}
@keyframes scan{{from{{left:0}}to{{left:calc(100% - 2px)}}}}
.visor-p .mira{{position:absolute;inset:0;pointer-events:none;background:
  linear-gradient(rgba(236,232,223,.18),rgba(236,232,223,.18)) 50% calc(50% - 22px)/1px 14px no-repeat,
  linear-gradient(rgba(236,232,223,.18),rgba(236,232,223,.18)) 50% calc(50% + 22px)/1px 14px no-repeat,
  linear-gradient(rgba(236,232,223,.18),rgba(236,232,223,.18)) calc(50% - 22px) 50%/14px 1px no-repeat,
  linear-gradient(rgba(236,232,223,.18),rgba(236,232,223,.18)) calc(50% + 22px) 50%/14px 1px no-repeat}}
.esq{{position:absolute;font:400 .64rem/1.5 var(--mono);letter-spacing:.08em;color:rgba(236,232,223,.72);text-shadow:0 1px 2px #000;text-transform:uppercase}}
.esq.tl{{top:10px;left:12px}}.esq.tr{{top:10px;right:12px;text-align:right}}.esq.bl{{bottom:10px;left:12px}}.esq.br{{bottom:10px;right:12px;text-align:right}}
.esq .o{{color:var(--realce)}}
.fases-p{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:8px}}
.fases-p div{{position:relative;aspect-ratio:1;border:1px solid var(--linea);overflow:hidden;background:#000}}
.fases-p img{{width:100%;height:100%;object-fit:cover;display:block;opacity:.85;transition:opacity .2s}}
.fases-p div:hover img{{opacity:1}}
.fases-p span{{position:absolute;left:7px;bottom:5px;font:500 .6rem var(--mono);letter-spacing:.14em;color:var(--tinta);text-shadow:0 1px 2px #000}}
/* destinos */
.destinos{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px}}
.destino{{position:relative;display:block;padding:20px 20px 22px;border:1px solid var(--linea);background:var(--panel);text-decoration:none!important;
  color:var(--tinta)!important;border-radius:2px;overflow:hidden;transition:all .25s}}
.destino::before{{content:none}}
.destino:hover{{border-color:var(--linea2);background:var(--panel2)}}
.destino:hover::before{{opacity:1}}
.destino .num{{font:500 .68rem var(--mono);color:var(--realce);letter-spacing:.14em}}
.destino .h{{display:block;font-family:var(--serif);font-size:1.75rem;line-height:1.1;margin:.8rem 0 .5rem}}
.destino .d{{display:block;color:var(--tinta2);font-size:.88rem;line-height:1.5;max-width:92%}}
.destino .fl{{position:absolute;right:18px;top:16px;font-family:var(--mono);color:var(--tenue);transition:all .25s}}
.destino:hover .fl{{color:var(--realce);transform:translate(3px,-3px)}}
/* matriz de error explicada */
.error{{display:grid;grid-template-columns:62px 1fr;gap:16px;align-items:start}}
.mini{{display:grid;grid-template-columns:1fr 1fr;gap:3px;width:62px}}
.mini i{{aspect-ratio:1;background:var(--panel2);border:1px solid var(--linea)}}
.mini i.on{{background:var(--realce);border-color:var(--realce)}}
.pie{{color:var(--tenue);font:400 .7rem/1.6 var(--mono);letter-spacing:.06em;margin-top:4.5rem;border-top:1px solid var(--linea);padding-top:1.1rem;
  display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap}}
</style>""", unsafe_allow_html=True)


def cabecera(ojo: str, titulo: str, texto: str, meta: list[tuple[str, str]] | None = None) -> None:
    """titulo admite <em> para la palabra destacada; meta son pares (etiqueta, valor)."""
    bloque = ""
    if meta:
        bloque = '<div class="meta">' + "".join(f"<div>{k}<b>{v}</b></div>" for k, v in meta) + "</div>"
    st.markdown(f'<div class="cabecera"><div class="ojo">{ojo}</div><div class="titular">{titulo}</div>'
                f'<p class="entradilla">{texto}</p>{bloque}</div>', unsafe_allow_html=True)


def seccion(numero: str, titulo: str, texto: str | None = None) -> None:
    st.markdown(f'<div class="seccion"><span class="num">{int(numero):02d} /</span><span class="h2">{titulo}</span>'
                f'<span class="regla"></span></div>', unsafe_allow_html=True)
    if texto:
        st.markdown(f'<div class="seccion-texto">{texto}</div>', unsafe_allow_html=True)


def tarjetas(items: list[dict]) -> None:
    """items: t (título), v (valor), d (detalle), ayuda (hover), barra (0–1), barra_txt, acento."""
    partes = []
    for it in items:
        ayuda = f'<div class="ayuda">{it["ayuda"]}</div>' if it.get("ayuda") else ""
        barra = ""
        if it.get("barra") is not None:
            barra = (f'<div class="barra"><i style="width:{100 * it["barra"]:.1f}%"></i>'
                     f'<span>{it.get("barra_txt", "")}</span></div>')
        clase = "kpi acento" if it.get("acento") else "kpi"
        partes.append(f'<div class="{clase}"><div class="t">{it["t"]}</div><div class="v">{it["v"]}</div>'
                      f'<div class="d">{it.get("d", "")}</div>{barra}{ayuda}</div>')
    st.markdown(f'<div class="rejilla">{"".join(partes)}</div>', unsafe_allow_html=True)


def aviso(texto: str, tipo: str = "info") -> None:
    st.markdown(f'<div class="nota {tipo}">{texto}</div>', unsafe_allow_html=True)


def pasos(items: list[tuple[str, str]]) -> None:
    st.markdown(f'<div class="hitos" style="--n:{len(items)}">' + "".join(
        f'<div class="hito"><div class="n">{i:02d}</div><b>{t}</b><span>{d}</span></div>'
        for i, (t, d) in enumerate(items, 1)) + "</div>", unsafe_allow_html=True)


def tabla(df: pd.DataFrame) -> None:
    st.markdown(df.to_html(index=False, classes="limpia", border=0, escape=False), unsafe_allow_html=True)


def destinos(items: list[tuple[str, str, str]]) -> None:
    """items: (ruta de la página, título, descripción)."""
    st.markdown('<div class="destinos">' + "".join(
        f'<a class="destino" href="{ruta}" target="_self"><span class="num">{i:02d}</span><span class="fl">↗</span>'
        f'<span class="h">{t}</span><span class="d">{d}</span></a>'
        for i, (ruta, t, d) in enumerate(items, 1)) + "</div>", unsafe_allow_html=True)


def pct(x: float, dec: int = 0) -> str:
    return f"{100 * x:.{dec}f} %".replace(".", ",")


def num(x: float, dec: int = 3) -> str:
    return f"{x:.{dec}f}".replace(".", ",")


def banda_de(p: float, info: dict) -> str:
    c1, c2 = info["cortes"]
    return "baja" if p < c1 else ("intermedia" if p < c2 else "alta")


def chip_banda(banda: str) -> str:
    color = "#0b0e12" if banda == "alta" else TINTA
    return f'<span class="chip" style="background:{BANDAS[banda]};color:{color}">{NOMBRE_BANDA[banda]}</span>'


# ---------------------------------------------------------------- lectura de una probabilidad

def lectura(titulo: str, p: float, info: dict, umbral: float, pie: str, extra: str = "") -> None:
    """Cifra grande, banda observada en la CV y regla con las tres bandas, el umbral y la aguja."""
    filas = info["bandas"]
    hi = max(0.6, min(1.0, filas[-1]["p_max"] + 0.05))
    x = lambda v: 100 * min(max(v, 0.0), hi) / hi
    limites = [0.0, *info["cortes"], hi]
    banda = banda_de(p, info)
    fila = next(f for f in filas if f["banda"] == banda)
    segmentos = []
    for f, a, b in zip(filas, limites[:-1], limites[1:]):
        ic = f["ic95"]
        tip = (f"<b>{NOMBRE_BANDA[f['banda']]}</b> · {num(a, 2)}–{num(b, 2)}<br>En la validación cruzada: "
               f"{f['pcr']} pCR de {f['n']} pacientes ({pct(f['tasa_pcr'])}; IC95 % {pct(ic[0])}–{pct(ic[1])})")
        segmentos.append(f'<div class="seg" style="left:{x(a):.2f}%;width:{x(b) - x(a):.2f}%;background:{BANDAS[f["banda"]]}">'
                         f'<div class="tip">{tip}</div></div>')
    marcas = "".join(
        f'<div class="tick{" m" if round(v * 100) % 10 == 0 else ""}" style="left:{x(v):.2f}%">'
        + (f"<span>{num(v, 1)}</span>" if round(v * 100) % 10 == 0 else "") + "</div>"
        for v in np.arange(0, hi + 1e-9, 0.05))
    clase = "pCR" if p >= umbral else "no pCR"
    st.markdown(
        f'<div class="lectura"><div class="cab"><span class="t">{titulo}</span><span>umbral {num(umbral, 3)} → {clase}</span></div>'
        f'<div class="cifra">{num(100 * p, 1)}<small>%</small></div>'
        f'<div class="que">{chip_banda(banda)} En la validación cruzada, <b>{pct(fila["tasa_pcr"])}</b> de las pacientes '
        f'de esta banda alcanzó pCR (IC95 % {pct(fila["ic95"][0])}–{pct(fila["ic95"][1])}).{extra}</div>'
        f'<div class="regla-p">{"".join(segmentos)}{marcas}'
        f'<div class="umbral {"izq" if p >= umbral else "der"}" style="left:{x(umbral):.2f}%"><span>UMBRAL</span></div>'
        f'<div class="aguja" style="left:{x(p):.2f}%"></div></div>'
        f'<div class="pie-l">{pie}</div></div>', unsafe_allow_html=True)


def lectura_vacia(titulo: str, texto: str) -> None:
    st.markdown(f'<div class="lectura vacia"><div class="cab"><span class="t">{titulo}</span></div>'
                f'<div class="cifra gris">—</div><div class="que">{texto}</div></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- portada

def portada(img: dict, titulo: str, texto: str, enlaces: list[tuple[str, str, bool]]) -> None:
    botones = "".join(f'<a class="{"prim" if prim else ""}" href="{ruta}" target="_self">{t} <span class="fl">→</span></a>'
                      for ruta, t, prim in enlaces)
    cohorte = COHORTES.get(img["cohorte"], img["cohorte"])
    fases = "".join(f'<div><img src="{img[f]}" alt="{f}"><span>{f}</span></div>' for f in ("PRE", "EARLY", "LATE"))
    st.markdown(f"""
<div class="portada">
  <div>
    <div class="ojo">Trabajo académico · Cáncer de mama · DCE-MRI</div>
    <div class="titular" style="font-size:clamp(2.4rem,4.2vw,3.7rem);line-height:1.04">{titulo}</div>
    <p class="entradilla">{texto}</p>
    <div class="cta">{botones}</div>
  </div>
  <div>
    <div class="visor-p">
      <img src="{img['EARLY']}" alt="Fase EARLY"><img class="r" src="{img['REALCE']}" alt="Mapa de realce">
      <div class="scan"></div><div class="mira"></div>
      <div class="esq tl">{cohorte} · {html.escape(img['paciente'])}<br>corte {img['corte']}</div>
      <div class="esq tr">DCE-MRI · T1<br><span class="o">EARLY + realce</span></div>
      <div class="esq bl">256 × 256 · 8 bit</div>
      <div class="esq br">paciente de train</div>
    </div>
    <div class="fases-p">{fases}</div>
  </div>
</div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- red neuronal animada

_RED = r"""
<link rel="stylesheet" href="__FUENTES__">
<div id="red">
  <canvas id="base"></canvas><canvas id="fx"></canvas>
  <div id="nodos"></div>
  <div id="tip"></div>
  <div id="barra">
    <div class="ley"><i class="ln n"></i>peso efectivo (grosor = magnitud) <i class="ln r"></i>empuja hacia pCR <i class="ln c"></i>en contra</div>
    <div class="ctl">
      <span>vista</span><button data-v="0" class="on">256 px</button><button data-v="1">128 px</button><button data-v="2">64 px</button>
      <button id="play">▶ Propagar</button>
    </div>
  </div>
</div>
<style>
:root{--l:#1d252f;--l2:#2a3440;--t:#ece8df;--t2:#a9afb7;--te:#6b7480;--r:#c97753;--c:#5a9ec4}
html,body{margin:0;background:transparent;color:var(--t);font-family:Geist,system-ui,sans-serif;overflow:hidden}
#red{position:relative;height:__ALTO__px;border:1px solid var(--l);border-radius:2px;overflow:hidden;
  background:radial-gradient(70% 80% at 55% 45%,#10161d 0%,#0b0e12 75%)}
#red::before{content:"";position:absolute;inset:0;pointer-events:none;background-image:linear-gradient(rgba(236,232,223,.03) 1px,transparent 1px),
  linear-gradient(90deg,rgba(236,232,223,.03) 1px,transparent 1px);background-size:28px 28px}
canvas{position:absolute;inset:0}
.col-tit{position:absolute;top:16px;transform:translateX(-50%);text-align:center;font:500 10px/1.4 'Geist Mono',monospace;
  letter-spacing:.16em;text-transform:uppercase;color:var(--te);white-space:nowrap;transition:color .4s}
.col-tit.on{color:var(--t2)}
.col-tit small{display:block;letter-spacing:.02em;text-transform:none;font-size:10px;color:var(--te)}
.col-tit small.m{color:#4d5763;font-size:9.5px}
.nodo{position:absolute;transform:translate(-50%,-50%) scale(.55);opacity:.08;transition:opacity .45s,transform .45s cubic-bezier(.2,.9,.3,1.3),filter .2s;cursor:crosshair;z-index:2}
.nodo.on{opacity:1;transform:translate(-50%,-50%) scale(1)}
.nodo.img img{display:block;width:100%;height:100%;border:1px solid var(--l2);border-radius:2px;box-sizing:border-box;background:#000}
.nodo.px img{image-rendering:pixelated}
.nodo.punto{border-radius:50%;border:1px solid;box-sizing:border-box}
.nodo.sel img{border-color:var(--t2);box-shadow:0 0 0 1px var(--t2)}
.nodo:hover{z-index:5}
.nodo.foco img{border-color:var(--r);box-shadow:0 0 0 1px var(--r)}
.nodo.punto.foco{box-shadow:0 0 0 3px rgba(201,119,83,.35)}
.nodo.salida{display:grid;place-items:center}
.nodo.salida svg{position:absolute;inset:0;width:100%;height:100%;overflow:visible}
.nodo.salida .pista{fill:#0b0f13;stroke:#2a3440;stroke-width:3}
.nodo.salida .valor{fill:none;stroke:var(--r);stroke-width:3.5;stroke-linecap:round}
.nodo.salida.on .valor{animation:llena 1.4s cubic-bezier(.2,.8,.2,1) both}
@keyframes llena{from{stroke-dasharray:0 400}}
.nodo.salida span{position:relative;font-family:'Instrument Serif',Georgia,serif;font-size:25px;line-height:1}
.nodo.salida span i{font-style:normal;font-size:13px;color:var(--t2)}
#tip{position:absolute;z-index:10;pointer-events:none;display:flex;gap:12px;align-items:flex-start;max-width:330px;padding:11px;
  background:rgba(10,13,17,.96);border:1px solid var(--l2);box-shadow:0 20px 50px rgba(0,0,0,.6);opacity:0;transition:opacity .12s;border-radius:2px}
#tip img{width:112px;height:112px;flex:none;border:1px solid var(--l2)}
#tip img.px{image-rendering:pixelated}
#tip b{display:block;font-weight:500;font-size:13px;margin-bottom:4px}
#tip span{display:block;font-size:12px;line-height:1.5;color:var(--t2)}
#barra{position:absolute;left:0;right:0;bottom:0;height:40px;display:flex;justify-content:space-between;align-items:center;padding:0 14px;
  border-top:1px solid var(--l);background:rgba(11,14,18,.7);font:400 10.5px 'Geist Mono',monospace;color:var(--te);letter-spacing:.04em}
.ley{display:flex;align-items:center;gap:7px;white-space:nowrap;overflow:hidden}
.ln{display:inline-block;width:18px;height:2px;margin-left:8px}
.ln.n{background:rgba(236,232,223,.6);margin-left:0}.ln.r{background:var(--r)}.ln.c{background:var(--c)}
.ctl{display:flex;align-items:center;gap:4px}
.ctl span{margin-right:4px;text-transform:uppercase;letter-spacing:.14em}
button{font:500 10.5px 'Geist Mono',monospace;color:var(--t2);background:transparent;border:1px solid var(--l2);padding:5px 9px;cursor:pointer;border-radius:2px;letter-spacing:.04em}
button:hover{border-color:var(--r);color:var(--t)}
button.on{background:var(--t);color:#0b0e12;border-color:var(--t)}
#play{margin-left:10px;border-color:var(--r);color:var(--r)}
#play:hover{background:var(--r);color:#0b0e12}
</style>
<script>
const D=__DATOS__;
const caps=D.capas,N=caps.length,AR=D.aristas,DUR=2800;
const root=document.getElementById('red'),cB=document.getElementById('base'),cF=document.getElementById('fx');
const gB=cB.getContext('2d'),gF=cF.getContext('2d'),cont=document.getElementById('nodos'),tip=document.getElementById('tip');
const reducido=matchMedia('(prefers-reduced-motion: reduce)').matches;
const COL={'0':'236,232,223','1':'201,119,83','-1':'90,158,196'};
let W=0,H=0,vista=0,foco=null,prog=reducido?N:-1,t0=null,visible=true,arrancado=reducido,parts=[],cand=[],ultimo=0,paso=1,izq=0;
const POS=caps.map(()=>[]),EL=caps.map(()=>[]);
const tipo=c=>c.id==='salida'?'salida':(c.nodos[0].img?'img':'punto');
function arco(p){const r=29,C=2*Math.PI*r;return `<svg viewBox="0 0 66 66"><circle cx="33" cy="33" r="${r}" class="pista"/><circle cx="33" cy="33" r="${r}" class="valor" stroke-dasharray="${(p*C).toFixed(1)} ${C.toFixed(1)}" transform="rotate(-90 33 33)"/></svg><span>${Math.round(p*100)}<i>%</i></span>`}
caps.forEach((c,k)=>{
  const t=document.createElement('div');t.className='col-tit';t.innerHTML=`${c.titulo}<small>${c.forma}</small>${c.mostrados?'<small class="m">'+c.mostrados+'</small>':''}`;cont.appendChild(t);c._t=t;
  c.nodos.forEach((n,i)=>{
    const e=document.createElement('div'),ti=tipo(c);e.className='nodo '+ti+(c.pixelado?' px':'');
    if(ti==='img'){const im=document.createElement('img');im.src=n.img[0];im.draggable=false;e.appendChild(im);}
    else if(ti==='punto'){const col=COL[String(n.signo||0)];e.style.background=`rgba(${col},${(.12+.88*n.v).toFixed(2)})`;e.style.borderColor=`rgba(${col},.85)`;}
    else e.innerHTML=arco(n.p);
    if(c.id==='vistas'&&i===0)e.classList.add('sel');
    e.addEventListener('mouseenter',()=>{foco=[k,i];e.classList.add('foco');verTip(c,n,k,i);candidatos();pintarBase();});
    e.addEventListener('mouseleave',()=>{foco=null;e.classList.remove('foco');tip.style.opacity=0;candidatos();pintarBase();});
    cont.appendChild(e);EL[k].push(e);
  });
});
function curva(x1,y1,x2,y2){const p=[],m=(x1+x2)/2;for(let s=0;s<=18;s++){const t=s/18,u=1-t;p.push([u*u*u*x1+3*u*u*t*m+3*u*t*t*m+t*t*t*x2,u*u*u*y1+3*u*u*t*y1+3*u*t*t*y2+t*t*t*y2]);}return p}
function maquetar(){
  const dpr=window.devicePixelRatio||1;W=root.clientWidth;H=root.clientHeight;
  for(const cv of [cB,cF]){cv.width=W*dpr;cv.height=H*dpr;cv.style.width=W+'px';cv.style.height=H+'px';cv.getContext('2d').setTransform(dpr,0,0,dpr,0,0);}
  izq=50;const der=58,arriba=80,abajo=54;paso=(W-izq-der)/(N-1);
  const S=Math.max(26,Math.min(58,paso*.55));
  caps.forEach((c,k)=>{
    const x=izq+k*paso,ti=tipo(c),n=c.nodos.length;
    const tam=ti==='img'?S*(k<2?1.04:(k===2?.9:.8)):ti==='punto'?(c.id==='pool'?11:13):66;
    const alto=H-arriba-abajo,hueco=Math.min(tam+(ti==='img'?11:10),alto/n),y0=arriba+alto/2-(n-1)*hueco/2;
    c._t.style.left=x+'px';
    c.nodos.forEach((nd,i)=>{const y=y0+i*hueco;POS[k][i]={x,y,r:tam/2};const e=EL[k][i];
      e.style.left=x+'px';e.style.top=y+'px';e.style.width=tam+'px';e.style.height=tam+'px';});
  });
  AR.forEach((l,k)=>l.forEach(a=>{const p=POS[k][a[0]],q=POS[k+1][a[1]];a.pts=curva(p.x+p.r+2,p.y,q.x-q.r-2,q.y);}));
  candidatos();pintarBase();activar();
}
function toca(k,a){if(!foco)return null;return (foco[0]===k&&foco[1]===a[0])||(foco[0]===k+1&&foco[1]===a[1])}
function alfa(k,a){let al=(a[3]===0?.035:.05)+(a[3]===0?.38:.5)*Math.pow(a[2],1.25);if(a.length>4&&a[4]!==vista)al*=.15;const t=toca(k,a);
  if(t===true)al=.35+.65*a[2];else if(t===false)al*=.2;return al}
function pintarBase(){
  gB.clearRect(0,0,W,H);
  AR.forEach((l,k)=>{const f=Math.max(0,Math.min(1,prog-k));if(f<=0)return;
    l.forEach(a=>{const p=a.pts,m=f*(p.length-1),e=Math.floor(m);
      gB.strokeStyle=`rgba(${COL[String(a[3])]},${alfa(k,a).toFixed(3)})`;gB.lineWidth=.45+2.3*a[2];
      gB.beginPath();gB.moveTo(p[0][0],p[0][1]);for(let s=1;s<=e;s++)gB.lineTo(p[s][0],p[s][1]);
      if(e<p.length-1&&m>e){const A=p[e],B=p[e+1],u=m-e;gB.lineTo(A[0]+(B[0]-A[0])*u,A[1]+(B[1]-A[1])*u);}
      gB.stroke();});});
}
function activar(){caps.forEach((c,k)=>{const on=prog>=k-.02;c._t.classList.toggle('on',on);EL[k].forEach(e=>e.classList.toggle('on',on));});}
function candidatos(){cand=[];AR.forEach((l,k)=>l.forEach(a=>{if(a.length>4&&a[4]!==vista)return;const t=toca(k,a);
  if(t===false)return;if(a[2]>(t?0:.22))cand.push(a);}));}
function verTip(c,n,k,i){
  const im=n.img?`<img src="${n.img[Math.min(vista,n.img.length-1)]}" class="${c.pixelado?'px':''}">`:'';
  tip.innerHTML=`${im}<div><b>${n.tit}</b><span>${n.txt}</span></div>`;
  const p=POS[k][i],w=tip.offsetWidth||300,h=tip.offsetHeight||130;
  let x=p.x+p.r+14;if(x+w>W-8)x=p.x-p.r-14-w;
  tip.style.left=Math.max(8,x)+'px';tip.style.top=Math.max(8,Math.min(H-h-48,p.y-h/2))+'px';tip.style.opacity=1;
}
document.querySelectorAll('button[data-v]').forEach(b=>b.addEventListener('click',()=>{
  vista=+b.dataset.v;document.querySelectorAll('button[data-v]').forEach(o=>o.classList.toggle('on',o===b));
  caps.forEach((c,k)=>c.nodos.forEach((n,i)=>{if(n.img&&n.img.length>1)EL[k][i].firstChild.src=n.img[vista];
    if(c.id==='vistas')EL[k][i].classList.toggle('sel',i===vista);}));
  candidatos();pintarBase();}));
function propagar(){if(reducido){prog=N;activar();pintarBase();return}parts=[];prog=0;t0=null;arrancado=true;activar();}
document.getElementById('play').addEventListener('click',propagar);
function cuadro(ts){
  requestAnimationFrame(cuadro);
  const dt=Math.min(64,ts-(ultimo||ts));ultimo=ts;
  if(!visible||!arrancado)return;
  if(prog<N){if(t0===null)t0=ts-prog/(N-1)*DUR;prog=Math.min(N,(ts-t0)/DUR*(N-1));activar();pintarBase();}
  gF.clearRect(0,0,W,H);
  if(prog<N-1){const x=izq+prog*paso,g=gF.createLinearGradient(x-60,0,x+8,0);
    g.addColorStop(0,'rgba(201,119,83,0)');g.addColorStop(.85,'rgba(236,232,223,.05)');g.addColorStop(1,'rgba(236,232,223,.22)');
    gF.fillStyle=g;gF.fillRect(x-60,46,68,H-100);return;}
  const meta=foco?40:60;
  for(let s=0;s<4&&parts.length<meta&&cand.length;s++){const a=cand[Math.floor(Math.random()*cand.length)];
    if(Math.random()<.2+a[2])parts.push({a,t:0,v:1/(700+600*Math.random())});}
  parts=parts.filter(p=>p.t<1);
  for(const p of parts){p.t+=p.v*dt;const q=p.a.pts,x=Math.min(.999,p.t)*(q.length-1),i=Math.floor(x),f=x-i,A=q[i],B=q[i+1];
    const px=A[0]+(B[0]-A[0])*f,py=A[1]+(B[1]-A[1])*f,c=COL[String(p.a[3])],w=p.a[2];
        gF.fillStyle=`rgba(${c},.8)`;gF.beginPath();gF.arc(px,py,1+1.3*w,0,6.3);gF.fill();}
}
new ResizeObserver(maquetar).observe(root);
new IntersectionObserver(es=>es.forEach(e=>{visible=e.isIntersecting;if(visible&&!arrancado)propagar();}),{threshold:.35}).observe(root);
maquetar();requestAnimationFrame(cuadro);
</script>
"""


def red_neuronal(datos: dict, alto: int = 600) -> None:
    """Red con conexiones: mapas de activación reales de un corte y pesos efectivos entre capas."""
    codigo = (_RED.replace("__FUENTES__", FUENTES).replace("__ALTO__", str(alto))
              .replace("__DATOS__", json.dumps(datos, ensure_ascii=False)))
    st.iframe(codigo, height=alto + 4)


# ---------------------------------------------------------------- arquitectura en bloques 3D

_BLOQUES = r"""
<link rel="stylesheet" href="__FUENTES__">
<div id="b3">
  <div id="lienzo"></div>
  <div id="tip"></div>
  <div id="aviso">No se pudo cargar three.js (requiere conexión a internet).</div>
  <div id="barra">
    <div class="ley">Arrastra para girar · clic y rueda para acercar · pasa el ratón por un bloque</div>
    <div class="ctl"><button data-c="iso" class="on">Perspectiva</button><button data-c="lado">Lateral</button>
      <button data-c="arriba">Superior</button><button id="giro">Girar</button></div>
  </div>
</div>
<style>
:root{--l:#1d252f;--l2:#2a3440;--t:#ece8df;--t2:#a9afb7;--te:#6b7480;--r:#c97753}
html,body{margin:0;background:transparent;color:var(--t);font-family:Geist,system-ui,sans-serif;overflow:hidden}
#b3{position:relative;height:__ALTO__px;border:1px solid var(--l);border-radius:2px;overflow:hidden;
  background:radial-gradient(80% 90% at 50% 40%,#131920 0%,#0b0e12 80%)}
#lienzo{position:absolute;inset:0 0 40px 0;cursor:grab}
#lienzo:active{cursor:grabbing}
#tip{position:absolute;z-index:5;pointer-events:none;max-width:290px;padding:11px 13px;background:rgba(11,14,18,.96);
  border:1px solid var(--l2);box-shadow:0 18px 40px rgba(0,0,0,.55);opacity:0;transition:opacity .12s;font-size:12px;line-height:1.5;color:var(--t2)}
#tip b{display:block;color:var(--t);font-weight:500;font-size:13px;margin-bottom:2px}
#tip .f{font-family:'Geist Mono',monospace;font-size:11px;color:var(--r);margin-bottom:5px}
#tip .p{font-family:'Geist Mono',monospace;font-size:11px;color:var(--te);margin-top:6px}
#aviso{display:none;position:absolute;inset:0;place-items:center;color:var(--te);font:400 12px 'Geist Mono',monospace}
#barra{position:absolute;left:0;right:0;bottom:0;height:40px;display:flex;justify-content:space-between;align-items:center;padding:0 14px;
  border-top:1px solid var(--l);background:rgba(11,14,18,.8);font:400 10.5px 'Geist Mono',monospace;color:var(--te);letter-spacing:.04em}
.ctl{display:flex;gap:4px}
button{font:500 10.5px 'Geist Mono',monospace;color:var(--t2);background:transparent;border:1px solid var(--l2);padding:5px 9px;cursor:pointer;border-radius:2px}
button:hover{border-color:var(--t2);color:var(--t)}
button.on{background:var(--t);color:#0b0e12;border-color:var(--t)}
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const CAPAS=__DATOS__;
const raiz=document.getElementById('b3'),zona=document.getElementById('lienzo'),tip=document.getElementById('tip');
if(typeof THREE==='undefined'||!THREE.OrbitControls){document.getElementById('aviso').style.display='grid';}
else document.fonts.ready.then(iniciar);
function iniciar(){
const COBRE=0xc97753,ACERO=0x5a9ec4,TINTA=0xece8df,GRIS=0x6b7480;
const reducido=matchMedia('(prefers-reduced-motion: reduce)').matches;
const escena=new THREE.Scene();escena.fog=new THREE.Fog(0x0b0e12,420,900);
const camara=new THREE.PerspectiveCamera(32,1,1,3000);
const rend=new THREE.WebGLRenderer({antialias:true,alpha:true});rend.setPixelRatio(Math.min(2,devicePixelRatio));
zona.appendChild(rend.domElement);
const ctl=new THREE.OrbitControls(camara,rend.domElement);ctl.enableDamping=true;ctl.dampingFactor=.08;
ctl.minDistance=120;ctl.maxDistance=900;ctl.autoRotateSpeed=.6;ctl.enableZoom=false;
// La rueda solo acerca tras hacer clic en la escena, para no secuestrar el scroll de la página.
rend.domElement.addEventListener('pointerdown',()=>{ctl.enableZoom=true;});
rend.domElement.addEventListener('pointerleave',()=>{ctl.enableZoom=false;});
escena.add(new THREE.AmbientLight(0xffffff,.75));const sol=new THREE.DirectionalLight(0xffffff,.55);sol.position.set(-120,220,180);escena.add(sol);
const lado=s=>16+9*Math.log2(s/8),fondo=c=>3+2.3*Math.sqrt(c),HUECO=30;
const cargador=new THREE.TextureLoader(),selec=[],conos=[];let x=0;
function rotulo(t1,t2){const c=document.createElement('canvas');c.width=640;c.height=150;const g=c.getContext('2d');
  g.textAlign='center';g.fillStyle='#ece8df';g.font='500 46px Geist, sans-serif';g.fillText(t1,320,60);
  g.fillStyle='#8d949d';g.font='34px "Geist Mono", monospace';g.fillText(t2,320,118);
  const tx=new THREE.CanvasTexture(c);tx.minFilter=THREE.LinearFilter;
  const sp=new THREE.Sprite(new THREE.SpriteMaterial({map:tx,transparent:true,depthWrite:false}));sp.scale.set(52,12.2,1);return sp;}
function caja(w,h,d,cx,cy){const g=new THREE.BoxGeometry(w,h,d);
  const m=new THREE.Mesh(g,new THREE.MeshStandardMaterial({color:0x9aa3ad,transparent:true,opacity:.05,depthWrite:false,roughness:.9}));
  const b=new THREE.LineSegments(new THREE.EdgesGeometry(g),new THREE.LineBasicMaterial({color:GRIS,transparent:true,opacity:.65}));
  m.position.set(cx,cy||0,0);b.position.copy(m.position);escena.add(m,b);return [m,b];}
const geo=[];
CAPAS.forEach((c,i)=>{
  let info;
  if(c.tipo==='mapas'){
    const L=lado(c.lado),D=fondo(c.canales),n=c.imgs.length;
    const [m,b]=caja(D,L,L,x+D/2);
    c.imgs.forEach((u,k)=>{const t=cargador.load(u);t.minFilter=THREE.LinearFilter;t.generateMipmaps=false;
      if(c.pixelado)t.magFilter=THREE.NearestFilter;
      const pl=new THREE.Mesh(new THREE.PlaneGeometry(L*.94,L*.94),new THREE.MeshBasicMaterial({map:t,side:THREE.DoubleSide,transparent:true,opacity:.96}));
      pl.rotation.y=-Math.PI/2;pl.position.set(x+(n===1?D/2:D*(.08+.84*k/(n-1))),0,0);escena.add(pl);});
    info={m,b,x0:x,x1:x+D,L};x+=D;
  }else if(c.tipo==='barra'){
    const n=c.valores.length,h=4.2,L=n*h;const [m,b]=caja(5,L+4,5,x+2.5);
    c.valores.forEach((v,k)=>{const q=new THREE.Mesh(new THREE.BoxGeometry(3.4,h*.78,3.4),
      new THREE.MeshStandardMaterial({color:TINTA,transparent:true,opacity:.15+.85*v,roughness:.7}));
      q.position.set(x+2.5,L/2-h/2-k*h,0);escena.add(q);});
    info={m,b,x0:x,x1:x+5,L};x+=5;
  }else if(c.tipo==='neuronas'){
    const n=c.valores.length,h=5.2,L=n*h;const [m,b]=caja(6,L+4,6,x+3);
    c.valores.forEach((v,k)=>{const col=c.signos[k]>0?COBRE:(c.signos[k]<0?ACERO:TINTA);
      const e=new THREE.Mesh(new THREE.SphereGeometry(1.9,18,12),new THREE.MeshStandardMaterial({color:col,transparent:true,opacity:.25+.75*v,roughness:.5}));
      e.position.set(x+3,L/2-h/2-k*h,0);escena.add(e);});
    info={m,b,x0:x,x1:x+6,L};x+=6;
  }else{
    const r=7,e=new THREE.Mesh(new THREE.SphereGeometry(r,32,20),new THREE.MeshStandardMaterial({color:COBRE,roughness:.45,metalness:.1,
      emissive:COBRE,emissiveIntensity:.15+.5*c.p}));
    e.position.set(x+r,0,0);escena.add(e);
    const b=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(2*r+3,2*r+3,2*r+3)),new THREE.LineBasicMaterial({color:GRIS,transparent:true,opacity:0}));
    b.position.copy(e.position);escena.add(b);
    const t=rotulo(Math.round(100*c.p)+' %','p del corte');t.position.set(x+r,-r-9,0);escena.add(t);
    info={m:e,b,x0:x,x1:x+2*r,L:2*r};x+=2*r;
  }
  info.m.userData.i=i;selec.push(info.m);geo.push(info);
  const r=rotulo(c.nombre,c.forma.replace(/ × /g,'×'));r.position.set((info.x0+info.x1)/2,info.L/2+14,0);escena.add(r);
  x+=HUECO;
});
const total=x-HUECO,cx=total/2;escena.position.x=-cx;
const rejilla=new THREE.GridHelper(Math.max(600,total*1.6),40,0x1d252f,0x141a20);rejilla.position.set(cx,-62,0);escena.add(rejilla);
// Conos de convolución entre bloques y abanicos hacia las capas densas.
const matCono=new THREE.LineBasicMaterial({color:COBRE,transparent:true,opacity:.55});
for(let i=0;i<geo.length-1;i++){
  const a=geo[i],b=geo[i+1];
  if(CAPAS[i].tipo==='mapas'&&CAPAS[i+1].tipo==='mapas'){
    const q=a.L*.13,oy=a.L*(.18*Math.sin(i*1.7)),oz=a.L*(.16*Math.cos(i*2.3)),ax=a.x1,bx=b.x0;
    const p=[[ax,oy-q,oz-q],[ax,oy+q,oz-q],[ax,oy+q,oz+q],[ax,oy-q,oz+q]],v=[bx,oy*b.L/a.L*.7,oz*b.L/a.L*.7];
    const pts=[];p.forEach((u,k)=>{pts.push(...u,...v);const w=p[(k+1)%4];pts.push(...u,...w);});
    const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(pts,3));
    escena.add(new THREE.LineSegments(g,matCono));
    const pulso=new THREE.Mesh(new THREE.PlaneGeometry(1,1),new THREE.MeshBasicMaterial({color:COBRE,transparent:true,opacity:.35,side:THREE.DoubleSide,depthWrite:false}));
    pulso.rotation.y=-Math.PI/2;escena.add(pulso);conos.push({pulso,ax,bx,oy,oz,v,q,fase:i*.37});
  }else{
    const pts=[],n=CAPAS[i+1].tipo==='salida'?1:6;
    for(let k=0;k<n;k++){const ya=(k/(Math.max(1,n-1))-.5)*a.L*.8,yb=n===1?0:(k/(n-1)-.5)*b.L*.8;
      for(let j=0;j<3;j++){const y0=(j/2-.5)*a.L*.8;pts.push(a.x1,y0,0,b.x0,yb,0);}}
    const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(pts,3));
    escena.add(new THREE.LineSegments(g,new THREE.LineBasicMaterial({color:TINTA,transparent:true,opacity:.12})));
  }
}
const VISTAS={iso:[-0.14*total,0.22*total,0.86*total],lado:[0,8,0.92*total],arriba:[0,0.95*total,30]};
let destino=null;
function vista(k){destino=new THREE.Vector3(...VISTAS[k]);document.querySelectorAll('button[data-c]').forEach(b=>b.classList.toggle('on',b.dataset.c===k));}
document.querySelectorAll('button[data-c]').forEach(b=>b.addEventListener('click',()=>vista(b.dataset.c)));
const giro=document.getElementById('giro');giro.addEventListener('click',()=>{ctl.autoRotate=!ctl.autoRotate;giro.classList.toggle('on',ctl.autoRotate);});
camara.position.set(-0.55*total,0.3*total,0.85*total);vista('iso');if(reducido)camara.position.copy(destino);
ctl.addEventListener('start',()=>{destino=null;});
function medir(){const w=zona.clientWidth,h=zona.clientHeight;rend.setSize(w,h);camara.aspect=w/h;camara.updateProjectionMatrix();}
new ResizeObserver(medir).observe(zona);medir();
const ray=new THREE.Raycaster(),ptr=new THREE.Vector2();let foco=null;
rend.domElement.addEventListener('pointermove',e=>{const r=rend.domElement.getBoundingClientRect();
  ptr.set((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1);ray.setFromCamera(ptr,camara);
  const h=ray.intersectObjects(selec)[0];const i=h?h.object.userData.i:null;
  if(foco!==null&&foco!==i){geo[foco].b.material.color.setHex(GRIS);geo[foco].b.material.opacity=CAPAS[foco].tipo==='salida'?0:.65;}
  foco=i;if(i===null){tip.style.opacity=0;return;}
  geo[i].b.material.color.setHex(COBRE);geo[i].b.material.opacity=1;const c=CAPAS[i];
  tip.innerHTML=`<b>${c.nombre}</b><div class="f">${c.forma}</div>${c.detalle}${c.params?`<div class="p">Parámetros: ${c.params}</div>`:''}${c.mostrados?`<div class="p">Se muestran ${c.mostrados} mapas reales del corte</div>`:''}`;
  const w=tip.offsetWidth,hh=tip.offsetHeight;let tx=e.clientX-r.left+16;if(tx+w>r.width-8)tx=e.clientX-r.left-16-w;
  tip.style.left=tx+'px';tip.style.top=Math.max(8,Math.min(r.height-hh-8,e.clientY-r.top-hh/2))+'px';tip.style.opacity=1;});
rend.domElement.addEventListener('pointerleave',()=>{tip.style.opacity=0;});
let visible=true;new IntersectionObserver(es=>es.forEach(e=>visible=e.isIntersecting)).observe(raiz);
const reloj=new THREE.Clock();
(function bucle(){requestAnimationFrame(bucle);if(!visible)return;const t=reloj.getElapsedTime();
  if(destino){camara.position.lerp(destino,.06);if(camara.position.distanceTo(destino)<1)destino=null;}
  if(!reducido)conos.forEach(c=>{const u=((t*.45+c.fase)%1),s=c.q*2*(1-u)+.6;
    c.pulso.position.set(c.ax+(c.bx-c.ax)*u,c.oy+(c.v[1]-c.oy)*u,c.oz+(c.v[2]-c.oz)*u);c.pulso.scale.set(s,s,1);c.pulso.material.opacity=.4*(1-u);});
  ctl.update();rend.render(escena,camara);})();
}
</script>
"""


def bloques_3d(capas: list[dict], alto: int = 560) -> None:
    """Arquitectura en bloques 3D (three.js): mapas reales como texturas, conos de convolución y hover."""
    codigo = (_BLOQUES.replace("__FUENTES__", FUENTES).replace("__ALTO__", str(alto))
              .replace("__DATOS__", json.dumps(capas, ensure_ascii=False)))
    st.iframe(codigo, height=alto + 4)


# ---------------------------------------------------------------- visor de cortes

_VISOR = r"""
<link rel="stylesheet" href="__FUENTES__">
<div id="v">
  <div id="vp">
    <img id="im0" alt=""><img id="im1" alt="">
    <svg id="vent" viewBox="0 0 256 256" preserveAspectRatio="none"></svg>
    <div class="esq tl" id="tl"></div><div class="esq tr" id="tr"></div>
    <div class="esq bl">256 × 256 · 8 bit<br>rueda o ↑↓: cambiar corte</div><div class="esq br" id="br"></div>
  </div>
  <div id="panel">
    <div class="et">Fase</div>
    <div class="seg" id="fases"><button data-f="PRE">PRE</button><button data-f="EARLY" class="on">EARLY</button><button data-f="LATE">LATE</button></div>
    <div class="et">Realce EARLY − PRE <b id="opv">55 %</b></div>
    <input type="range" id="op" min="0" max="100" value="55">
    <label class="chk"><input type="checkbox" id="chv" checked><span></span>Ventanas que mira la red</label>
    <div class="et" style="margin-top:14px">Cortes <b>p calibrada por corte</b></div>
    <div id="tira"></div>
  </div>
</div>
<style>
:root{--l:#1d252f;--l2:#2a3440;--t:#ece8df;--t2:#a9afb7;--te:#6b7480;--r:#c97753}
html,body{margin:0;background:transparent;color:var(--t);font-family:Geist,system-ui,sans-serif}
#v{display:grid;grid-template-columns:minmax(0,__LADO__px) minmax(250px,1fr);gap:18px;height:__ALTO__px}
#vp{position:relative;aspect-ratio:1;max-height:__ALTO__px;background:#000;border:1px solid var(--l2);overflow:hidden;cursor:ns-resize}
#vp img{position:absolute;inset:0;width:100%;height:100%;image-rendering:auto;user-select:none;-webkit-user-drag:none}
#im1{transition:opacity .15s}
#vent{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
#vent rect{fill:none;stroke-width:.7;vector-effect:non-scaling-stroke}
#vent text{font:500 6px 'Geist Mono',monospace;letter-spacing:.5px}
.esq{position:absolute;font:400 10.5px/1.5 'Geist Mono',monospace;letter-spacing:.06em;color:rgba(236,232,223,.78);text-shadow:0 1px 2px #000;text-transform:uppercase;pointer-events:none}
.tl{top:9px;left:11px}.tr{top:9px;right:11px;text-align:right}.bl{bottom:9px;left:11px;color:rgba(236,232,223,.5)}.br{bottom:9px;right:11px;text-align:right}
.esq .o{color:var(--r)}
#panel{display:flex;flex-direction:column;min-height:0}
.et{font:500 10px 'Geist Mono',monospace;letter-spacing:.16em;text-transform:uppercase;color:var(--te);margin:0 0 8px;display:flex;justify-content:space-between}
.et b{font-weight:500;color:var(--t2);letter-spacing:.04em;text-transform:none}
.seg{display:grid;grid-template-columns:repeat(3,1fr);border:1px solid var(--l2);margin-bottom:16px}
.seg button{font:500 11px 'Geist Mono',monospace;letter-spacing:.1em;background:transparent;color:var(--t2);border:0;padding:8px 0;cursor:pointer}
.seg button+button{border-left:1px solid var(--l2)}
.seg button.on{background:var(--t);color:#0b0e12}
input[type=range]{width:100%;accent-color:var(--r);margin:0 0 14px}
.chk{display:flex;align-items:center;gap:9px;font-size:13px;color:var(--t2);cursor:pointer;user-select:none}
.chk input{display:none}.chk span{width:28px;height:15px;border-radius:9px;background:var(--l2);position:relative;transition:.2s}
.chk span::after{content:"";position:absolute;top:2px;left:2px;width:11px;height:11px;border-radius:50%;background:var(--t2);transition:.2s}
.chk input:checked+span{background:var(--r)}.chk input:checked+span::after{left:15px;background:#0b0e12}
#tira{flex:1;overflow-y:auto;position:relative;border-top:1px solid var(--l);scrollbar-width:thin;scrollbar-color:var(--l2) transparent}
.f{display:grid;grid-template-columns:38px 54px 1fr 46px;gap:10px;align-items:center;padding:6px 6px 6px 0;border-bottom:1px solid var(--l);cursor:pointer;transition:background .15s}
.f:hover{background:rgba(236,232,223,.03)}
.f.on{background:rgba(201,119,83,.08);box-shadow:inset 2px 0 0 var(--r)}
.f img{width:38px;height:38px;display:block;border:1px solid var(--l2)}
.f .n{font:400 11px 'Geist Mono',monospace;color:var(--t2)}
.f .b{height:4px;background:var(--l);position:relative}
.f .b i{position:absolute;left:0;top:0;bottom:0;background:var(--r);opacity:.85}
.f .p{font:500 11.5px 'Geist Mono',monospace;text-align:right;color:var(--t)}
</style>
<script>
const C=__CORTES__,M=__META__;
let i=__INICIAL__,fase='EARLY';
const im0=document.getElementById('im0'),im1=document.getElementById('im1'),op=document.getElementById('op'),vp=document.getElementById('vp');
const tira=document.getElementById('tira'),pmax=Math.max(.6,...C.map(c=>c.p));
const pc=v=>(100*v).toFixed(1).replace('.',',')+' %';
const vent=document.getElementById('vent');
[[256,'#a9afb7','256'],[128,'#a9afb7','128'],[64,'#c97753','64']].forEach(([s,c,t])=>{const o=(256-s)/2;
  vent.insertAdjacentHTML('beforeend',`<rect x="${o+.5}" y="${o+.5}" width="${s-1}" height="${s-1}" stroke="${c}" stroke-dasharray="${s===256?'0':'3 2'}" opacity="${s===256?0:.85}"/><text x="${o+3}" y="${o+8}" fill="${c}">${t===`256`?'':t+' px'}</text>`);});
C.forEach((c,k)=>{const f=document.createElement('div');f.className='f';
  f.innerHTML=`<img src="${c.EARLY}"><span class="n">${c.nombre}</span><span class="b"><i style="width:${(100*c.p/pmax).toFixed(1)}%"></i></span><span class="p">${pc(c.p)}</span>`;
  f.addEventListener('click',()=>ir(k));tira.appendChild(f);});
function pintar(){const c=C[i];im0.src=c[fase];im1.src=c.REALCE;im1.style.opacity=op.value/100;im0.style.filter=`brightness(${(1-.4*op.value/100).toFixed(2)})`;
  document.getElementById('opv').textContent=op.value+' %';
  document.getElementById('tl').innerHTML=`${M.titulo}<br>corte ${c.nombre} · ${i+1}/${C.length}`;
  document.getElementById('tr').innerHTML=`fase ${fase}<br><span class="o">realce ${op.value} %</span>`;
  document.getElementById('br').innerHTML=`p corte<br><span class="o">${pc(c.p)}</span>`;
  [...tira.children].forEach((f,k)=>f.classList.toggle('on',k===i));}
function ir(k){i=Math.max(0,Math.min(C.length-1,k));pintar();const f=tira.children[i];
  if(f.offsetTop<tira.scrollTop||f.offsetTop+f.offsetHeight>tira.scrollTop+tira.clientHeight)tira.scrollTop=f.offsetTop-tira.clientHeight/2;}
document.querySelectorAll('#fases button').forEach(b=>b.addEventListener('click',()=>{fase=b.dataset.f;
  document.querySelectorAll('#fases button').forEach(o=>o.classList.toggle('on',o===b));pintar();}));
op.addEventListener('input',pintar);
document.getElementById('chv').addEventListener('change',e=>vent.style.display=e.target.checked?'':'none');
vp.addEventListener('wheel',e=>{e.preventDefault();ir(i+(e.deltaY>0?1:-1));},{passive:false});
document.addEventListener('keydown',e=>{if(e.key==='ArrowDown')ir(i+1);if(e.key==='ArrowUp')ir(i-1);});
ir(i);
</script>
"""


def visor(cortes: list[dict], titulo: str, alto: int = 500, inicial: int = 0) -> None:
    """Visor tipo estación de lectura: fases, mapa de realce, ventanas de la red y p por corte."""
    codigo = (_VISOR.replace("__FUENTES__", FUENTES).replace("__ALTO__", str(alto)).replace("__LADO__", str(alto))
              .replace("__CORTES__", json.dumps(cortes)).replace("__META__", json.dumps({"titulo": titulo}))
              .replace("__INICIAL__", str(inicial)))
    st.iframe(codigo, height=alto + 4)


# ---------------------------------------------------------------- gráficas (Altair, con hover)

def _base(grafica: alt.Chart, alto: int = 300) -> alt.Chart:
    return (grafica.properties(height=alto, background="transparent", padding={"left": 4, "right": 8, "top": 6, "bottom": 4})
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor="#8d949d", titleColor="#8d949d", gridColor="#161d25", domainColor="#2a3440",
                            tickColor="#2a3440", labelFont="Geist Mono", titleFont="Geist", labelFontSize=10.5,
                            titleFontSize=11.5, titleFontWeight=400, titlePadding=10, labelPadding=6)
            .configure_legend(labelColor=TINTA2, orient="top", labelFont="Geist", title=None, symbolStrokeWidth=3,
                              labelFontSize=11.5, symbolSize=120, symbolOpacity=1)
            .configure_title(font="Instrument Serif", fontSize=21, color=TINTA, anchor="start", fontWeight=400, offset=14)
            .configure_text(font="Geist Mono"))


def grafica(chart: alt.Chart) -> None:
    st.altair_chart(chart, width="stretch", theme=None)


def curvas_epoca(curvas: list[dict], clave_fit: str, clave_sel: str, titulo: str, eje: str,
                 dominio=None, marca_x: int | None = None) -> alt.Chart:
    filas = []
    for c in curvas:
        for clave, serie in ((clave_fit, "Entrenamiento"), (clave_sel, "Validación")):
            filas.append({"época": c["epoca"], "serie": serie, "valor": c[clave],
                          "p25": c[clave + "_p25"], "p75": c[clave + "_p75"], "redes": c["n"]})
    df = pd.DataFrame(filas)
    color = alt.Color("serie:N", scale=alt.Scale(domain=["Entrenamiento", "Validación"], range=[ACERO, COBRE]))
    x = alt.X("época:Q", axis=alt.Axis(tickMinStep=1, values=list(range(1, 13)), grid=False), scale=alt.Scale(domain=[1, 12]))
    y = alt.Y("valor:Q", title=eje, scale=alt.Scale(domain=dominio, zero=False) if dominio else alt.Scale(zero=False))
    banda = alt.Chart(df).mark_area(opacity=0.16).encode(x=x, y="p25:Q", y2="p75:Q", color=color)
    linea = alt.Chart(df).mark_line(strokeWidth=2).encode(x=x, y=y, color=color)
    sel = alt.selection_point(fields=["época"], nearest=True, on="pointerover", empty=False, clear="pointerout")
    puntos = alt.Chart(df).mark_circle(size=80, stroke=PANEL, strokeWidth=2).encode(
        x=x, y=y, color=color, opacity=alt.condition(sel, alt.value(1), alt.value(0.0)),
        tooltip=[alt.Tooltip("época:Q"), alt.Tooltip("serie:N"), alt.Tooltip("valor:Q", format=".4f"),
                 alt.Tooltip("p25:Q", title="percentil 25", format=".4f"),
                 alt.Tooltip("p75:Q", title="percentil 75", format=".4f"), alt.Tooltip("redes:Q")]
    ).add_params(sel)
    regla = alt.Chart(df).mark_rule(color=TENUE, strokeDash=[2, 3]).encode(x=x).transform_filter(sel)
    capas = [banda, linea, regla, puntos]
    if marca_x:
        capas.append(alt.Chart(pd.DataFrame({"x": [marca_x]})).mark_rule(color=TINTA2, strokeWidth=1, strokeDash=[1, 3])
                     .encode(x="x:Q"))
    return _base(alt.layer(*capas).properties(title=titulo))


def curvas_comparadas(control: list[dict], actual: list[dict]) -> alt.Chart:
    filas = []
    for nombre, curvas in (("Receta antigua (control)", control), ("party_final_v2", actual)):
        for c in curvas:
            if c["epoca"] > 13:
                continue
            filas.append({"época": c["epoca"], "receta": nombre, "brecha": c["fit_auc"] - c["sel_auc"],
                          "AUC entrenamiento": c["fit_auc"], "AUC validación": c["sel_auc"]})
    df = pd.DataFrame(filas)
    sel = alt.selection_point(fields=["época"], nearest=True, on="pointerover", empty=False, clear="pointerout")
    color = alt.Color("receta:N", scale=alt.Scale(domain=["Receta antigua (control)", "party_final_v2"],
                                                  range=[GRIS, COBRE]))
    base = alt.Chart(df).encode(x=alt.X("época:Q", axis=alt.Axis(tickMinStep=1, grid=False)), color=color)
    linea = base.mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=36, filled=True)).encode(
        y=alt.Y("brecha:Q", title="AUC entrenamiento − AUC validación"))
    puntos = base.mark_circle(size=100, stroke=PANEL, strokeWidth=2).encode(
        y="brecha:Q", opacity=alt.condition(sel, alt.value(1), alt.value(0)),
        tooltip=["receta:N", "época:Q", alt.Tooltip("AUC entrenamiento:Q", format=".3f"),
                 alt.Tooltip("AUC validación:Q", format=".3f"), alt.Tooltip("brecha:Q", format=".3f")]
    ).add_params(sel)
    cero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=TENUE, strokeDash=[2, 3]).encode(y="y:Q")
    return _base(alt.layer(cero, linea, puntos).properties(title="Brecha de memorización por época"), 280)


def roc(curvas: dict[str, list], colores: dict[str, str], titulo: str = "Curva ROC") -> alt.Chart:
    filas = [{"modelo": n, "1 − especificidad": x, "sensibilidad": y} for n, pts in curvas.items() for x, y in pts]
    df = pd.DataFrame(filas)
    color = alt.Color("modelo:N", scale=alt.Scale(domain=list(colores), range=list(colores.values())))
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color=TENUE, strokeDash=[2, 3]).encode(x="x:Q", y="y:Q")
    linea = alt.Chart(df).mark_line(strokeWidth=2, interpolate="step-after").encode(
        x=alt.X("1 − especificidad:Q", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("sensibilidad:Q", scale=alt.Scale(domain=[0, 1])), color=color,
        tooltip=["modelo:N", alt.Tooltip("1 − especificidad:Q", format=".2f"), alt.Tooltip("sensibilidad:Q", format=".2f")])
    return _base(alt.layer(diag, linea).properties(title=titulo), 360)


def pr(puntos: list, prevalencia: float) -> alt.Chart:
    df = pd.DataFrame(puntos, columns=["sensibilidad", "precisión"])
    area = alt.Chart(df).mark_area(color=COBRE, opacity=0.08).encode(x="sensibilidad:Q", y="precisión:Q")
    linea = alt.Chart(df).mark_line(strokeWidth=2, color=COBRE).encode(
        x=alt.X("sensibilidad:Q", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("precisión:Q", scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("sensibilidad:Q", format=".2f"), alt.Tooltip("precisión:Q", format=".2f")])
    azar = alt.Chart(pd.DataFrame({"y": [prevalencia], "t": [f"azar = prevalencia ({prevalencia:.2f})"]})).mark_rule(
        color=TENUE, strokeDash=[2, 3]).encode(y="y:Q", tooltip=["t:N"])
    return _base(alt.layer(area, azar, linea).properties(title="Precisión–sensibilidad"), 360)


def matriz(r: dict) -> alt.Chart:
    filas = []
    for real, n_real in (("no pCR", r["TN"] + r["FP"]), ("pCR", r["TP"] + r["FN"])):
        for pred in ("no pCR", "pCR"):
            k = {("no pCR", "no pCR"): "TN", ("no pCR", "pCR"): "FP", ("pCR", "no pCR"): "FN", ("pCR", "pCR"): "TP"}[(real, pred)]
            nombre = {"TN": "Verdaderos negativos", "FP": "Falsos positivos", "FN": "Falsos negativos", "TP": "Verdaderos positivos"}[k]
            filas.append({"real": real, "predicción": pred, "n": r[k], "fila": r[k] / n_real, "tipo": nombre,
                          "texto": f"{r[k]}  ·  {round(100 * r[k] / n_real)} %"})
    df = pd.DataFrame(filas)
    base = alt.Chart(df).encode(x=alt.X("predicción:N", sort=["no pCR", "pCR"], title="Predicción", axis=alt.Axis(labelAngle=0)),
                                y=alt.Y("real:N", sort=["no pCR", "pCR"], title="Realidad"))
    celdas = base.mark_rect(stroke=PANEL, strokeWidth=4).encode(
        color=alt.Color("fila:Q", scale=alt.Scale(domain=[0, 1], range=["#141b23", COBRE]), legend=None),
        tooltip=["tipo:N", "n:Q", alt.Tooltip("fila:Q", title="% de la fila", format=".1%")])
    texto = base.mark_text(fontSize=15, fontWeight=500).encode(text="texto:N", color=alt.value(TINTA))
    return _base(alt.layer(celdas, texto), 260)


def calibracion(puntos: list[dict]) -> alt.Chart:
    df = pd.DataFrame(puntos)
    lim = max(0.6, df[["predicha", "observada"]].max().max() + 0.05)
    diag = alt.Chart(pd.DataFrame({"x": [0, lim], "y": [0, lim]})).mark_line(color=TENUE, strokeDash=[2, 3]).encode(x="x:Q", y="y:Q")
    linea = alt.Chart(df).mark_line(color=COBRE, strokeWidth=2, point=alt.OverlayMarkDef(size=80, filled=True, stroke=PANEL, strokeWidth=2)).encode(
        x=alt.X("predicha:Q", title="Probabilidad predicha (media del decil)", scale=alt.Scale(domain=[0, lim])),
        y=alt.Y("observada:Q", title="pCR observada en el decil", scale=alt.Scale(domain=[0, lim])),
        tooltip=[alt.Tooltip("predicha:Q", format=".3f"), alt.Tooltip("observada:Q", format=".3f"), "n:Q"])
    return _base(alt.layer(diag, linea).properties(title="Calibración por deciles"), 320)


def bandas_grafica(info: dict, titulo: str) -> alt.Chart:
    df = pd.DataFrame(info["bandas"])
    df["nombre"] = df["banda"].map(NOMBRE_BANDA)
    df["lo"], df["hi"] = df["ic95"].str[0], df["ic95"].str[1]
    df["rango"] = df.apply(lambda f: f"{f.p_min:.2f}–{f.p_max:.2f}", axis=1)
    df["texto"] = df["tasa_pcr"].map(lambda v: f"{100 * v:.0f} %")
    orden = [NOMBRE_BANDA[b] for b in ("baja", "intermedia", "alta")]
    x = alt.X("nombre:N", sort=orden, title=None, axis=alt.Axis(labelAngle=0, labelFont="Geist", labelFontSize=11.5))
    barras = alt.Chart(df).mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2, size=64).encode(
        x=x, y=alt.Y("tasa_pcr:Q", title="pCR observada", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 0.8])),
        color=alt.Color("banda:N", scale=alt.Scale(domain=list(BANDAS), range=list(BANDAS.values())), legend=None),
        tooltip=[alt.Tooltip("nombre:N", title="banda"), alt.Tooltip("rango:N", title="probabilidad predicha"),
                 alt.Tooltip("n:Q", title="pacientes"), alt.Tooltip("pcr:Q", title="con pCR"),
                 alt.Tooltip("tasa_pcr:Q", title="pCR observada", format=".1%"),
                 alt.Tooltip("lo:Q", title="IC95 % inferior", format=".1%"),
                 alt.Tooltip("hi:Q", title="IC95 % superior", format=".1%")])
    error = alt.Chart(df).mark_rule(color=TINTA2, strokeWidth=1.2).encode(x=x, y="lo:Q", y2="hi:Q")
    texto = alt.Chart(df).mark_text(dy=-10, dx=22, fontSize=11, color=TINTA2, align="left").encode(x=x, y="hi:Q", text="texto:N")
    prev = alt.Chart(pd.DataFrame({"y": [info["prevalencia"]]})).mark_rule(color=TENUE, strokeDash=[3, 3]).encode(y="y:Q")
    return _base(alt.layer(barras, error, texto, prev).properties(title=titulo), 300)


def mancuernas(df: pd.DataFrame) -> alt.Chart:
    """df: fila (texto), sin (AUC sin imagen), con (AUC con imagen). Una línea por combinación."""
    largo = df.melt(id_vars="fila", value_vars=["sin", "con"], var_name="imagen", value_name="AUC")
    largo["imagen"] = largo["imagen"].map({"sin": "Sin imagen", "con": "Con imagen"})
    y = alt.Y("fila:N", title=None, sort=df["fila"].tolist(), axis=alt.Axis(labelFont="Geist", labelFontSize=11.5, labelLimit=260))
    regla = alt.Chart(df).mark_rule(color="#3a4552", strokeWidth=2).encode(y=y, x=alt.X("sin:Q", title="AUC media de los 5 folds",
                                                                                        scale=alt.Scale(zero=False)), x2="con:Q")
    puntos = alt.Chart(largo).mark_circle(size=130, opacity=1, stroke=PANEL, strokeWidth=2).encode(
        y=y, x="AUC:Q",
        color=alt.Color("imagen:N", scale=alt.Scale(domain=["Sin imagen", "Con imagen"], range=[GRIS, COBRE])),
        tooltip=["fila:N", "imagen:N", alt.Tooltip("AUC:Q", format=".3f")])
    return _base(alt.layer(regla, puntos), 34 * len(df) + 40)


def pie() -> None:
    st.markdown('<div class="pie"><span>PARTYNET · BREASTDCEDL · CNN 2D PROPIA ENTRENADA DESDE CERO</span>'
                '<span>USO EDUCATIVO · NO ES UN DISPOSITIVO MÉDICO NI SUSTITUYE EL JUICIO CLÍNICO</span></div>',
                unsafe_allow_html=True)


def escapar(texto: str) -> str:
    return html.escape(str(texto))
