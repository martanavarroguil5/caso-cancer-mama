"""Piezas visuales compartidas: estilo, cabeceras, tarjetas, bandas, red 3D y gráficas."""
from __future__ import annotations

import html
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

RECURSOS = Path(__file__).resolve().parent / "recursos"

AZUL, NARANJA, AQUA, AMARILLO, GRIS = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984"
TINTA, TINTA2, BORDE = "#0f1b2d", "#4a5568", "#e3e7ee"
BANDAS = {"baja": "#b7d3f6", "intermedia": "#5598e7", "alta": "#1c5cab"}
NOMBRE_BANDA = {"baja": "Probabilidad baja", "intermedia": "Zona intermedia", "alta": "Probabilidad alta"}


def recurso(nombre: str) -> dict | None:
    ruta = RECURSOS / f"{nombre}.json"
    return json.loads(ruta.read_text()) if ruta.exists() else None


def estilo() -> None:
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root{--fondo:#f6f7f9;--superficie:#fff;--tinta:#0f1b2d;--tinta2:#4a5568;--tenue:#7a8597;
--borde:#e3e7ee;--acento:#2a78d6;--acento-suave:#eaf2fd;--naranja:#eb6834;--sombra:0 1px 2px rgba(15,27,45,.04),0 4px 16px rgba(15,27,45,.06)}
html,body,.stApp,[class*="css"]{font-family:'Inter',system-ui,sans-serif}
.stApp{background:var(--fondo);color:var(--tinta)}
.block-container{max-width:1180px;padding-top:2.2rem;padding-bottom:4rem}
h1,h2,h3,h4{letter-spacing:-.02em;color:var(--tinta)}
[data-testid="stSidebar"]{background:var(--superficie);border-right:1px solid var(--borde)}
[data-testid="stAppDeployButton"]{display:none}
[data-testid="stMetricValue"]{font-variant-numeric:tabular-nums}
.etiqueta{font-size:.72rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--acento);margin-bottom:.35rem}
.cabecera h1{font-size:2.3rem;line-height:1.15;margin:0 0 .6rem 0;font-weight:700}
.cabecera p{font-size:1.05rem;color:var(--tinta2);max-width:820px;line-height:1.6;margin:0}
.cabecera{padding:.4rem 0 1.4rem 0;border-bottom:1px solid var(--borde);margin-bottom:1.6rem}
.seccion{display:flex;gap:.9rem;align-items:baseline;margin:2.4rem 0 .4rem 0}
.seccion .num{font-size:.8rem;font-weight:700;color:var(--acento);background:var(--acento-suave);border-radius:6px;padding:.15rem .5rem;font-variant-numeric:tabular-nums}
.seccion h3{margin:0;font-size:1.35rem;font-weight:650}
.seccion-texto{color:var(--tinta2);max-width:860px;line-height:1.6;margin:.2rem 0 1rem 0}
.rejilla{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin:.4rem 0 1rem 0}
.tarjeta{position:relative;background:var(--superficie);border:1px solid var(--borde);border-radius:12px;padding:16px 18px;box-shadow:var(--sombra);transition:transform .15s ease,box-shadow .15s ease}
.tarjeta:hover{transform:translateY(-2px);box-shadow:0 2px 4px rgba(15,27,45,.05),0 10px 28px rgba(15,27,45,.10)}
.tarjeta .t{font-size:.78rem;color:var(--tenue);font-weight:600;text-transform:uppercase;letter-spacing:.06em}
.tarjeta .v{font-size:1.9rem;font-weight:700;margin:.25rem 0 .15rem 0;font-variant-numeric:tabular-nums;color:var(--tinta)}
.tarjeta .d{font-size:.86rem;color:var(--tinta2);line-height:1.45}
.tarjeta .ayuda{visibility:hidden;opacity:0;position:absolute;left:12px;right:12px;top:calc(100% + 6px);z-index:20;background:var(--tinta);color:#fff;font-size:.82rem;line-height:1.45;padding:10px 12px;border-radius:8px;transition:opacity .15s}
.tarjeta:hover .ayuda{visibility:visible;opacity:1}
.aviso{border-radius:10px;padding:12px 16px;font-size:.92rem;line-height:1.55;margin:.6rem 0 1rem 0;border:1px solid}
.aviso.info{background:#eef5fe;border-color:#cde2fb;color:#163a66}
.aviso.atencion{background:#fff7ec;border-color:#f7dcb4;color:#6a4100}
.aviso.clave{background:var(--superficie);border-color:var(--borde);border-left:4px solid var(--acento);color:var(--tinta)}
.pasos{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:.6rem 0 1.2rem 0}
.paso{background:var(--superficie);border:1px solid var(--borde);border-radius:12px;padding:14px;position:relative;transition:border-color .15s,transform .15s}
.paso:hover{border-color:var(--acento);transform:translateY(-2px)}
.paso .n{width:26px;height:26px;border-radius:50%;background:var(--acento);color:#fff;font-weight:700;font-size:.8rem;display:flex;align-items:center;justify-content:center;margin-bottom:.5rem}
.paso b{display:block;font-size:.95rem;margin-bottom:.2rem}
.paso span{font-size:.84rem;color:var(--tinta2);line-height:1.45}
.resultado{background:var(--superficie);border:1px solid var(--borde);border-radius:14px;padding:20px 22px;box-shadow:var(--sombra)}
.resultado .p{font-size:2.8rem;font-weight:700;font-variant-numeric:tabular-nums;line-height:1}
.resultado .sub{color:var(--tinta2);font-size:.9rem;margin-top:.35rem}
.chip{display:inline-block;font-size:.78rem;font-weight:600;padding:.2rem .6rem;border-radius:999px;margin-right:.35rem}
.bandabar{display:flex;height:16px;border-radius:999px;overflow:visible;position:relative;margin:28px 0 8px 0}
.bandabar .seg{position:relative;height:100%;cursor:help}
.bandabar .seg:first-child{border-radius:999px 0 0 999px}.bandabar .seg:last-child{border-radius:0 999px 999px 0}
.bandabar .seg+.seg{border-left:2px solid #fff}
.bandabar .seg .tip{visibility:hidden;opacity:0;position:absolute;bottom:26px;left:50%;transform:translateX(-50%);width:230px;background:var(--tinta);color:#fff;font-size:.8rem;line-height:1.45;padding:9px 11px;border-radius:8px;z-index:30;transition:opacity .15s}
.bandabar .seg:hover .tip{visibility:visible;opacity:1}
.bandabar .marca{position:absolute;top:-9px;width:4px;height:34px;background:var(--tinta);border-radius:2px;box-shadow:0 0 0 3px #fff}
.bandabar .marca span{position:absolute;top:-24px;left:50%;transform:translateX(-50%);font-size:.78rem;font-weight:700;white-space:nowrap}
.escala{display:flex;justify-content:space-between;font-size:.74rem;color:var(--tenue)}
table.limpia{width:100%;border-collapse:collapse;font-size:.9rem;background:var(--superficie);border-radius:10px;overflow:hidden}
table.limpia th{text-align:left;font-weight:600;color:var(--tinta2);background:#f1f3f7;padding:9px 12px;border-bottom:1px solid var(--borde)}
table.limpia td{padding:9px 12px;border-bottom:1px solid var(--borde);font-variant-numeric:tabular-nums}
table.limpia tr:hover td{background:#f7faff}
.pie{color:var(--tenue);font-size:.8rem;margin-top:3rem;border-top:1px solid var(--borde);padding-top:1rem}
</style>""", unsafe_allow_html=True)


def cabecera(etiqueta: str, titulo: str, texto: str) -> None:
    st.markdown(f'<div class="cabecera"><div class="etiqueta">{etiqueta}</div><h1>{titulo}</h1>'
                f'<p>{texto}</p></div>', unsafe_allow_html=True)


def seccion(numero: str, titulo: str, texto: str | None = None) -> None:
    st.markdown(f'<div class="seccion"><span class="num">{numero}</span><h3>{titulo}</h3></div>',
                unsafe_allow_html=True)
    if texto:
        st.markdown(f'<div class="seccion-texto">{texto}</div>', unsafe_allow_html=True)


def tarjetas(items: list[dict]) -> None:
    """items: t (título), v (valor), d (detalle), ayuda (texto al pasar el ratón)."""
    partes = []
    for it in items:
        ayuda = f'<div class="ayuda">{it["ayuda"]}</div>' if it.get("ayuda") else ""
        partes.append(f'<div class="tarjeta"><div class="t">{it["t"]}</div><div class="v">{it["v"]}</div>'
                      f'<div class="d">{it.get("d", "")}</div>{ayuda}</div>')
    st.markdown(f'<div class="rejilla">{"".join(partes)}</div>', unsafe_allow_html=True)


def aviso(texto: str, tipo: str = "info") -> None:
    st.markdown(f'<div class="aviso {tipo}">{texto}</div>', unsafe_allow_html=True)


def pasos(items: list[tuple[str, str]]) -> None:
    st.markdown('<div class="pasos">' + "".join(
        f'<div class="paso"><div class="n">{i}</div><b>{t}</b><span>{d}</span></div>'
        for i, (t, d) in enumerate(items, 1)) + "</div>", unsafe_allow_html=True)


def tabla(df: pd.DataFrame) -> None:
    st.markdown(df.to_html(index=False, classes="limpia", border=0, escape=False), unsafe_allow_html=True)


def pct(x: float, dec: int = 0) -> str:
    return f"{100 * x:.{dec}f} %".replace(".", ",")


def num(x: float, dec: int = 3) -> str:
    return f"{x:.{dec}f}".replace(".", ",")


def banda_de(p: float, info: dict) -> str:
    c1, c2 = info["cortes"]
    return "baja" if p < c1 else ("intermedia" if p < c2 else "alta")


def barra_bandas(p: float, info: dict, maximo: float | None = None) -> None:
    """Barra de tres bandas (terciles OOF) con la posición de la paciente y hovers."""
    filas = info["bandas"]
    lo = 0.0
    hi = maximo or max(0.6, min(1.0, filas[-1]["p_max"] + 0.05))
    limites = [lo, *info["cortes"], hi]
    segmentos = []
    for f, a, b in zip(filas, limites[:-1], limites[1:]):
        ic = f["ic95"]
        tip = (f"<b>{NOMBRE_BANDA[f['banda']]}</b><br>Probabilidad {num(a, 2)}–{num(b, 2)}<br>"
               f"En la CV: {f['pcr']} pCR de {f['n']} pacientes ({pct(f['tasa_pcr'])}; "
               f"IC95 % {pct(ic[0])}–{pct(ic[1])})")
        segmentos.append(f'<div class="seg" style="width:{100 * (b - a) / (hi - lo):.2f}%;'
                         f'background:{BANDAS[f["banda"]]}"><div class="tip">{tip}</div></div>')
    x = 100 * (min(max(p, lo), hi) - lo) / (hi - lo)
    marca = f'<div class="marca" style="left:calc({x:.2f}% - 2px)"><span>{num(p, 2)}</span></div>'
    st.markdown(f'<div class="bandabar">{"".join(segmentos)}{marca}</div>'
                f'<div class="escala"><span>{num(lo, 1)}</span><span>Probabilidad de pCR</span>'
                f'<span>{num(hi, 1)}</span></div>', unsafe_allow_html=True)


def chip_banda(banda: str) -> str:
    color = "#fff" if banda == "alta" else TINTA
    return (f'<span class="chip" style="background:{BANDAS[banda]};color:{color}">'
            f'{NOMBRE_BANDA[banda]}</span>')


# ---------------------------------------------------------------- red en 3D

def red_3d(capas: list[dict], alto: int = 460) -> None:
    """Capas como cajas 3D (CSS). Arrastrar para girar; pasar el ratón para ver detalles."""
    datos = json.dumps(capas, ensure_ascii=False)
    components.html(f"""
<div id="escena-wrap">
  <div id="escena"></div>
  <div id="tip"></div>
  <div id="ayuda">Arrastra para girar · pasa el ratón por una capa · doble clic para recentrar</div>
</div>
<style>
body{{margin:0;font-family:Inter,system-ui,sans-serif}}
#escena-wrap{{position:relative;height:{alto}px;background:radial-gradient(ellipse at 50% 40%,#ffffff 0%,#eef2f8 70%);
  border:1px solid #e3e7ee;border-radius:14px;overflow:hidden;perspective:1400px;cursor:grab;user-select:none}}
#escena{{position:absolute;left:50%;top:52%;transform-style:preserve-3d}}
.caja{{position:absolute;transform-style:preserve-3d;transition:filter .15s}}
.cara{{position:absolute;border:1px solid rgba(15,27,45,.25);box-sizing:border-box;backface-visibility:visible}}
.caja:hover .cara{{filter:brightness(1.12);border-color:#0f1b2d}}
.rotulo{{position:absolute;font-size:11px;font-weight:600;color:#0f1b2d;white-space:nowrap;transform-style:preserve-3d}}
#tip{{position:absolute;pointer-events:none;background:#0f1b2d;color:#fff;font-size:12.5px;line-height:1.5;
  padding:10px 12px;border-radius:8px;max-width:280px;opacity:0;transition:opacity .12s;z-index:5}}
#ayuda{{position:absolute;bottom:10px;left:0;right:0;text-align:center;font-size:11.5px;color:#7a8597}}
</style>
<script>
const capas = {datos};
const escena = document.getElementById('escena'), tip = document.getElementById('tip');
const wrap = document.getElementById('escena-wrap');
let rx = -18, ry = -32, arrastre = null;
const gap = 34;
let total = capas.reduce((s,c)=>s+c.d+gap,0)-gap, x = -total/2;
function cara(w,h,tr,color,alpha){{
  const f=document.createElement('div');f.className='cara';
  f.style.width=w+'px';f.style.height=h+'px';f.style.marginLeft=(-w/2)+'px';f.style.marginTop=(-h/2)+'px';
  f.style.background=color;f.style.opacity=alpha;f.style.transform=tr;return f;}}
capas.forEach((c,i)=>{{
  const b=document.createElement('div');b.className='caja';
  const cx=x+c.d/2; x+=c.d+gap;
  b.style.transform=`translate3d(${{cx}}px,0,0)`;
  const w=c.d,h=c.h,dz=c.w;
  b.appendChild(cara(dz,h,`rotateY(90deg) translateZ(${{w/2}}px)`,c.color,.92));
  b.appendChild(cara(dz,h,`rotateY(-90deg) translateZ(${{w/2}}px)`,c.color,.92));
  b.appendChild(cara(w,h,`translateZ(${{dz/2}}px)`,c.color,.80));
  b.appendChild(cara(w,h,`rotateY(180deg) translateZ(${{dz/2}}px)`,c.color,.80));
  b.appendChild(cara(w,dz,`rotateX(90deg) translateZ(${{h/2}}px)`,c.color,.95));
  b.appendChild(cara(w,dz,`rotateX(-90deg) translateZ(${{h/2}}px)`,c.color,.95));
  const r=document.createElement('div');r.className='rotulo';r.textContent=c.nombre;
  r.style.transform=`translate3d(-50%,${{h/2+14}}px,${{dz/2}}px)`;b.appendChild(r);
  b.addEventListener('mousemove',e=>{{const q=wrap.getBoundingClientRect();
    tip.innerHTML=`<b>${{c.nombre}}</b><br>${{c.forma}}<br>${{c.detalle}}${{c.params?'<br>Parámetros: '+c.params:''}}`;
    tip.style.left=Math.min(e.clientX-q.left+14,q.width-290)+'px';tip.style.top=(e.clientY-q.top+14)+'px';tip.style.opacity=1;}});
  b.addEventListener('mouseleave',()=>tip.style.opacity=0);
  escena.appendChild(b);}});
function pintar(){{escena.style.transform=`rotateX(${{rx}}deg) rotateY(${{ry}}deg)`;}}
wrap.addEventListener('mousedown',e=>{{arrastre=[e.clientX,e.clientY,rx,ry];wrap.style.cursor='grabbing';}});
window.addEventListener('mouseup',()=>{{arrastre=null;wrap.style.cursor='grab';}});
window.addEventListener('mousemove',e=>{{if(!arrastre)return;ry=arrastre[3]+(e.clientX-arrastre[0])*.4;
  rx=Math.max(-80,Math.min(80,arrastre[2]-(e.clientY-arrastre[1])*.4));pintar();}});
wrap.addEventListener('dblclick',()=>{{rx=-18;ry=-32;pintar();}});
let t0=null;function intro(t){{if(!t0)t0=t;const k=Math.min(1,(t-t0)/1600);ry=-75+43*(1-Math.pow(1-k,3));pintar();if(k<1&&!arrastre)requestAnimationFrame(intro);}}
pintar();requestAnimationFrame(intro);
</script>""", height=alto + 6)


# ---------------------------------------------------------------- gráficas (Altair, con hover)

def _base(grafica: alt.Chart, alto: int = 300) -> alt.Chart:
    return (grafica.properties(height=alto)
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor=TINTA2, titleColor=TINTA2, gridColor="#eef0f4", domainColor=BORDE,
                            tickColor=BORDE, labelFont="Inter", titleFont="Inter", titleFontWeight=500)
            .configure_legend(labelColor=TINTA2, titleColor=TINTA2, orient="top", labelFont="Inter",
                              title=None, symbolStrokeWidth=3)
            .configure_title(font="Inter", fontSize=14, color=TINTA, anchor="start", fontWeight=600))


def curvas_epoca(curvas: list[dict], clave_fit: str, clave_sel: str, titulo: str, eje: str,
                 dominio=None, marca_x: int | None = None) -> alt.Chart:
    filas = []
    for c in curvas:
        for clave, serie in ((clave_fit, "Entrenamiento"), (clave_sel, "Validación")):
            filas.append({"época": c["epoca"], "serie": serie, "valor": c[clave],
                          "p25": c[clave + "_p25"], "p75": c[clave + "_p75"], "redes": c["n"]})
    df = pd.DataFrame(filas)
    color = alt.Color("serie:N", scale=alt.Scale(domain=["Entrenamiento", "Validación"], range=[AZUL, NARANJA]))
    x = alt.X("época:Q", axis=alt.Axis(tickMinStep=1, values=list(range(1, 13))), scale=alt.Scale(domain=[1, 12]))
    y = alt.Y("valor:Q", title=eje, scale=alt.Scale(domain=dominio, zero=False) if dominio else alt.Scale(zero=False))
    banda = alt.Chart(df).mark_area(opacity=0.14).encode(x=x, y="p25:Q", y2="p75:Q", color=color)
    linea = alt.Chart(df).mark_line(strokeWidth=2.5).encode(x=x, y=y, color=color)
    sel = alt.selection_point(fields=["época"], nearest=True, on="pointerover", empty=False, clear="pointerout")
    puntos = alt.Chart(df).mark_circle(size=70).encode(
        x=x, y=y, color=color, opacity=alt.condition(sel, alt.value(1), alt.value(0.0)),
        tooltip=[alt.Tooltip("época:Q"), alt.Tooltip("serie:N"), alt.Tooltip("valor:Q", format=".4f"),
                 alt.Tooltip("p25:Q", title="percentil 25", format=".4f"),
                 alt.Tooltip("p75:Q", title="percentil 75", format=".4f"), alt.Tooltip("redes:Q")]
    ).add_params(sel)
    regla = alt.Chart(df).mark_rule(color=GRIS, strokeDash=[3, 3]).encode(x=x).transform_filter(sel)
    capas = [banda, linea, regla, puntos]
    if marca_x:
        capas.append(alt.Chart(pd.DataFrame({"x": [marca_x]})).mark_rule(color=TINTA2, strokeWidth=1)
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
                                                  range=[GRIS, AZUL]))
    base = alt.Chart(df).encode(x=alt.X("época:Q", axis=alt.Axis(tickMinStep=1)), color=color)
    linea = base.mark_line(strokeWidth=2.5, point=True).encode(y=alt.Y("brecha:Q", title="AUC entrenamiento − AUC validación"))
    puntos = base.mark_circle(size=90).encode(
        y="brecha:Q", opacity=alt.condition(sel, alt.value(1), alt.value(0)),
        tooltip=["receta:N", "época:Q", alt.Tooltip("AUC entrenamiento:Q", format=".3f"),
                 alt.Tooltip("AUC validación:Q", format=".3f"), alt.Tooltip("brecha:Q", format=".3f")]
    ).add_params(sel)
    cero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=GRIS, strokeDash=[2, 2]).encode(y="y:Q")
    return _base(alt.layer(cero, linea, puntos).properties(title="Brecha de memorización por época"), 280)


def roc(curvas: dict[str, list], colores: dict[str, str], titulo: str = "Curva ROC") -> alt.Chart:
    filas = [{"modelo": n, "1 − especificidad": x, "sensibilidad": y} for n, pts in curvas.items() for x, y in pts]
    df = pd.DataFrame(filas)
    color = alt.Color("modelo:N", scale=alt.Scale(domain=list(colores), range=list(colores.values())))
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color=GRIS, strokeDash=[3, 3]).encode(x="x:Q", y="y:Q")
    linea = alt.Chart(df).mark_line(strokeWidth=2.4, interpolate="step-after").encode(
        x=alt.X("1 − especificidad:Q", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("sensibilidad:Q", scale=alt.Scale(domain=[0, 1])), color=color,
        tooltip=["modelo:N", alt.Tooltip("1 − especificidad:Q", format=".2f"), alt.Tooltip("sensibilidad:Q", format=".2f")])
    return _base(alt.layer(diag, linea).properties(title=titulo, width="container"), 360)


def pr(puntos: list, prevalencia: float) -> alt.Chart:
    df = pd.DataFrame(puntos, columns=["sensibilidad", "precisión"])
    linea = alt.Chart(df).mark_line(strokeWidth=2.4, color=AZUL).encode(
        x=alt.X("sensibilidad:Q", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("precisión:Q", scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("sensibilidad:Q", format=".2f"), alt.Tooltip("precisión:Q", format=".2f")])
    azar = alt.Chart(pd.DataFrame({"y": [prevalencia], "t": [f"azar = prevalencia ({prevalencia:.2f})"]})).mark_rule(
        color=GRIS, strokeDash=[3, 3]).encode(y="y:Q", tooltip=["t:N"])
    return _base(alt.layer(azar, linea).properties(title="Curva precisión–sensibilidad"), 360)


def matriz(r: dict) -> alt.Chart:
    filas = []
    for real, n_real in (("no pCR", r["TN"] + r["FP"]), ("pCR", r["TP"] + r["FN"])):
        for pred in ("no pCR", "pCR"):
            k = {("no pCR", "no pCR"): "TN", ("no pCR", "pCR"): "FP", ("pCR", "no pCR"): "FN", ("pCR", "pCR"): "TP"}[(real, pred)]
            nombre = {"TN": "Verdaderos negativos", "FP": "Falsos positivos", "FN": "Falsos negativos", "TP": "Verdaderos positivos"}[k]
            filas.append({"real": real, "predicción": pred, "n": r[k], "fila": r[k] / n_real, "tipo": nombre,
                          "texto": f"{r[k]}  ·  {round(100 * r[k] / n_real)} %"})
    df = pd.DataFrame(filas)
    base = alt.Chart(df).encode(x=alt.X("predicción:N", sort=["no pCR", "pCR"], title="Predicción"),
                                y=alt.Y("real:N", sort=["no pCR", "pCR"], title="Realidad"))
    celdas = base.mark_rect(cornerRadius=6, stroke="#fff", strokeWidth=3).encode(
        color=alt.Color("fila:Q", scale=alt.Scale(domain=[0, 1], range=["#eaf2fd", "#1c5cab"]), legend=None),
        tooltip=["tipo:N", "n:Q", alt.Tooltip("fila:Q", title="% de la fila", format=".1%")])
    texto = base.mark_text(fontSize=16, fontWeight=600).encode(
        text="texto:N", color=alt.condition("datum.fila > 0.55", alt.value("white"), alt.value(TINTA)))
    return _base(alt.layer(celdas, texto).properties(width="container"), 260)


def calibracion(puntos: list[dict]) -> alt.Chart:
    df = pd.DataFrame(puntos)
    lim = max(0.6, df[["predicha", "observada"]].max().max() + 0.05)
    diag = alt.Chart(pd.DataFrame({"x": [0, lim], "y": [0, lim]})).mark_line(color=GRIS, strokeDash=[3, 3]).encode(x="x:Q", y="y:Q")
    linea = alt.Chart(df).mark_line(color=AZUL, strokeWidth=2, point=alt.OverlayMarkDef(size=80, filled=True)).encode(
        x=alt.X("predicha:Q", title="Probabilidad predicha (media del decil)", scale=alt.Scale(domain=[0, lim])),
        y=alt.Y("observada:Q", title="pCR observada en el decil", scale=alt.Scale(domain=[0, lim])),
        tooltip=[alt.Tooltip("predicha:Q", format=".3f"), alt.Tooltip("observada:Q", format=".3f"), "n:Q"])
    return _base(alt.layer(diag, linea).properties(title="Calibración por deciles"), 320)


def bandas_grafica(info: dict, titulo: str) -> alt.Chart:
    df = pd.DataFrame(info["bandas"])
    df["nombre"] = df["banda"].map(NOMBRE_BANDA)
    df["lo"], df["hi"] = df["ic95"].str[0], df["ic95"].str[1]
    df["rango"] = df.apply(lambda f: f"{f.p_min:.2f}–{f.p_max:.2f}", axis=1)
    orden = [NOMBRE_BANDA[b] for b in ("baja", "intermedia", "alta")]
    x = alt.X("nombre:N", sort=orden, title=None, axis=alt.Axis(labelAngle=0))
    barras = alt.Chart(df).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, size=70).encode(
        x=x, y=alt.Y("tasa_pcr:Q", title="pCR observada", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 0.8])),
        color=alt.Color("banda:N", scale=alt.Scale(domain=list(BANDAS), range=list(BANDAS.values())), legend=None),
        tooltip=[alt.Tooltip("nombre:N", title="banda"), alt.Tooltip("rango:N", title="probabilidad predicha"),
                 alt.Tooltip("n:Q", title="pacientes"), alt.Tooltip("pcr:Q", title="con pCR"),
                 alt.Tooltip("tasa_pcr:Q", title="pCR observada", format=".1%"),
                 alt.Tooltip("lo:Q", title="IC95 % inferior", format=".1%"),
                 alt.Tooltip("hi:Q", title="IC95 % superior", format=".1%")])
    error = alt.Chart(df).mark_rule(color=TINTA, strokeWidth=1.5).encode(x=x, y="lo:Q", y2="hi:Q")
    prev = alt.Chart(pd.DataFrame({"y": [info["prevalencia"]]})).mark_rule(color=GRIS, strokeDash=[4, 3]).encode(y="y:Q")
    return _base(alt.layer(barras, error, prev).properties(title=titulo), 300)


def pie() -> None:
    st.markdown('<div class="pie">Proyecto docente BreastDCEDL · CNN 2D propia entrenada desde cero · '
                'Uso exclusivamente educativo: no es un dispositivo médico ni sustituye el juicio clínico.</div>',
                unsafe_allow_html=True)


def escapar(texto: str) -> str:
    return html.escape(str(texto))
