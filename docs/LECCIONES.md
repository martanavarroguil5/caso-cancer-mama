# Lecciones de todo el proyecto y diseño de PartyNet

26/10/2026. Síntesis de los ~30 informes de `main` (resultados 01–31,
`docs/EXPERIMENTOS_DESCARTADOS.md`, auditorías) y de la literatura. Los
informes originales siguen en `main`:
`git show main:docs/EXPERIMENTOS_DESCARTADOS.md`.

## 1. Datos

- Train: **1.097 pacientes, 10.945 cortes** (≈10 por paciente, mín. 5), 322 pCR
  (29,4 %). Test público: 176 pacientes, 53 pCR. Validación privada: 177, solo profesorado.
- Tres cohortes con prevalencias distintas: Duke 21 % (209), I-SPY1 25 % (104),
  I-SPY2 32 % (784). **Solo saber la cohorte da AUC media ≈ 0,549.**
- El subtipo es un predictor fuerte de pCR: HR+/HER2− 14 % (453), triple
  negativo 38 % (354), HER2+ 43 % (287). HR/HER2/volumen están en
  `patients.csv` para casi todas las pacientes de train.
- La auditoría de los 32.835 PNG de train no encontró archivos corruptos ni
  duplicados entre pacientes. Incidencias: ISPY1_1139 tiene EARLY = LATE en
  los diez cortes; ISPY1_1193 tiene un LATE anormalmente brillante; 15 pacientes
  tienen posibles traslaciones entre fases. Se dejan tal cual: son 2/1.097
  y cualquier exclusión debería decidirse sin mirar resultados.

## 2. Qué se probó y qué dio

AUC por paciente con la media de probabilidades de los cortes. «Media» es la
media de AUC de los cinco folds exteriores; los protocolos varían, así que la
tabla es descriptiva.

| Familia (fecha) | Media AUC CV | Nota |
|---|---:|---|
| V001 CNN histórica (30/09) | 0,5945 agrupada | Selección con el propio fold; test 0,546 |
| Cuatro mejoras (05/10): control / Dropout2d / afines / EMA / WD | 0,550 / 0,573 / 0,567 / 0,556 / 0,554 | Todas positivas e inconcluyentes por separado |
| BCE por paciente; lotes mezclados; GroupNorm | 0,540; 0,521; 0,480 | Peores |
| V005/V006 CNN por fases (14/10) | 0,533 / 0,549 | Normal / ponderada |
| **V007 cinética multiescala (21/10)** | **0,577 ± 0,030** | La mejor solo imagen; test 0,608 (ya visto) |
| Híbrida de rasgos (25/10) | 0,567 | Test 0,504 |
| Control 27 (selección por pérdida) | 0,556 ± 0,035 | Referencia de las últimas familias |
| Espacial compacta (25/10) | 0,546 | 105 k parámetros |
| Campo de adquisición (25/10, 2 folds) | 0,550 frente a 0,560 | |
| MixStyle (26/10) | 0,558 | |
| Nativa 256 uniforme / atención (26/10) | 0,535 / 0,540 | |
| Multimodal V002 con clínica (07/10) | 0,7246 agrupada | **No admisible como solución**: usa clínica como entrada |

## 3. Patrones que se repiten

1. **Techo de ≈0,55–0,58**, apenas por encima de la referencia de cohorte
   (0,549). Ni la capacidad (105 k–594 k parámetros), ni la resolución, ni la
   atención, ni la normalización de estilo lo han movido.
2. **Sobreajuste temprano**: la época elegida tiene mediana 2–8 y la brecha AUC
   fit−selección es de 0,2–0,27 al final. La red memoriza cortes correlacionados.
3. **Protocolo con poca potencia**: con unas 220 pacientes exteriores por fold,
   la DE entre folds es 0,03–0,05 y el IC95 % pareado de un Δ mide unos ±0,04.
   Para que el IC excluya el 0, un Δ real tiene que rondar 0,04–0,05. Cada familia
   de 20–40 entrenamientos salía «inconcluyente» casi por construcción.
4. **Selección con 132 pacientes, ruidosa**: el pico de AUC interno (0,71) cayó
   a 0,577 en exterior. Dedicar el 15 % a parar quita datos de fit.
5. **Umbrales**: las políticas de especificidad ≥ 95 % producen sensibilidad
   ≈ 0–7 %. En la demo en directo eso equivale a decir siempre «no pCR».
6. La clínica (subtipo y volumen) contiene la señal que las imágenes no han
   sabido extraer: el multimodal llega a 0,72 y el subtipo por sí solo explica
   gran parte.
7. Los ensembles de semillas y la media de cortes ayudan poco pero de forma
   consistente. Las técnicas «inconcluyentes positivas» (Dropout2d, afines, EMA)
   nunca se combinaron.

## 4. Literatura

- **BreastDCEDL** ([Fridman et al., arXiv 2506.12190](https://arxiv.org/html/2506.12190v3)):
  ViT con las tres fases como RGB, probablemente preentrenado (no admisible aquí).
  Test **0,72** global; I-SPY2 0,78, I-SPY1 0,68, **Duke 0,54**. Sensibilidad 0,27 con
  especificidad 0,95. El volumen tumoral es la variable más importante en sus
  modelos con clínica. Un modelo grande preentrenado sigue sin funcionar en Duke.
- El subtipo molecular se puede predecir desde DCE-MRI: en la literatura, AUC
  ≈ 0,7–0,9 según la cohorte, y más bajas en validación externa
  ([multitarea ER/PR/HER2, arXiv 2601.07001](https://arxiv.org/abs/2601.07001v1)).
  Aprender varias tareas a la vez mejoró la predicción de pCR temprana en
  [arXiv 2502.00051](https://arxiv.org/pdf/2502.00051).
- Las CNN desde cero con pocos datos sobreajustan; ayudan la parametrización
  compartida y los objetivos que regularizan la representación
  ([MIMS, arXiv 1907.02413](https://arxiv.org/pdf/1907.02413)).

## 5. Hipótesis de PartyNet

**Resultado: rechazada** (§6). El modelo final usa el mismo tronco sin tareas
auxiliares y con la receta media.

> Si la representación por corte se obliga a predecir también **HR, HER2 y
> log-volumen tumoral** (objetivos de entrenamiento, nunca entradas), aprende
> rasgos de biología tumoral asociados a pCR en lugar de memorizar cortes o
> cohorte. La AUC exterior debería subir y la brecha de sobreajuste bajar.

Cumple el PDF: CNN 2D propia desde cero, sin preentrenados, un único logit en
inferencia, BCEWithLogitsLoss normal y ponderada (`pos_weight = N0/N1`). Las
variables clínicas solo aparecen como etiquetas auxiliares en train; la
aplicación solo recibe PNG.

### Qué se conserva y qué cambia

| Componente | Decisión | Motivo |
|---|---|---|
| Tronco | Igual que V007: 6 canales (fases + realces firmados), vistas 256/128/64 → 128, 4 bloques 24/48/96/160, avg+max | La mejor media CV; no se vuelve a buscar arquitectura |
| **Cabezas auxiliares** | HR, HER2 (BCE) y log-volumen (Smooth L1), λ = 1 | **Novedad** (§5) |
| Aumentos | Rot90 × flip + afín suave (±5 % escala, ±3 % traslación), idénticos en las tres fases | Los afines dieron +0,016 aislados |
| EMA de pesos | 0,995, con calentamiento | +0,006 aislado; suaviza el óptimo temprano |
| Dropout espacial | 0,10 | +0,022 aislado |
| Optimización | AdamW, LR 3·10⁻⁴, WD 0,01, lote 64, coseno de 40 épocas, warmup 1, bf16 | Receta modal de V007/27 |
| Parada | AUC de paciente en selección (15 %) con media móvil de 3 épocas, paciencia 8, mínimo 6 | En la prueba previa (fold 0, solo selección), la BCE ponderada era mínima en la época 3, con AUC de selección 0,49, mientras la AUC subía a 0,57 en la época 10: con `pos_weight` la BCE mide sobre todo la descalibración |
| Ensemble y TTA | 3 semillas por fold y 4 reflexiones en inferencia | Reduce varianza a coste cero de datos |
| Modelo final | **100 % de train**, épocas = mediana de las elegidas en CV, 5 semillas | Usa las 132 pacientes de selección |
| Umbral | Youden sobre las OOF calibradas de la CV | Sensibilidad útil en la demo; nunca test |

### Comparaciones que hace la CV

- **λ = 1 frente a λ = 0** con la misma pérdida, semillas, roles, inicialización,
  orden de lotes y aumentos. Es la prueba de la hipótesis.
- **Ponderada frente a normal**: requisito del PDF.
- Cada brazo frente a la **referencia de cohorte**. Además se informa de la AUC
  **dentro de cada cohorte**, para separar biología de cohorte, y por subtipo.
- Diagnóstico: AUC de HR/HER2 predichos desde la imagen en exterior. Si salen
  ≈ 0,5, las cabezas no aprendieron y la hipótesis no se ha probado.

### Límites conocidos

- El train ya se ha usado muchas veces: la CV es desarrollo adaptativo. El test
  público (0,608 para V007) ya está visto. La confirmación honesta es la
  validación privada del profesorado.
- La calibración y el umbral salen de las OOF de 3 redes por fold, entrenadas
  con ~746 pacientes. El paquete final tiene 5 redes con 1.097. Es monótono, así
  que la AUC no cambia, pero el punto operativo puede desplazarse.
- El subtipo en el mundo real no se conoce al leer la imagen. Aquí solo se usa
  para entrenar; en inferencia no hace falta.

## 6. Resultados de la rama (26–27/10/2026)

Cinco CV de 5 folds por paciente con los mismos roles y semillas (42, 2026, 7).
Por eso las comparaciones entre recetas son pareadas. AUC exterior por
paciente: media de cortes, de semillas y de 4 reflexiones (TTA). Todo es
desarrollo dentro de train; test y validación privada siguen cerrados.

| Receta y pérdida | Media AUC ± DE | Δ frente al control [IC95 %] | Época elegida | Pico selección | Brecha AUC elegida / final | Tendencia 1–14 |
|---|---:|---|---:|---:|---:|---:|
| Control normal | 0,5802 ± 0,025 | — | 5 | 3 | 0,099 / 0,263 | no medida |
| Control ponderada | 0,5814 ± 0,029 | — | 4 | 2 | 0,083 / 0,264 | no medida |
| Control + aux, normal | 0,5622 ± 0,058 | −0,018 [−0,042; +0,005] | 10 | — | — | — |
| Control + aux, ponderada | 0,5671 ± 0,038 | −0,014 [−0,036; +0,008] | 8 | — | — | — |
| Lenta normal | 0,5494 ± 0,021 | −0,031 [−0,060; −0,002] | 12 | 7 | 0,001 / 0,079 | 14/15 |
| Lenta ponderada | 0,5505 ± 0,038 | −0,031 [−0,060; +0,000] | 12 | 10 | 0,015 / 0,071 | 14/15 |
| **Media normal** | **0,5842 ± 0,036** | **+0,004 [−0,015; +0,022]** | **8** | 6 | 0,064 / 0,144 | 7/15 |
| Media ponderada | 0,5734 ± 0,027 | −0,008 [−0,025; +0,009] | 12 | 11 | 0,067 / 0,162 | 4/15 |
| Presupuesto lineal, normal | 0,5718 ± 0,013 | −0,008 | 12 (fija) | 7 | 0,083 | 10/15 (1–12) |
| **Presupuesto lineal, ponderada** | **0,5814 ± 0,018** | **0,000** | **12 (fija)** | 7 | **0,094** | 8/15 (1–12) |
| Presupuesto coseno, normal | 0,5718 ± 0,015 | −0,008 | 12 (fija) | 7 | 0,039 | 11/15 (1–12) |
| Presupuesto coseno, ponderada | 0,5802 ± 0,028 | −0,001 | 12 (fija) | 6 | 0,070 | 10/15 (1–12) |

En las recetas de presupuesto, la época elegida es la última del presupuesto,
así que la brecha elegida coincide con la final. Su Δ frente al control es
descriptivo (diferencia de medias del mismo brazo). La comparación pareada que
decide es con el mejor modelo anterior: **presupuesto lineal ponderada − media
normal = −0,003, IC95 % [−0,019; +0,014]**, prácticamente un empate. Frente a
la media del mismo brazo (ponderada): +0,008 [−0,011; +0,028].

Referencia solo cohorte: 0,549 ± 0,027. La media normal la supera en +0,035
[−0,002; +0,071], con P(Δ ≤ 0) = 0,03. Es lo más cerca que ha estado el
proyecto de demostrar señal por encima de la cohorte, aunque el IC todavía
toca el 0.

![Pérdida y AUC por época, control frente a receta media](img/perdidas_vs_control.png)

![Pérdida y AUC por época, control, media y presupuesto lineal](img/perdidas_tres_recetas.png)

### Qué se aprendió

1. **Multitarea (HR, HER2, volumen): rechazada.** La red aprende el volumen
   (ρ ≈ 0,7) pero no el subtipo (AUC HR ≈ 0,53, HER2 ≈ 0,49). El volumen se
   lleva capacidad y la AUC baja, sobre todo dentro de Duke.
2. **El pico temprano venía de cómo se define una época.** Con los ~10 cortes
   casi idénticos de cada paciente, la época 3 equivale a ~30 pasadas por
   paciente. El control memorizaba: la BCE de selección sube desde la época 3
   y la brecha final es 0,26.
3. **Regularizar fuerte estira las curvas, pero quita señal.** La receta lenta
   (LR 2e-4, WD 0,05, dropout 0,5, giro ±15°, cutout, label smoothing) cumple
   la tendencia 1–14 en 14/15 redes, pero su AUC cae al nivel de la cohorte:
   infraajuste.
4. **Basta con cambiar el muestreo.** La receta media conserva la
   regularización del control y solo ve 3 cortes al azar por paciente y época
   (más un giro de ±10° y EMA 0,98). Elige el doble de tarde (8 frente a 4),
   reduce la brecha final de 0,26 a 0,14 y discrimina igual o algo mejor. La
   BCE de selección deja de divergir, aunque sigue sin bajar: lo que aprende
   generaliza poco.
5. **Normal frente a ponderada:** sin diferencia relevante en ninguna receta
   (|Δ| ≤ 0,014, todos los IC incluyen 0). Cumple la comparación del PDF.
6. **Criterios fijados antes de la CV media:** pico entre 10 y 15, tendencia en
   ≥ 50 % de las redes y Δ ≥ −0,01. Ningún brazo de la receta media cumple los
   tres; se toman como orientativos.
7. **El calendario de LR era la pieza que faltaba.** Con la receta media, el
   coseno de 60 épocas seguía con el LR alto cuando la red empezaba a
   memorizar. [Budgeted Training (Li et al., ICLR 2020)](https://arxiv.org/pdf/1905.04753)
   recomienda ajustar el calendario al presupuesto y decaer hasta 0. En pilotos
   del fold 0 con 16 épocas, la BCE de selección tocaba mínimo en la época 11
   sin perder AUC. Con presupuesto de 12 épocas y la última época como
   checkpoint, desaparece la elección ruidosa de época con 131 pacientes.
8. **Resultado: época 12 con la misma calidad.** La ponderada con presupuesto
   lineal empata con la mejor AUC (0,581 frente a 0,584), reduce a la mitad la
   variación entre folds (DE 0,018; ningún fold < 0,558) y baja la brecha final
   de 0,144 a 0,094. Lineal y coseno quedan igual; se elige la lineal por ser
   la recomendada y la de menor dispersión. Con este calendario, la ponderada
   gana a la normal (+0,010 [−0,002; +0,022]).
9. **Descartado en los pilotos:** 2 cortes por paciente y época (AUC de
   selección ≈ 0,56). SWA no se probó: con pocos datos promedia soluciones ya
   sobreajustadas ([SWAD, Cha et al. 2021](https://arxiv.org/pdf/2102.08604)), y
   la EMA ya promedia pesos.

## 7. Modelos finales

### `modelos/party_final_v2/` — vigente (27/10/2026)

- **Receta de presupuesto, LR lineal hasta 0, pérdida ponderada**
  (`pos_weight` = N0/N1 de train), sin tareas auxiliares. 12 épocas fijas,
  5 semillas (42, 2026, 7, 1234, 31415), todas las 1.097 pacientes de train.
  Unos 38 s por red, con 7,3 GB de memoria máxima. En la época 12, la AUC de
  fit de cada red es 0,61–0,65.
- **Política, ajustada solo con las OOF de la CV:** Platt a = 0,666,
  b = −0,843. Umbral Youden 0,288 sobre la probabilidad calibrada: en OOF,
  sensibilidad 77,6 %, especificidad 36,3 %, F1 pCR 0,469 y macro-F1 0,484.
  El paquete guarda también el umbral alternativo de máximo macro-F1 (0,303):
  sensibilidad 46,6 %, especificidad 64,3 % y macro-F1 0,545. La regla
  prefijada es Youden; cambiarla es una decisión de uso (más o menos falsos
  pCR), no se toma mirando test.
- GPU y CPU coinciden con una diferencia de unas 2 cienmilésimas en la
  probabilidad.

### `modelos/party_final/` — anterior (26/10/2026), conservado

- **Receta media, pérdida normal, sin tareas auxiliares.** 8 épocas (mediana
  de las elegidas en la CV), 5 semillas (42, 2026, 7, 1234, 31415), todas las
  1.097 pacientes de train. 593.716 parámetros por red; unos 25 s por red en
  la RTX 3090, con 7,3 GB de memoria máxima.
- **Inferencia:** PNG/255, TTA con 4 reflexiones, media de las 5 redes; para
  cada paciente, media de sus cortes. GPU y CPU coinciden con una diferencia
  de unas 6 millonésimas.
- **Política, ajustada solo con las OOF de la CV:** Platt a = 0,849,
  b = −0,100. Umbral Youden 0,297 sobre la probabilidad calibrada. En OOF:
  sensibilidad 55,9 %, especificidad 55,4 %, F1 pCR 0,425 y macro-F1 0,531
  (frente a 0,414 de decir siempre «no pCR»).
- El paquete completo (5 pesos, `paquete.json` con SHA-256, historias y
  registro; ≈12 MB) está versionado en Git.
- **Test aún no evaluado.** `evaluar-test` se ejecuta una sola vez y escribe la
  marca en el paquete.

## 8. Imagen + datos clínicos: fusión tardía (27/10/2026)

Desde el 27/10 el profesor permite usar datos clínicos. El diseño y las
comparaciones se fijaron antes de entrenar en
[PROTOCOLO_FUSION.md](PROTOCOLO_FUSION.md) (commit `5670b2f`). La CNN
`party_final_v2` no cambia: su probabilidad entra como una variable más en una
regresión logística con edad, log(1 + volumen), HR y HER2. Con datos
incompletos se usa un submodelo por patrón de variables disponibles (Fletcher
Mercaldo y Blume, 2020), sin imputar.

**CV anidada.** La puntuación de imagen de las pacientes de ajuste de cada fold
sale de 60 redes internas (5 exteriores × 4 interiores × 3 semillas, ≈25 s cada
una) que nunca vieron el fold evaluado. Resultados en las 1.097 pacientes:

| Modelo | Media AUC ± DE | AP | Brier |
|---|---:|---:|---:|
| Solo imagen | 0,581 ± 0,018 | 0,340 | 0,207 |
| Solo subtipo (HR, HER2) | 0,679 ± 0,031 | 0,429 | 0,188 |
| Subtipo + imagen | 0,699 ± 0,031 | 0,467 | 0,188 |
| Clínica completa | 0,708 ± 0,033 | 0,496 | 0,184 |
| **Clínica completa + imagen** | **0,714 ± 0,033** | **0,506** | **0,183** |

| Comparación | Δ AUC | IC95 % pareado |
|---|---:|---|
| **Principal:** completa + imagen − completa | +0,006 | [−0,001; +0,013] |
| Subtipo + imagen − subtipo | +0,020 | [+0,008; +0,033] |
| Completa + imagen − imagen | +0,132 | [+0,085; +0,181] |
| Ponderada − normal (completa + imagen) | −0,001 | [−0,003; +0,000] |
| Exploratoria: interacción imagen × subtipo | −0,008 | [−0,016; −0,000] |
| Diagnóstico: ingenuo − anidado | −0,001 | [−0,006; +0,004] |

Qué se aprendió:

1. **Con toda la clínica, la imagen aporta poco y no significativo**
   (+0,006, P(Δ≤0) = 0,054). Su coeficiente estandarizado es positivo en los
   5 folds (+0,08 a +0,20), pero pequeño frente a HR (−0,64) o HER2 (+0,45).
2. **Con clínica parcial la imagen sí suma:** +0,020 sobre el subtipo y
   +0,02 sobre edad o volumen. Es el caso realista si en la evaluación privada
   faltan datos.
3. La imagen aporta información distinta de la clínica (no se correlaciona con
   volumen ni edad, ni predice HR/HER2), pero débil.
4. La pérdida ponderada no cambia la AUC de la logística. La interacción
   imagen × subtipo empeora y no se adopta.
5. El stacking ingenuo daba lo mismo que el anidado: con una logística de 5
   coeficientes, el optimismo era despreciable. El anidado queda como la cifra
   defendible.
6. **Bandas (terciles OOF):** con clínica + imagen, la pCR observada es del
   13 %, 27 % y 48 % en las bandas baja, intermedia y alta; con imagen sola,
   del 21 %, 33 % y 33 %.

### `modelos/party_fusion_v1/` — vigente para imagen + clínica

Solo `paquete.json` (≈28 KB): referencia a `party_final_v2` verificada por el
SHA-256 de su `paquete.json` y 15 submodelos logísticos (8 con imagen y 7 solo
clínicos) ajustados con las 1.097 pacientes. La imagen de cada paciente es su
OOF de `cv_presupuesto`. Umbral de cada submodelo: Youden sobre sus OOF
anidadas (completa + imagen: 0,258, sensibilidad 76,4 % y especificidad 56,9 %),
con el de máximo macro-F1 como alternativa (0,365: 53,4 % y 77,3 %). Sin
ningún dato clínico se usa el paquete de imagen tal cual.

## 9. Web

`app/` es una web Streamlit con cinco páginas: inicio, predicción, CNN de
imagen, imagen + clínica y metodología. Lee los paquetes congelados y los
resúmenes de `app/recursos/*.json`, que genera `python -m pcr_party
recursos-app` a partir de `runs/` (solo train) y se versionan. La predicción
muestra la probabilidad, la banda con la pCR observada en ella (hover), el
detalle por corte, el desacuerdo entre las 5 redes y el submodelo clínico
usado. Las pacientes de ejemplo son de train y se marcan como tales.

## 10. ¿Puede la imagen aportar más junto a la clínica? (27/10/2026)

### Análisis previo (solo OOF de train, `runs/analisis_aporte_imagen/`)

- La puntuación de la CNN es casi independiente de la clínica (la clínica
  explica el 0,6 % de su varianza; no predice HR ni HER2). Aporta poco
  (+0,005) porque es débil, no porque sea redundante.
- La señal se concentra en HER2+ (AUC 0,61) y triple negativo (0,58); en
  HR+/HER2− es casi nula (0,54).
- No mejora con otra agregación de cortes (todas 0,570–0,576), con la
  heterogeneidad entre cortes, localizando el tumor (los PNG ya están
  centrados en él) ni con rasgos cinéticos clásicos (AUC ≤ 0,53).
- Para sumar +0,03 sobre la clínica, la imagen sola tendría que rondar 0,63.

### Piloto: CNN entrenada sobre el offset clínico (`docs/PROTOCOLO_OFFSET.md`)

La red no recibe la clínica: aprende solo el residuo, con
logit = offset clínico fijo + f(imagen). Hay dos brazos: una cabeza, y tres
cabezas por subtipo con el sesgo común. El offset se calcula sin fugas y hay
30 redes (`runs/cv_offset/`).

| Modelo | AUC media ± DE | Δ frente a fusión v1 [IC95 %] | Δ frente a su clínica [IC95 %] |
|---|---|---|---|
| fusión v1 (referencia) | 0,714 ± 0,033 | — | +0,006 [−0,001; +0,013] |
| offset | 0,711 ± 0,032 | −0,002 [−0,009; +0,005] | +0,004 [−0,008; +0,015] |
| offset por subtipo | 0,708 ± 0,031 | −0,005 [−0,011; +0,001] | +0,001 [−0,008; +0,009] |

**Ningún brazo cumple el criterio fijado** (Δ ≥ +0,010 frente a la fusión e IC
frente a su clínica sin el 0). `party_fusion_v1` sigue vigente.

Qué se aprendió:

- Concentrar el gradiente en lo que la clínica no explica no hace la señal de
  imagen más fuerte. Con una sola cabeza, f(x) sola baja a 0,559: aprende un
  residuo más específico, pero no mayor.
- Por subtipo, el offset mejora algo dentro de HER2+, triple negativo y
  HR+/HER2− (0,658, 0,621 y 0,574 frente a 0,654, 0,616 y 0,565). La mejora
  dentro de cada subgrupo es pequeña y no aparece en el conjunto.
- Las cabezas por subtipo empeoran (−0,003 frente a una cabeza): con
  ≈300–450 pacientes por subtipo la cabeza extra solo añade varianza.
- La fila «subtipo desconocido» del `RESUMEN.md` tiene n = 3 y no es
  interpretable.
- Conclusión: con ≈1.100 pacientes y una CNN desde cero, el techo está en la
  señal de la imagen, no en cómo se combina con la clínica.
