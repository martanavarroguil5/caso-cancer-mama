# Búsqueda de un modelo exclusivamente de imagen — 25/10/2026

## Objetivo y reglas

Maximizar ROC-AUC de pCR por paciente, aspiración 0,70, sin ocultar sobreajuste.
Solo píxeles PRE/EARLY/LATE como predictores; sin clínica, identificadores,
cohorte ni metadatos de adquisición como entradas. CNN propia, pesos aleatorios,
sin transfer learning, respetando el caso docente existente. GPU RTX 3090 24 GB.

El test entregado permanece cerrado. Los pilotos utilizan fit/selección interna
del fold exterior 0; las decisiones de desarrollo se consideran exploratorias.
Los folds exteriores se evalúan después de fijar la receta. Ningún corte de una
paciente cruza particiones. El historial previo ya exploró estos datos: las OOF
futuras se describirán como validación de desarrollo, nunca validación clínica.

Se guardan curvas de AUC y BCE sin aumentos tanto en fit como selección, pérdidas
con aumentos aparte, checkpoints, configuración, semillas y predicciones. Parada
por pérdida interna con paciencia; AUC interna sirve para comparar pilotos. No
se escoge el pico de AUC exterior ni se invierten signos usando su etiqueta.
Antes de entrenamientos largos: prueba corta, pilotos y revisión de curvas.
Revisar progreso como máximo cada ~20 minutos durante ejecuciones prolongadas.

## Evidencia inicial

Los ensayos CNN anteriores logran aproximadamente 0,53–0,58 AUC. Hay 1.097
pacientes y 10.945 cortes de desarrollo. Los cortes están muy correlacionados.
Se inspeccionó una cuadrícula de las tres fases y ambas clases: la anatomía y
la escala cambian; no todas las lesiones ocupan el centro exacto del recorte.
Se probará reducción de capacidad, aprendizaje directo por paciente (MIL),
representación explícita del realce y resúmenes estadísticos de los píxeles.
Estos últimos siguen siendo imagen exclusivamente; se etiquetarán claramente.

## Fuentes consultadas

- [BreastDCEDL, repositorio original](https://github.com/naomifridman/BreastDCEDL): heterogeneidad de cohortes y tres fases; sus pesos no se utilizan.
- [BreastDCEDL, Scientific Data](https://www.nature.com/articles/s41597-026-06589-6): diferencias de adquisición; la normalización y el endpoint no permiten trasladar directamente cifras publicadas a estos PNG.
- [Ilse et al., 2018, Attention-based Deep Multiple Instance Learning](https://proceedings.mlr.press/v80/ilse18a.html): una etiqueta por bolsa/paciente, agregación invariante al orden.
- [Wu y He, 2018, Group Normalization](https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.php): alternativa a estadísticas de lote inestables con pocos pacientes.

## Registro

Pendiente: implementar y verificar el piloto paciente/MIL y las representaciones.

### Piloto features_v1

Semilla 42, fold exterior 0 sin evaluar. Época por pérdida interna: 45/59; AUC fit 0.7178, selección 0.6056; BCE selección 0.5992; 0.08 min, 15649 parámetros.

### Piloto features_linearish

Semilla 42, fold exterior 0 sin evaluar. Época por pérdida interna: 80/80; AUC fit 0.6062, selección 0.5892; BCE selección 0.6400; 0.11 min, 15649 parámetros.

### Piloto cnn_mil_v1

Semilla 42, fold exterior 0 sin evaluar. Época por pérdida interna: 5/15; AUC fit 0.5629, selección 0.5482; BCE selección 0.6068; 0.12 min, 118346 parámetros.

### Piloto hybrid_v1

Semilla 42, fold exterior 0 sin evaluar. Época por pérdida interna: 44/45; AUC fit 0.6474, selección 0.6213; BCE selección 0.5963; 0.36 min, 133930 parámetros.

### Piloto hybrid_weighted_v1

Semilla 42, fold exterior 0 sin evaluar. Época por pérdida interna: 43/45; AUC fit 0.6267, selección 0.6064; BCE selección 0.6890; 0.37 min, 133930 parámetros.

### Piloto hybrid_fast_v2

Semilla 42, fold exterior 0 sin evaluar. Época por pérdida interna: 17/35; AUC fit 0.6882, selección 0.6035; BCE selección 0.5927; 0.29 min, 133930 parámetros.

### Piloto features_roi_v2

Semilla 42, fold exterior 0 sin evaluar. Época por pérdida interna: 31/49; AUC fit 0.7764, selección 0.6275; BCE selección 0.5882; 0.06 min, 30177 parámetros.

### Piloto hybrid_roi_v2

Semilla 42, fold exterior 0 sin evaluar. Época por pérdida interna: 23/41; AUC fit 0.7841, selección 0.6351; BCE selección 0.5849; 0.40 min, 148458 parámetros.

### Piloto hybrid_roi_bn_v2

Semilla 42, fold exterior 0 sin evaluar. Época por pérdida interna: 11/29; AUC fit 0.6367, selección 0.6180; BCE selección 0.5917; 0.26 min, 260034 parámetros.

### Piloto cnn_roi_bn_v2

Semilla 42, fold exterior 0 sin evaluar. Época por pérdida interna: 18/36; AUC fit 0.5543, selección 0.5763; BCE selección 0.6009; 0.33 min, 229922 parámetros.

### Piloto hybrid_rank_v2

Semilla 42, fold exterior 0 sin evaluar. Época por pérdida interna: 16/34; AUC fit 0.7174, selección 0.6576; BCE selección 0.5903; 0.40 min, 148458 parámetros.

### Piloto linear_roi_v3

Semilla 42, fold exterior 0 sin evaluar. Época por AUC interna con brecha máxima 0,10: 24/48; AUC fit 0.5991, selección 0.6442; BCE selección 0.5926; 0.06 min, 909 parámetros.

### Piloto hybrid_linear_v3

Semilla 42, fold exterior 0 sin evaluar. Época por AUC interna con brecha máxima 0,10: 30/54; AUC fit 0.5886, selección 0.6458; BCE selección 0.5955; 0.45 min, 118801 parámetros.

### Piloto hybrid_roi_linear_v3

Semilla 42, fold exterior 0 sin evaluar. Época por AUC interna con brecha máxima 0,10: 14/38; AUC fit 0.5950, selección 0.6469; BCE selección 0.5904; 0.38 min, 119255 parámetros.

### Piloto hybrid_texture_v3

Semilla 42, fold exterior 0 sin evaluar. Época por AUC interna con brecha máxima 0,10: 18/42; AUC fit 0.5908, selección 0.6379; BCE selección 0.5806; 0.42 min, 119831 parámetros.

### Piloto linear_texture_v3

Semilla 42, fold exterior 0 sin evaluar. Época por AUC interna con brecha máxima 0,10: 50/74; AUC fit 0.5965, selección 0.6620; BCE selección 0.5821; 0.08 min, 1485 parámetros.

### Piloto cnn_focus_v3

Semilla 42, fold exterior 0 sin evaluar. Época por AUC interna con brecha máxima 0,10: 34/58; AUC fit 0.6260, selección 0.6273; BCE selección 0.6106; 0.52 min, 229922 parámetros.

### Piloto linear_stable_v4, contexto 0

Promedio exponencial de pesos; época interna 22/62; AUC fit 0.6969, selección 0.6449; BCE selección 0.5935; 0.07 min.

### Piloto linear_stable_v4, contexto 3

Promedio exponencial de pesos; época interna 1/41; AUC fit 0.5876, selección 0.4611; BCE selección 0.5993; 0.04 min.

## Revisión de estabilidad antes de evaluación exterior

La confirmación interna de tres semillas en dos contextos da medias AUC de
0,6598 (red lineal sobre textura), 0,6547 (ensemble textura + CNN con ranking),
0,6226 (CNN con ranking) y 0,6062 (híbrido inicial). En ningún caso se han leído
predicciones exteriores. No se interpretan como prueba independiente.

Se detecta una debilidad en la red lineal v3: LR alto respecto de coeficientes
muy penalizados produce oscilaciones; seleccionar el máximo de AUC entre muchas
épocas puede sobreajustar la propia validación, aunque fit tenga AUC baja. Un
control de regresión logística convergida no reproduce toda esa ventaja. Se
inicia una tanda de estabilidad con LR menor, promedio exponencial de pesos,
capacidad reducida y brecha máxima 0,10. Es una corrección metodológica; no se
promociona v3 por tener el mayor pico interno. Las curvas y resultados anteriores
se conservan para hacer visible este problema.

La inferencia CPU y la evaluación GPU de nueve ejemplos internos difieren menos
de 0,00010 en probabilidad; los descriptores difieren como máximo 0,00021 por
redondeo de las operaciones de recorte e interpolación. No hay tripletas
idénticas de 128 px entre pacientes distintas en train (auditoría de 10.945).

### Piloto hybrid_linear_stable_v4, contexto 0

Promedio exponencial de pesos; época interna 15/43; AUC fit 0.7068, selección 0.6374; BCE selección 0.5942; 0.43 min.

### Piloto hybrid_linear_stable_v4, contexto 3

Promedio exponencial de pesos; época interna 3/35; AUC fit 0.6740, selección 0.5856; BCE selección 0.5924; 0.35 min.

### Piloto hybrid_mlp_stable_v4, contexto 0

Promedio exponencial de pesos; época interna 40/68; AUC fit 0.6954, selección 0.6147; BCE selección 0.6009; 0.67 min.

### Piloto hybrid_mlp_stable_v4, contexto 3

Promedio exponencial de pesos; época interna 23/51; AUC fit 0.6762, selección 0.6013; BCE selección 0.5937; 0.51 min.

### Piloto cnn_axial_v5, contexto 0

CNN 2D y Conv1d axial, época interna 14/36; AUC fit 0.5487, selección 0.5924; BCE selección 0.6043; 0.44 min.

### Piloto cnn_axial_v5, contexto 3

CNN 2D y Conv1d axial, época interna 6/28; AUC fit 0.5750, selección 0.5841; BCE selección 0.6026; 0.34 min.

### Piloto cnn_axial_weighted_v5, contexto 0

CNN 2D y Conv1d axial, época interna 2/25; AUC fit 0.5293, selección 0.5520; BCE selección 0.6885; 0.30 min.

### Piloto cnn_axial_weighted_v5, contexto 3

CNN 2D y Conv1d axial, época interna 34/56; AUC fit 0.5842, selección 0.5678; BCE selección 0.7009; 0.67 min.

### Piloto hybrid_axial_v5, contexto 0

CNN 2D y Conv1d axial, época interna 11/33; AUC fit 0.6314, selección 0.6420; BCE selección 0.6054; 0.40 min.

### Piloto hybrid_axial_v5, contexto 3

CNN 2D y Conv1d axial, época interna 18/40; AUC fit 0.6540, selección 0.5749; BCE selección 0.5928; 0.48 min.

## Protocolo final anidado, fijado antes de outer

Se conservan tres recetas CNN propias: `hybrid_v1` (BCE normal, resúmenes
simples), `hybrid_rank_v2` (CNN, ROI y ranking auxiliar, checkpoint por BCE) y
`hybrid_mlp_stable_v4` (CNN y MLP de ocho unidades con textura, pesos EMA).
Cada fold compara sus tres ensembles internos de semillas 42/2026/3407 y
selecciona por AUC interna, exigiendo brecha fit−selección <= 0,12. En empate,
menor BCE. Las redes lineales oscilatorias v3 quedan fuera del conjunto final.

Las cinco elecciones se cierran y guardan **antes de calcular cualquier outer**.
Se ajusta Platt sobre logits del ensemble de selección interna, con C=1, escala
interna y pendiente no negativa. No cambia el ranking de cada fold; permite
comparar escalas entre folds. Se informarán AUC cruda, calibrada y media por
fold para evitar atribuir a la red una mejora de mera alineación de escalas.

El candidato axial usa backbone CNN 2D y Conv1d entre cortes; conserva el
orden axial sin introducir convoluciones 3D ni metadatos numéricos de posición.
Los pilotos normal/ponderado no justifican ampliarlo. Se mantienen sus curvas.

La validación final sigue siendo desarrollo adaptativo, dado todo el historial
del proyecto. El test entregado y cualquier validación privada permanecen
cerrados. No se convertirá un resultado insuficiente en éxito cambiando el
objetivo después de ver outer.

## Primer resultado exterior: insuficiente

La búsqueda anidada de tres CNN alcanza AUC OOF calibrada 0,5638, cruda 0,5245,
y media por fold 0,5673 ± 0,0352. No mejora convincentemente la referencia
histórica de imagen. No se considera un buen modelo ni se declara logrado el
objetivo. El refit a igual número de pasos quedó prefijado antes de consultar
outer (ver su plan y fecha); termina sin modificar parámetros por este resultado.

Se continúa con una hipótesis diferente: textura a resolución original y redes
más pequeñas, evitando que la selección dependa de máximos entre muchas épocas.
La evaluación posterior se etiquetará como desarrollo adaptativo, porque ya se
ha visto este primer resultado exterior. El test sigue cerrado.

### CNN espacial spatial128_normal, contexto 0

Época 29/54, AUC fit 0.6507, selección 0.5949; BCE 0.5943. Solo selección interna.

### CNN espacial spatial128_normal, contexto 3

Época 29/54, AUC fit 0.6614, selección 0.6113; BCE 0.5900. Solo selección interna.

### CNN espacial spatial256_normal, contexto 0

Época 19/44, AUC fit 0.6104, selección 0.5541; BCE 0.6040. Solo selección interna.

### CNN espacial spatial256_normal, contexto 3

Época 30/55, AUC fit 0.7069, selección 0.5710; BCE 0.5964. Solo selección interna.

### CNN espacial spatial256_weighted, contexto 0

Época 21/46, AUC fit 0.6385, selección 0.5446; BCE 0.7256. Solo selección interna.

### CNN espacial spatial256_weighted, contexto 3

Época 28/53, AUC fit 0.7127, selección 0.5733; BCE 0.6843. Solo selección interna.

La CNN 2D que conserva una cuadrícula espacial 4×4 alcanza AUC interna 0,595/0,611
en los dos contextos. Las dos vistas nativas (256 global y centro 128) y BCE
ponderada no mejoran esos pilotos. No se amplían a cinco folds por esa evidencia.
El refit prefijado de la primera búsqueda tuvo media exterior 0,5630 y AUC
agrupada calibrada 0,5181; tampoco se adopta. Se guardan todas las pérdidas.
