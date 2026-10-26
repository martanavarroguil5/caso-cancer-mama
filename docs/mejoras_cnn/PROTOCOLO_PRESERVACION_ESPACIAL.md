# Preservación espacial y atención dentro de cada corte — 26/10/2026

Autorización: continuar investigación y modificar arquitectura/entrenamiento para
intentar mejorar el modelo. Se mantienen el enunciado, test/privada cerrados,
versiones históricas inmutables, solo main y prohibición de push sin nueva orden.

## Fundamento y novedad

La CNN V007 genera tres vistas de 128 px y las reduce a mapas de 4×4 antes del
pooling global. Esto motiva probar conservación de detalle, pero no demuestra
que esa reducción sea la causa de su rendimiento. El ensayo espacial28 usó una
vista128 y cuadrícula4×4; este ensayo procesa todos los píxeles originales256 y
conserva16×16. Reducir capacidad por sí solo ya resultó insuficiente; no se
presenta esta prueba como repetición de esa hipótesis aislada.

- [Zhang, ICML2019](https://proceedings.mlr.press/v97/zhang19a.html): filtrar antes
  de submuestrear puede mejorar estabilidad frente a desplazamientos. Benchmarks
  de imagen natural, no pCR. Adaptamos una reducción con máximo denso y filtro
  binomial fijo; no importamos arquitecturas ni pesos del artículo.
- [Woo et al., ECCV2018](https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.php): motiva
  ponderar características espacialmente. No implementamos CBAM ni utilizamos su
  backbone o modelos. Aquí la puerta espacial es propia y está acotada.
- [Estudio empírico de pooling en imagen médica](https://pmc.ncbi.nlm.nih.gov/articles/PMC8804673/):
  resultados dependientes de tarea y tamaño de lesión en otros datos; no una
  garantía para las RM de este caso.

Se han revisado los descartes de LR/WD, capacidad/dropout, GroupNorm, SAM, FiLM,
BCE por paciente, mixup, EMA, afines, brillo compartido, selección por pérdida,
CNN espacial128 y MixStyle. No se acumulan estos cambios sin control ni se
trasladan cifras internas de las redes MIL como si cumplieran la media por corte.

## Contrato de entrada y arquitectura propia

CNN 2D completamente desde cero; sin ResNet, EfficientNet, VGG, DenseNet,
preentrenados, transfer learning, nuevas modalidades, clínica ni etiquetas privadas.
PRE/EARLY/LATE PNG gris256×256 uint8, orden fijo y división entre255. Las fases
no son RGB. Solo geometría compartida: hflip0,5 y rot90 del control. Dentro de la
red se derivan EARLY−PRE, LATE−PRE y EARLY−LATE; no se cambian los PNG.

| Etapa | Forma por corte |
|---|---|
| Entrada /255 | 3×256×256 |
| Fases y diferencias firmadas | 6×256×256 |
| Bloque 1 | 16×128×128 |
| Bloque 2 | 32×64×64 |
| Bloque 3 | 64×32×32 |
| Bloque 4 | 96×16×16 |
| Pooling espacial | 96 |
| Cabeza | 96→32→1 logit |

Cada bloque tiene dos Conv3×3 stride1 sin bias, BN y ReLU. La reducción combina
máximo2×2 stride1, filtro separable binomial[1,2,1]²/16 fijo y stride2 con un píxel de relleno de ceros. Se reconoce que los bordes se atenúan por ese padding.
Dropout2d0,1 tras el encoder; dropout0,35 en la cabeza antes de ambas lineales.
Sin máscaras, selección aprendida de cortes, atención entre cortes ni 3D.

Dos recetas prefijadas, con todos los demás parámetros iguales:

1. **uniforme**: puerta1×1 de96 pesos congelados en0, que produce pesos uniformes.
   214.497 parámetros entrenables, 214.593 totales.
2. **atencion**: esa puerta1×1 aprende desde0. Sus pesos espaciales son
   `0.5/256 + 0.5*softmax(score/2)`; suman1 y nunca concentran más de
   `0.5 + 0.5/256` en una posición. 214.593 parámetros entrenables.

Las dos redes tienen exactamente la misma inicialización de todos sus tensores
para una semilla. La atención pondera posiciones dentro de un corte; la predicción
por paciente sigue siendo media de probabilidades de todos sus cortes. El mapa
aprendido no es una segmentación tumoral ni una explicación clínica validada.

## Cuarenta entrenamientos y congelación

Dos recetas × cinco folds × BCE normal/ponderada × semillas42/2026 =40 runs.
Roles auditados del ensayo27: fit745–746, selección132 y exterior219–220 pacientes.
Los roles son idénticos entre recetas; no se mezclan cortes de una paciente.
Control cerrado27 sin reentrenar: mediaAUC0,556270. Cambia la arquitectura completa
frente a ese control: no atribuir causalmente todo el efecto a antialiasing.
La diferencia atención−uniforme sí ofrece una ablación más específica, secundaria.

BCEWithLogitsLoss por corte, contribución1/n_cortes por paciente y normalizador
fijo; pos_weight=N0/N1 con pacientes de fit. AdamW, LR0,0003, WD0,01, batch32,
coseno40, min_lr1e-5, paciencia8, min_deltaBCE0,001, observar al menos14 épocas.
Un checkpoint puede elegirse antes de14. No fabricar curvas ni forzar un óptimo.

Checkpoint principal por mínima BCE por paciente en selección (misma ponderación
que fit); BCE normal o ponderada elegida por AUC de la pareja de dos semillas
en selección, empate normal. Se conserva el mejor AUC de la misma trayectoria
como diagnóstico secundario, sin cambiar a él después de mirar exterior.

**Las diez elecciones de las dos recetas deben quedar congeladas antes de la
primera inferencia exterior de esta familia.** No modificar código, config o
particiones durante una matriz sana. Fallo real que cambie receta exige nueva
salida y no mezclar resultados. Todos los checkpoints se preservan localmente.

## Análisis prefijado

Métrica principal: media de AUC cruda entre cinco folds de cada receta menos la
del control27, sobre las mismas1.097 pacientes. Dos hipótesis: Bonferroni2,
IC97,5% de diferencias pareadas. Bootstrap4000 por paciente estratificado
fold/clase, semilla20261026. Intervalos condicionados a estos pesos: no corrigen
búsqueda histórica, variabilidad completa de entrenamiento ni dependencia entre
modelos de CV que comparten fit.

Secundarios: atención−uniforme pareado, cohorte entrenada solo en fit, AUC agrupada
cruda y calibrada por parejas, BCE fijas, checkpoint AUC, AP, Brier/log-loss,
cohortes, sensibilidad/especificidad, precisión y F1 por clase, macroF1, FP/FN,
comparador siempre no pCR y probabilidades constantes de prevalencia de fit.
No reinterpretar métricas de selección como AUC exterior ni seleccionar el
mejor fold. Comparaciones con V007 histórica son descriptivas por otro protocolo.

Platt no negativo C1 y umbrales95% especificidad / máximo macroF1 elegidos en
selección de la misma pareja antes de inferir exterior. No ajustar umbrales con
exterior ni transferirlos como validados a un ensemble final de diez redes.

Gate para motivar una réplica: mediaAUC≥0,60, AUC agrupada calibrada por
parejas≥0,60, delta≥0,01 frentecontrol27, límite inferiorIC97,5% del delta>0 y
mejora en al menos3/5 folds. Pasarlo no declara éxito final ni habilita test.
Si se cumple, se registra una réplica con nuevas semillas y roles internos,
control reentrenado y sin variar las recetas. Si no se cumple, se cierra esta
familia y se documenta el límite; no se hacen variantes contra el mismo exterior.

## Verificación y conservación

Antes de pacientes: tests de dimensiones nativas/mapa16, mapas temporales,
inferencia independiente por corte, pesos de atención normalizados/acotados,
inicialización común, filtro fijo, gradientes, métricas por paciente, separación
de roles y reanudación exacta. Prueba GPU solo sintética:40 pasos por receta,
pérdida normal/ponderada, AdamW/AMP y descenso de pérdida; no cuenta como runreal.

Durante entrenamiento: pérdidas/pesos finitos, pasos reales AdamW/AMP, progreso,
GPU, brechas y parada temprana. Curvas fit/selección en eval, sin aumentos/dropout,
media de todos los cortes y misma BCE por paciente. Tendencia conjunta1–14 es
solo diagnóstico, no prueba de generalización ni umbral de aceptación.

Después: revisar inferencia congelada, cálculos independientes, hashes de fuentes,
roles, metadatos y versiones anteriores; curvas, OOF e informe en español. Conservar
pesos elegidos como candidatos solo si merecen réplica, sin activar aplicaciones.
No datasets, optimizer/RNG ni checkpoints de trabajo en Git. Ensayos descartados
con informe y código en historia de main, retirado del árbol activo. No push.

Todo sigue siendo desarrollo adaptativo: las pacientes de train se han observado
en el proyecto y el test ya tiene antecedentes históricos. No se ha recuperado
un test ciego cambiando nombres o particiones. Test y privada cerrados en esta
familia. No hay promesa de alcanzar0,60.

```bash
.venv/bin/python -B -m cancer_mama.preservacion_espacial comprobar
.venv/bin/python -B -m cancer_mama.preservacion_espacial preparar
.venv/bin/python -u -B -m cancer_mama.preservacion_espacial entrenar
```
