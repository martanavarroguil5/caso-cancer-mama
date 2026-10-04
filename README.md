# Caso cáncer de mama

Predicción educativa de respuesta patológica completa (pCR) a partir de resonancia
DCE anterior al tratamiento. Se conserva la base de Marta Navarro Guil,
commit `69a44eb` (28/09/2026), y se añade únicamente el paso de entrenamiento.
El proyecto queda en `main`, con los archivos numerados y resultados por etapa.

## Preparación

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python descargar_datos.py
```

En Windows, utiliza `.venv\Scripts\python.exe`. Los datos se guardan en
`breastdcedl/` y los pesos quedan fuera de Git. No hace falta volver a descargarlos
si ya están en el ordenador. Para usar el modelo final en otro ordenador, copia
también `resultados/04_entrenamiento/modelos/`: los diez archivos `.pt` no se
suben a Git y deben conservarse junto a su manifiesto.

## Pasos

```bash
.venv/bin/python 01_auditoria_datos.py
.venv/bin/python 02_eda_profesional.py
.venv/bin/python 03_preparar_datos.py --fold 0
.venv/bin/python 04_entrenamiento.py verificar
```

Los pasos 01-03 y `pipeline_datos.py` son los originales de Marta. El paso 04
reutiliza su Dataset: convierte PRE/EARLY/LATE a float32 /255, conserva los cinco
folds originales y aplica las transformaciones conjuntamente. Usa la configuración
sin estandarización; las estadísticas del paso 03 siguen siendo descriptivas.
Los CSV fuente no se modifican y el entrenamiento no abre imágenes ni convierte
etiquetas del test reservado.

## Red y entrenamiento

`04_entrenamiento.py` contiene toda la CNN, el aprendizaje, la comparación OOF y
la predicción. La entrada es `[N,3,256,256]`. La red aplica `2*x-1` y cuatro bloques
de dos convoluciones Conv-BN-ReLU con 24/48/96/160 canales. Combina pooling global
promedio y máximo y una cabeza 320→64→1 con dropout 0,20. Son 551.913 parámetros;
la salida es un logit. Se inicializa desde cero con Kaiming, sin pesos preentrenados.

Se utiliza AdamW (LR 0,0008, weight decay 0,0001), descenso coseno hasta 0,00001,
lotes de 64, clipping de gradiente a 5 y AMP FP16 en CUDA. La validación se hace
en float32. Máximo 46 épocas, mínimo 12, paciencia 10 y min_delta 0,001. Se guarda
el checkpoint de mayor AUC por paciente; min_delta controla la paciencia.

Los aumentos son espejo horizontal con p=0,5 y rotaciones exactas de
0/90/180/270°, iguales en los tres canales. Se comparan BCE normal y ponderada
N0/N1, contando cortes solamente en el subconjunto que aprende cada fold.

Para probar un lote sin entrenar el experimento completo:

```bash
.venv/bin/python 04_entrenamiento.py entrenar --prueba --folds 0 --semillas 42 --perdidas ponderada --dispositivo cpu --workers 0
```

Para una ejecución concreta en la RTX 3090:

```bash
.venv/bin/python 04_entrenamiento.py entrenar --folds 0 --semillas 42 --perdidas ponderada --dispositivo cuda
```

Para reproducir las 40 ejecuciones del diseño del informe (dos configuraciones,
dos pérdidas, dos semillas y cinco folds):

```bash
.venv/bin/python 04_entrenamiento.py entrenar --configuraciones base_raw raw_rot90 --dispositivo cuda
.venv/bin/python 04_entrenamiento.py comparar
```

`base_raw` conserva solo el espejo horizontal; `raw_rot90` añade las rotaciones.
Las dos ejecuciones exploratorias anteriores con diferencias de realce quedan
registradas en el respaldo del experimento original, fuera de este proyecto.

Los nuevos resultados van a `resultados/04_entrenamiento/ejecuciones/`,
separados del modelo final histórico. Cada ejecución guarda configuración,
entorno, curvas, predicciones OOF, `best.pt`, `last.pt` e `inference.pt`.
Volver a lanzar el mismo comando reanuda desde la última época completa;
un cambio de datos, código o configuración exige usar otra `--salida`.
La reanudación conserva optimizador, calendario, scaler y RNG. La identidad
numérica se comprueba en CPU con una interrupción real; depende del mismo entorno
y hardware y no se garantiza entre plataformas.

La comparación exige dos semillas y cinco folds completos por pérdida. Compara
media/máximo/mediana por paciente, ajusta Platt con OOF y elige umbral mediante
Youden. Guarda `comparacion/modelo_desarrollo.json`, sin sustituir el modelo final.
Elegir candidatos y checkpoints con OOF introduce optimismo: no es validación
anidada. Un entrenamiento nuevo genera resultados nuevos.

## Modelo final del informe

`resultados/04_entrenamiento/modelo_final.json` referencia los diez pesos
originales de `modelos/`: cinco folds × semillas 42 y 2026, raw_rot90 y BCE
ponderada. Sus bytes y SHA-256 se conservan. Primero se promedian las probabilidades
sigmoid de los modelos y después las de los cortes de la paciente.

Calibración congelada: `sigmoid(0.6680108289790444 * logit(p) - 0.815361445727788)`,
con p limitada a `[1e-7,1-1e-7]`. Umbral congelado: `0.2982406880601241`.
La calibración de este modelo histórico se conserva; las ejecuciones nuevas
calculan sus propios parámetros con OOF.

Las métricas originales están en `historico/test_metrics.json`: AUC 0,546
(IC95% 0,459-0,635), sensibilidad 0,453, especificidad 0,593 y accuracy 0,551,
en 176 pacientes. El test ya se evaluó el 30/09/2026. El paso 04 no tiene una
acción para repetirlo. Los resultados indican generalización limitada y no
acreditan utilidad clínica.

Para predecir un corte nuevo con los pesos conservados:

```bash
.venv/bin/python 04_entrenamiento.py predecir --pre paciente_z000_PRE.png --early paciente_z000_EARLY.png --late paciente_z000_LATE.png
```

Se pueden pasar varios archivos en cada opción, en el mismo orden y de una sola
paciente. Deben ser PNG monocromos de 8 bits y 256×256. Para usar pesos nuevos,
indica `--manifest ruta/comparacion/modelo_desarrollo.json`.

## Evidencia y comprobaciones

`resultados/04_entrenamiento/historico/` reúne el PDF original, selección,
comparación de pérdidas, OOF, curvas de los diez modelos y resultados del test.
El manifiesto original se conserva sin editar, con sus rutas y checksum antiguos;
`procedencia.json` registra el traslado. El manifiesto operativo usa rutas relativas.

El PDF describe la ejecución del 30/09/2026, incluidos su web y nombres de archivos
anteriores. La web y sus exportaciones se retiraron en esta limpieza.
La correspondencia de código es:

| Archivo anterior | Ubicación actual |
|---|---|
| cnn.py | CNN y bloques en 04_entrenamiento.py |
| datos.py | pipeline_datos.py y validaciones del paso 04 |
| entrenar.py | entrenamiento y checkpoints en el paso 04 |
| evaluar.py | comparación OOF, calibración e inferencia en el paso 04 |
| configs/seleccion_final.json | modelo_final.json y historico/seleccion_original.json |

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m unittest discover -s tests -v
.venv/bin/python -B 04_entrenamiento.py verificar
```

Las pruebas verifican la arquitectura, escala y alineación de fases, separación
por paciente, OOF y reanudación idéntica, sin repetir el test reservado.
El respaldo de la limpieza, con el código, ramas y experimentos previos,
está fuera de la carpeta del proyecto. Para compartir código entre ordenadores,
usa `git pull` antes de trabajar y revisa `git status` antes de commit/push.

## Ajustes indicados por el profesor

La acción `ajustar` entrena y revisa en bloques de diez épocas, reanudando pesos,
optimizador y calendario; no comienza de cero en cada bloque. Compara seis
configuraciones con el mismo fold, semilla, pérdida ponderada, lote y presupuesto.

| Candidato | Pooling | LR | Dropout | Weight decay |
|---|---|---:|---:|---:|
| referencia | Original | 0,0008 | 0,20 | 0,0001 |
| pool_intermedio | Entre bloques | 0,0008 | 0,20 | 0,0001 |
| pool_lr_baja | Entre bloques | 0,0003 | 0,20 | 0,0001 |
| pool_lr_dropout | Entre bloques | 0,0003 | 0,35 | 0,0001 |
| pool_lr_dropout_wd | Entre bloques | 0,0003 | 0,35 | 0,001 |
| pool_dropout_wd | Entre bloques | 0,0008 | 0,35 | 0,001 |

La variante nueva añade MaxPool 2×2 después de cada bloque. En los bloques 2-4
utiliza convoluciones con stride 1 y pooling para reducir la resolución. El primer
bloque conserva su stride 2 y pooling. Así mantiene las salidas 64/32/16/8 y
551.913 parámetros. Se entrena desde cero; los diez modelos históricos continúan
utilizando su arquitectura original.

```bash
.venv/bin/python 04_entrenamiento.py ajustar --folds 0 --semillas 42 --epocas 30 --revision-cada 10 --lote 16 --dispositivo cuda
```

El lote 16 limita memoria cuando la GPU se comparte. Todos los candidatos usan
el mismo tamaño para que la comparación sea pareada. En GPU exclusiva se puede
indicar `--lote 64` y una `--salida` nueva para conservar ambos experimentos.

Cada revisión (épocas 10, 20 y 30) guarda ranking JSON, CSV comparativo, curvas,
checkpoint de esa época y mejores pesos hasta ese bloque. Las curvas muestran
pérdida train/validación, AUC, F1 y accuracy. Las métricas train se calculan sin
aumentos y con dropout desactivado al revisar; las de validación se calculan cada
época. La pérdida durante aprendizaje se distingue de la pérdida train en
modo de evaluación.

F1 corresponde a la clase pCR=1 y se calcula por paciente, con umbral 0,5, junto
con precisión, sensibilidad y accuracy equilibrada. La AUC es el criterio principal
y F1 desempata. La comparación con un fold es exploratoria: la mejor configuración
debe confirmarse en otros folds antes de afirmar una mejora general. El test
reservado ya observado no participa en estos ajustes.

Los resultados quedan en `resultados/04_entrenamiento/ajustes/`. Una caché uint8
exclusiva de train evita decodificar repetidamente los mismos PNG. Tanto la caché
como los nuevos experimentos quedan fuera de Git. Para reanudar el ajuste, vuelve
a lanzar exactamente el mismo comando.

Para lanzar un entrenamiento concreto con pooling intermedio y revisar cada diez
épocas:

```bash
.venv/bin/python 04_entrenamiento.py entrenar --pooling intermedio --folds 0 --semillas 42 --perdidas ponderada --revision-cada 10 --dispositivo cuda
```

Fuentes de implementación: [MaxPool2d de PyTorch](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MaxPool2d.html)
y [definición de F1 en scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).

## Comparación controlada del tamaño de la CNN

`comparar-tamano` compara la CNN con pooling ajustado y una variante más pequeña.
Los canales son 24/48/96/160 frente a 16/32/64/128: 551.913 frente a 310.513
parámetros (un 43,7 % menos). El resto del diseño y del
entrenamiento es idéntico: pooling entre bloques, cabeza de 64 neuronas,
dropout 0,35, LR 0,0008, weight decay 0,001, BCE ponderada y rotaciones.
La red histórica del PDF y sus diez pesos se conservan.

Se entrenan las dos variantes en los cinco folds originales y las semillas 42 y
2026: veinte ejecuciones de treinta épocas con el mismo presupuesto. Las
revisiones de train sin aumentos se hacen en 10/20/30 y la validación cada época.
Se guarda el checkpoint de mayor AUC por paciente hasta ese presupuesto.

```bash
.venv/bin/python -B 04_entrenamiento.py comparar-tamano --epocas 30 --revision-cada 10 --lote 64 --workers 2 --paralelos 2 --dispositivo cuda
```

Este ensayo utiliza lote 64 en ambas variantes con la GPU disponible. Es una
comparación nueva y no mezcla las métricas anteriores obtenidas con lote 16.
Dos procesos independientes aprovechan la RTX 3090; cada uno conserva su RNG,
checkpoints y registro. Se puede indicar `--paralelos 1` para ejecutar en serie.
Volver a lanzar el mismo comando reanuda las ejecuciones pendientes.

Para comprobar ambos modelos con un lote real antes de entrenar:

```bash
.venv/bin/python -B 04_entrenamiento.py comparar-tamano --prueba --workers 2 --paralelos 2 --dispositivo cuda
```

Los resultados se guardan en `resultados/04_entrenamiento/comparacion_tamano/`.
El protocolo, código por SHA-256 y caché de train quedan identificados antes de
entrenar. Cada run guarda configuración, entorno, pesos, curvas y predicciones OOF.

La evaluación exige veinte runs completos con sus checksums y predicciones
coincidentes con las pacientes y folds de train. Informa media y dispersión de
las diez ejecuciones por variante, diferencias pareadas por fold/semilla y OOF
por paciente tras promediar las dos semillas de validación y sus cortes. El
umbral permanece en 0,5: no se ajusta para favorecer una variante.

`resultado.json` incluye AUC, average precision, F1, sensibilidad, especificidad,
precisión y Brier. `metricas_cohortes.csv` desglosa las métricas por cohorte.
Los intervalos de diferencias se
calculan mediante 2.000 remuestreos pareados y estratificados de pacientes.
Están condicionados a esas predicciones OOF; no recogen toda la variación del
entrenamiento. La selección de checkpoints usa validación y no es validación
anidada ni externa. El test reservado no se abre.

Archivos principales: `comparacion_runs.csv`, `diferencias_pareadas.csv`,
`metricas_cohortes.csv`, `oof_pacientes.csv`, `curvas_comparacion.png`,
`roc_precision_recall.png` y `resultado.json`. Estos resultados y la caché se
mantienen fuera de Git, dentro de una sola carpeta de la etapa 04.

## Resultados de los ajustes del 03/10/2026

Se ejecutaron las seis combinaciones a 30 épocas, revisando en 10/20/30,
con semilla 42, BCE ponderada, rotaciones, lote 16 y la misma GPU. En fold 0
se seleccionó `pool_dropout_wd`: pooling entre bloques, LR 0,0008, dropout 0,35
y weight decay 0,001. Su mejor checkpoint fue el de la época 11; la referencia
alcanzó su máximo en la época 12. Se comparan mejores checkpoints hasta el
presupuesto fijado, no exclusivamente los pesos de la última época.

La configuración seleccionada y la referencia se entrenaron después en folds
1 y 2, con el mismo presupuesto. AUC y F1 son por paciente; F1 usa umbral 0,5.

| Fold | AUC referencia | AUC pooling ajustado | F1 referencia | F1 pooling ajustado |
|---|---:|---:|---:|---:|
| 0 | 0.6292 | 0.6498 | 0.4459 | 0.5114 |
| 1 | 0.5748 | 0.5903 | 0.3740 | 0.4683 |
| 2 | 0.6653 | 0.6408 | 0.4810 | 0.4512 |
| Media | 0.6231 | 0.6270 | 0.4336 | 0.4770 |

La diferencia media es +0.0038 en AUC y
+0.0433 en F1. La mejora no es uniforme entre folds.
Las curvas muestran sobreajuste: la pérdida train puede seguir bajando mientras
sube la de validación. Por eso se conservan los mejores checkpoints, y aumentar
épocas por sí solo no se considera una mejora.

Esta comparación abarca tres de los cinco folds y una semilla. La elección de
hiperparámetros usa fold 0 y los checkpoints usan validación. Son resultados
exploratorios; no equivalen a una evaluación independiente ni anidada y no
sustituyen las métricas del PDF. Los diez pesos, calibración y umbral del modelo
histórico siguen conservados.

Evidencia en `resultados/04_entrenamiento/ajustes/`:
`comparacion.csv`, `revision_010/020/030.json`, `resultado_ajuste.json`,
`confirmacion.csv`, `resultado_confirmacion.json`, `comparacion_curvas.png`
y `confirmacion_curvas.png`. Cada run conserva código identificado por SHA-256,
configuración, entorno, curvas, OOF y pesos. Se ejecutaron diez entrenamientos
de 30 épocas en total; se cargaron cero imágenes del test reservado.

## Resultado de la comparación del 04/10/2026

Se completaron veinte entrenamientos: dos variantes × cinco folds × semillas
42 y 2026. Ambas utilizan lote 64 y treinta épocas con revisión en 10/20/30.
El código del experimento es el commit `726b704`; configuración, entorno y
SHA-256 de código, caché, pesos y predicciones están registrados. La carga de
caché se optimizó antes del ensayo definitivo; once épocas coincidieron
exactamente con la ejecución diagnóstica previa en ambas variantes.

La CNN actual tiene 551.913 parámetros y la pequeña 310.513 (43,7 % menos).
Los canales son la única diferencia del modelo. Se conserva cabeza de 64,
pooling entre bloques, dropout 0,35, LR 0,0008, weight decay 0,001, BCE ponderada
y aumentos. El checkpoint de cada run se elige por AUC de su validación.

| Métrica | Actual: media ± DE | Pequeña: media ± DE | OOF actual | OOF pequeña |
|---|---:|---:|---:|---:|
| AUC | 0.6115 ± 0.0346 | 0.5954 ± 0.0286 | 0.5932 | 0.5922 |
| Average precision | 0.3983 ± 0.0481 | 0.3933 ± 0.0373 | 0.3792 | 0.3840 |
| F1 | 0.3363 ± 0.1485 | 0.3750 ± 0.0778 | 0.3571 | 0.4252 |
| Precisión | 0.3677 ± 0.0443 | 0.3724 ± 0.0437 | 0.3947 | 0.3868 |
| Sensibilidad | 0.3660 ± 0.2039 | 0.4394 ± 0.2306 | 0.3261 | 0.4720 |
| Especificidad | 0.7428 ± 0.1525 | 0.6718 ± 0.2186 | 0.7923 | 0.6890 |
| Brier | 0.2365 ± 0.0112 | 0.2471 ± 0.0269 | 0.2297 | 0.2415 |

La media/DE describe las diez ejecuciones por variante. OOF combina las dos
semillas de validación por corte y después los cortes por paciente: 1.097
pacientes con predicciones hechas sin usarlas para aprender los pesos del run.
Las métricas OOF y la media de runs son resúmenes distintos. F1, sensibilidad
y especificidad usan el mismo umbral fijo de 0,5, sin calibración ni ajuste.

- AUC: diferencia OOF pequeña − actual -0.0010; IC95% [-0.0372, +0.0346].
- Average precision: diferencia OOF pequeña − actual +0.0048; IC95% [-0.0358, +0.0439].
- F1: diferencia OOF pequeña − actual +0.0680; IC95% [+0.0183, +0.1164].

La pequeña mejora AUC en 4/10 pares fold/semilla y average precision
en 4/10. No se ha demostrado una mejora concluyente en AUC: el intervalo pareado de desarrollo incluye cero.

La mejora OOF en F1 con umbral 0,5 tiene un IC pareado por encima de cero,
condicionado a estas predicciones. La sensibilidad aumenta de
0.3261 a 0.4720, pero la
especificidad baja de 0.7923 a
0.6890. La pequeña acierta
47 positivos más y produce
80 falsos positivos adicionales.
La AUC media por ejecución es menor y el Brier OOF es peor: el ahorro de
parámetros y el mayor F1 no constituyen una mejora global.

La comparación por cohorte muestra diferencias que el promedio puede ocultar:

| Cohorte | Pacientes | AUC OOF actual | AUC OOF pequeña |
|---|---:|---:|---:|
| duke | 209 | 0.6324 | 0.5209 |
| spy1 | 104 | 0.6193 | 0.6267 |
| spy2 | 784 | 0.5670 | 0.5908 |

Persiste el sobreajuste. En la época 30, la AUC media de train/validación es
0.9370/0.5318
en la actual y
0.8971/0.5287
en la pequeña. Estas cifras describen las curvas al final del presupuesto;
las métricas de las tablas corresponden a los mejores checkpoints de validación.
Reducir la red no ha resuelto la generalización. Se conservan ambas variantes
para investigación, sin promover automáticamente la pequeña.

Los IC se obtienen con 2.000 remuestreos pareados y estratificados por paciente,
condicionados a las predicciones OOF. No son una validación anidada, no incluyen
toda la incertidumbre de entrenamiento y no demuestran rendimiento en pacientes
de otra fuente. La elección previa de la configuración y de checkpoints utiliza
datos de desarrollo. El test reservado no se ha abierto; el modelo histórico,
sus pesos, calibración y umbral siguen conservados.

Resultados en `resultados/04_entrenamiento/comparacion_tamano/`: `resultado.json`,
`resumen_metricas.csv`, `comparacion_runs.csv`, `diferencias_pareadas.csv`,
`metricas_cohortes.csv`, `curvas_comparacion.png` y `roc_precision_recall.png`.
Se conservan los logs y pesos de las veinte ejecuciones dentro de esa carpeta.

## Experimento de generalización por paciente (04/10/2026)

La acción `generalizar` compara una sola hipótesis: aplicar BCE ponderada después
de la media de probabilidades de los cortes de una paciente, en lugar de aplicar
BCE ponderada a cada corte. Conserva la CNN actual ajustada (551.913 parámetros),
las fases, la escala fija, pooling entre bloques, dropout 0,35, LR 0,0008,
weight decay 0,001, AdamW, calendario, clipping y aumentos. No añade módulos ni
pesos preentrenados. El modelo histórico y las etapas 01–03 se verifican por hash.

La auditoría encuentra 1.097 pacientes y 10.945 cortes, casi siempre diez por
paciente (rango 5–10). La prevalencia por paciente y corte es prácticamente igual,
por lo que corregir solamente el número de cortes tendría poco efecto. La hipótesis
escogida alinea la función de coste con la agregación de inferencia y permite que
cortes poco informativos no reciban individualmente toda la supervisión clínica.
No presupone que esta modificación vaya a mejorar.

Ambas variantes usan exactamente los mismos lotes de pacientes completas: seis
pacientes, hasta 60 cortes con `--lote 64`. No hay relleno de imágenes. La referencia
se vuelve a entrenar con este protocolo; por tanto, sus resultados no se comparan
como equivalentes a los veinte runs anteriores con cortes mezclados en lotes.
Ambas usan el mismo N0/N1 de **cortes del subconjunto que aprende**, para aislar
la pérdida como hipótesis principal. La variante por paciente da un término por
paciente; la referencia da un término por corte. Como casi todas aportan diez
cortes, la diferencia de ponderación entre pacientes es pequeña. Los aumentos
son independientes entre cortes e idénticos entre las tres fases de cada corte.

Para una paciente con logits z_i, q=mean(sigmoid(z_i)). La nueva pérdida es
`-w*y*log(q) -(1-y)*log(1-q)`, promediada por paciente. Los logaritmos se calculan
con logsigmoid y logsumexp en float32, sin recortar probabilidades que anulen
sus gradientes. La CNN y la inferencia siguen siendo las del proyecto.

Protocolo fijado antes de observar los resultados nuevos:

- Cinco folds externos originales; semillas 42 y 2026; dos variantes; 30 épocas.
- En cada complemento externo: 70 % aprendizaje, 15 % selección de checkpoint
  y 15 % calibración/umbral. Reparto por paciente, estratificado por cohorte × pCR,
  con semillas de partición fijas e independientes de las semillas de entrenamiento.
- Checkpoint de máxima AUC por paciente en selección interna; primero en empates.
  No se utiliza el fold externo para curvas, parada o elección de épocas.
- Media de probabilidades por corte y semilla. Platt no negativo, regularizado
  con C=1, ajustado solamente en pacientes de calibración interna. Youden en esas
  mismas pacientes. Ajustar ambas decisiones aquí es desarrollo interno; sus
  resultados se evalúan exclusivamente en el fold externo.
- Todos los modelos, calibradores y umbrales quedan fijados antes de inferir
  los folds externos. El resultado conserva el hash de las decisiones.
- Criterio principal: media de AUC cruda de los cinco folds externos del ensemble
  de dos semillas. OOF agrupada es secundaria: las escalas de distintos folds
  pueden cambiar su orden relativo. AP, métricas al umbral fijo 0,5 y al umbral
  interno congelado, Brier, log-loss, ECE en diez bins fijos, curvas de calibración
  y resultados por cohorte completan la comparación.
- No se promueve un modelo automáticamente. Se conserva la referencia si no
  hay evidencia convincente en AUC y consistencia entre cohortes.

```bash
# Diagnóstico separado, sin posibilidad de selección:
.venv/bin/python -B 04_entrenamiento.py generalizar --prueba --workers 2 --paralelos 2 --dispositivo cuda --salida resultados/04_entrenamiento/generalizacion_diagnostico
# Experimento completo o reanudación, exactamente el mismo comando:
.venv/bin/python -B 04_entrenamiento.py generalizar --epocas 30 --revision-cada 10 --lote 64 --workers 2 --paralelos 2 --dispositivo cuda
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m unittest discover -s tests -v
```

Resultados en `resultados/04_entrenamiento/generalizacion_paciente/`: protocolo,
particiones por paciente, auditoría de todos los PNG **train** contra la caché,
inventario SHA-256 de imágenes y duplicados exactos entre pacientes de train,
configuraciones, entornos, versiones de paquetes, copia del código, curvas, checkpoints,
predicciones de selección/calibración/evaluación y métricas. Reanudar conserva
optimizador, calendario, scaler y RNG. Los cambios de configuración, código,
datos o archivos protegidos exigen una salida nueva. `selection_slices.csv`
identifica las predicciones internas; `oof_slices.csv` dentro de cada run se
conserva por compatibilidad del motor y **también es selección interna**, no la
OOF externa. La OOF externa es `oof_pacientes.csv` en la raíz del experimento.
La caché previa se reutiliza y se verifica contra los PNG para evitar otra copia.

Limitaciones: este diseño usa un holdout interno por fold, no una búsqueda completa
de hiperparámetros en múltiples folds internos. Entrena con aproximadamente el
56 % del desarrollo. Los hiperparámetros y la hipótesis están informados por
análisis históricos de estas mismas pacientes: separar las decisiones nuevas
no las convierte en una muestra independiente. Los IC de diferencias usan
2.000 bootstrap pareados por paciente, estratificados por fold/cohorte/clase,
con modelos, calibradores y umbrales fijos; no incorporan toda la incertidumbre
del entrenamiento ni del desarrollo adaptativo. Los folds comparten aprendizaje.
La calibración usa unas 132 pacientes por fold y puede ser inestable. El test
reservado ya observado permanece cerrado. No se pueden descartar duplicados
con test sin abrir sus imágenes, ni identidades diferentes de un mismo sujeto
sin información adicional de las fuentes. La selección de cortes difiere entre
Duke e I-SPY y puede introducir sesgos.

Fuentes metodológicas: [selección anidada de scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html),
[calibración de probabilidades](https://scikit-learn.org/stable/modules/calibration.html),
[BCE ponderada de PyTorch](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BCEWithLogitsLoss.html)
y [aprendizaje por bolsas de instancias, Ilse et al.](https://proceedings.mlr.press/v80/ilse18a.html).
Esta implementación usa una media fija y no implementa la atención del artículo.
