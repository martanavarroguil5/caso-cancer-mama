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
