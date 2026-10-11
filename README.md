# Caso cáncer de mama

Predicción educativa de respuesta patológica completa (pCR) a partir de resonancia
DCE anterior al tratamiento y variables clínicas. La base activa es el **multimodal
clínico v004**, ya entrenado: CNN 2D propia, BCE normal, selección interna por
log-loss y ensemble de diez modelos. Sus pesos se cargan directamente; no hace falta
volver a entrenarlos.

El ensayo de realce temporal, con y sin mayor regularización, queda cerrado como
**fracaso del objetivo de mejora**. Después, la comparación de BCE normal
redujo la pérdida frente al control equiparado y superó la repetición con otras
semillas. Las curvas todavía muestran sobreajuste; no se afirma una mejora
concluyente de ROC-AUC ni de generalización externa. Véanse el
[informe de confirmación](resultados/11_confirmacion_bce_normal_20261011/INFORME_RESULTADOS.md)
y el [registro de ensayos descartados](docs/EXPERIMENTOS_DESCARTADOS.md).
Los modelos y resultados anteriores permanecen como histórico.

## Estructura

```text
caso-cancer-mama/
├── src/cancer_mama/       # datos, EDA, multimodal e informes
├── tests/                # contratos de datos, pesos y reproducibilidad
├── docs/                 # guías, protocolos y experimentos cerrados
├── resultados/           # evidencia y ejecuciones locales por etapa
├── modelos/versiones/    # pesos, manifiestos y código compatible congelado
├── scripts/entrenar_multimodal_gpu.ps1
├── app.py                # aplicación Streamlit con el multimodal v004
├── pyproject.toml
├── requirements.txt
└── CONTRIBUTING.md
```

El dataset `breastdcedl/`, los entornos y checkpoints de trabajo son locales.
Las versiones congeladas de `modelos/` contienen los pesos necesarios para inferir
y están en Git. El código de datos conserva la base de Marta Navarro Guil,
commit `69a44eb` (28/09/2026).

## Preparación

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
```

En Windows, sustituir `.venv/bin/python` por `.venv/Scripts/python.exe`.
Para GPU, instalar primero la compilación de PyTorch adecuada para el sistema
desde su [selector oficial](https://pytorch.org/get-started/locally/).
No copiar un entorno virtual de otro ordenador.

Para utilizar el modelo con imágenes nuevas no es necesario descargar el dataset.
Para preparar o entrenar, usar la copia local de `breastdcedl/`; si no existe:

```bash
.venv/bin/python -m cancer_mama.descarga
```

Los módulos `cancer_mama.auditoria`, `cancer_mama.eda` y `cancer_mama.preparacion`
conservan los pasos de auditoría, exploración y preparación. Consultar `--help`
de cada módulo antes de repetirlos sobre una copia ya preparada.

## Usar los pesos multimodales existentes

Desde la raíz, verificar los diez pesos de la base activa:

```bash
.venv/bin/python -B -m cancer_mama.entrenamiento verificar
```

El manifiesto por defecto es
`modelos/versiones/v004_multimodal_bce_normal_20261011/manifiesto.json`.
Conserva mediana entre cortes y probabilidades directas de BCE normal, sin Platt
posterior. El umbral operativo Youden queda guardado en el manifiesto; se ajustó
en OOF y no constituye una evaluación independiente de sensibilidad/especificidad.
La verificación comprueba hashes y compatibilidad de pesos; no entrena ni abre test.

Para predecir, pasar las tres fases y los datos clínicos de la misma paciente:

```bash
.venv/bin/python -B -m cancer_mama.entrenamiento predecir \
  --pre paciente_z000_PRE.png \
  --early paciente_z000_EARLY.png \
  --late paciente_z000_LATE.png \
  --edad 49 --volumen-tumoral 12 --hr 1 --her2 0
```

Las fases son PNG monocromos de 8 bits y 256 × 256 píxeles. Se admiten varios
cortes de una única paciente, en el mismo orden en las tres opciones. HR y HER2
se codifican como 0/1. Una variable omitida se trata como ausente: cada modelo
usa su imputación guardada. Para declarar todos los datos desconocidos, pasar
`--edad nan`; no sustituir ausencias por cero.

El ensemble promedia las diez probabilidades por corte y aplica la mediana entre
cortes. En v004 la transformación posterior es identidad. `tum_vol` se transforma
mediante `log1p`; los cuatro
valores se imputan y escalan con estadísticas del train de cada fold, junto con
cuatro indicadores de ausencia.

## Aplicación educativa

```bash
.venv/bin/python -m streamlit run app.py
```

La aplicación carga v004 y permite subir PRE/EARLY/LATE e introducir edad,
volumen tumoral, HR y HER2, indicando los desconocidos. Los ejemplos locales
solo aportan imágenes: sus datos clínicos deben introducirse expresamente.
La vista de realce EARLY−PRE es descriptiva; la CNN recibe las tres fases originales.

La interfaz estima un solo corte; la evaluación principal de desarrollo usa
varios cortes por paciente. Para agregar todos los cortes disponibles usar la CLI.
El resultado guardado se invalida cuando cambian las imágenes o los datos clínicos.

## Entrenamiento histórico ponderado

El módulo `cancer_mama.entrenamiento` conserva el procedimiento histórico
ponderado como control. No reproduce el nuevo candidato v004: para este se usa
el protocolo BCE normal de más abajo. Su CLI de inferencia sí carga v004 por
defecto.

La CNN multimodal tiene 551.921 parámetros, ocho convoluciones, BatchNorm,
canales 24/48/96/160, pooling intermedio y rama clínica. Se mantienen dropout 0,35,
weight decay 0,001, AdamW con LR 0,0008, calendario coseno, AMP para aprendizaje
en CUDA, aumentos geométricos compartidos entre fases y BCE ponderada por corte.
La imputación, escala e inicialización clínica se ajustan solo con train del fold.

Se conserva la selección de checkpoint por máxima AUC de validación por paciente
(media de cortes en esa selección), máximo 46 épocas, mínimo 12 y paciencia 10.
OOF e inferencia usan FP32, sin aumentos. La comparación final usa la mediana
ya adoptada; no vuelve a buscar otra arquitectura, pérdida o agregación.

Los perfiles de imágenes solas, BCE por paciente y realce temporal se retiraron
del entrenador. Ya no existen `--configuraciones`, `--perdidas` ni `--pooling`.
Las versiones históricas se ejecutan con su código congelado mediante
[modelos/versionar.py](modelos/README.md).

Solo si se decide realizar un nuevo entrenamiento:

```bash
.venv/bin/python -B -m cancer_mama.entrenamiento entrenar \
  --dispositivo cuda --salida resultados/04_entrenamiento/multimodal_actual
.venv/bin/python -B -m cancer_mama.entrenamiento comparar \
  --salida resultados/04_entrenamiento/multimodal_actual
```

Por defecto son diez runs: semillas 42/2026 × cinco folds, todos ponderados.
La salida nueva mantiene v002 intacto. Checkpoints, historial, revisiones y curvas
permiten reanudar una ejecución compatible sin repetir lo ya completado.
La reanudación exige el mismo código, datos y configuración; tras un cambio
de código se debe usar otra salida, no forzar la reanudación de un ensayo antiguo.

Para comprobar la instalación con un lote y una época, excluidos de comparación:

```bash
.venv/bin/python -B -m cancer_mama.entrenamiento entrenar --prueba \
  --semillas 42 --folds 0 --epocas 1 --workers 0 --lote 8 --dispositivo cuda
```

El [lanzador y guía GPU](docs/GUIA_GPU_MULTIMODAL.md) describen el mismo perfil
en Windows. No se inicia ningún entrenamiento al abrir la aplicación o verificar pesos.

## Ensayo de pérdidas y LR del multimodal

El módulo experimental `cancer_mama.experimento_loss` compara LR 0,0008/0,0002
con selección independiente por AUC o log-loss por paciente. Registra pérdidas
comparables desde época 0 y reserva una selección interna separada del fold
exterior. Conserva los pesos, la calibración y la aplicación de v002.

El [protocolo del ensayo](docs/mejoras_cnn/PROTOCOLO_LOSS_LR_MULTIMODAL.md) fija
particiones, métricas y criterios antes de entrenar. En el servidor Linux:

```bash
.venv/bin/python -u -B -m cancer_mama.experimento_loss ejecutar --prueba
.venv/bin/python -u -B -m cancer_mama.experimento_loss ejecutar
```

El segundo comando ejecuta veinte trayectorias y genera la comparación al acabar.
Se puede reanudar con el mismo comando; el progreso queda en
`resultados/04_entrenamiento/loss_lr_multimodal_20261010/estado.json`.

El ensayo terminó y la variante LR 0,0008 con selección por log-loss se conserva
como `v003_multimodal_loss_20261010`. **Es una candidata anterior que no demostró mejora concluyente.** El
[informe de resultados](resultados/08_informe_loss_20261010/INFORME_RESULTADOS.md)
muestra una reducción de pérdida cruda del 5,6 % frente al control equiparado.
Tras calibrar ambos con selección interna, la reducción es del 0,9 % y el intervalo
de diferencia incluye cero; no demuestra superioridad sobre v002.

```bash
.venv/bin/python -B -m cancer_mama.informe_loss
.venv/bin/python -B -m cancer_mama.informe_loss --solo-presentacion
```

El primer comando requiere la ejecución local completa; el segundo regenera las
figuras y el informe con la evidencia conservada. La candidata se puede cargar con
`modelos/versionar.py predecir v003_multimodal_loss_20261010 -- ...`, pasando las
mismas imágenes y variables clínicas que a v002.

## Ensayos históricos descartados: transferencia y fusión

Estos ensayos usan arquitecturas prohibidas por el enunciado y no forman parte
de la solución docente. Sus comandos se conservan exclusivamente para auditoría.

El [protocolo de transferencia](docs/mejoras_cnn/PROTOCOLO_TRANSFER_MULTIMODAL.md)
recoge estudios comparables, diferencias entre sus datos y los nuestros, y una
rejilla fijada antes de entrenar. ResNet50 ImageNet V2 extrae características
congeladas a dos escalas. Se entrenan tres formas de fusionarlas con la clínica,
con imputación, PCA y elección de hiperparámetros dentro de fit/selección interna.

```bash
.venv/bin/python -m pip install -r requirements-transfer.txt
.venv/bin/python -B -m cancer_mama.experimento_transfer ejecutar --prueba
.venv/bin/python -u -B -m cancer_mama.experimento_transfer ejecutar
```

El servidor ya tiene torch 2.14.0; solo se añadió torchvision 0.29.0. Estos
ensayos no modificaron la aplicación. Se conservan probabilidades por paciente,
selección de modelos,
controles AUC/log-loss del ensayo anterior y diferencias pareadas con incertidumbre.
Este módulo requiere la evidencia local del ensayo anterior para verificar su
control. `estado.json` informa progreso y errores; el mismo comando reanuda fases
completadas si fuentes, entorno y datos siguen siendo idénticos.

La primera tanda no mejoró al control. La [segunda tanda](docs/mejoras_cnn/PROTOCOLO_CARACTERISTICAS_LOCALES.md)
usa características espaciales de AlexNet pool2, selección supervisada dentro de
fit y SVM/logística. Mantiene los controles y reporta la búsqueda adaptativa:

```bash
.venv/bin/python -B -m cancer_mama.experimento_local --prueba
.venv/bin/python -u -B -m cancer_mama.experimento_local
```

La [tercera tanda](docs/mejoras_cnn/PROTOCOLO_AJUSTE_PARCIAL.md) permite adaptar
layer4 de ResNet18 con L2-SP, conservando una base clínica fija y una corrección
visual acotada. Entrena por paciente con dos semillas y selecciona por pérdida:

```bash
.venv/bin/python -B -m cancer_mama.experimento_finetune --prueba
.venv/bin/python -u -B -m cancer_mama.experimento_finetune
```

La [cuarta tanda](docs/mejoras_cnn/PROTOCOLO_TEXTURA.md) extrae descriptores de
textura y diferencias entre fases de los PNG, conservando las mismas variables
clínicas. No interpreta esos descriptores como medidas físicas de perfusión:

```bash
.venv/bin/python -B -m cancer_mama.experimento_textura --prueba
.venv/bin/python -u -B -m cancer_mama.experimento_textura
```

## Restricción docente y mejora compatible

El enunciado PDF (página 6, sección 7) obliga a usar una CNN 2D desde cero y
prohíbe modelos preentrenados y transfer learning. Las tandas ResNet/AlexNet
anteriores quedan descartadas para la entrega; la textura es un comparador
auxiliar. Se conservan por trazabilidad en el
[informe de investigación descartada](resultados/09_informe_transferencia_20261010/INFORME_RESULTADOS.md).
Ninguna sustituyó v002.

La [comparación BCE normal y mixup](docs/mejoras_cnn/PROTOCOLO_BCE_MIXUP.md)
conserva la CNN original y el contrato PNG. Usa como control la BCE ponderada ya
entrenada con la misma separación interna por paciente:

```bash
.venv/bin/python -B -m cancer_mama.experimento_bce_mixup --prueba
.venv/bin/python -u -B -m cancer_mama.experimento_bce_mixup
```

La tanda compatible completó veinte entrenamientos: BCE normal mostró una
reducción de pérdida interna; mixup no mejoró esa alternativa. Las curvas todavía
muestran sobreajuste y siete de diez checkpoints de cada brazo seleccionan época
0. Véase el [informe completo](resultados/10_informe_bce_mixup_20261011/INFORME_RESULTADOS.md).

La [comprobación de estabilidad](docs/mejoras_cnn/PROTOCOLO_CONFIRMACION_BCE.md)
repite candidata y control con semillas 73 y 1701, manteniendo configuración y
particiones. No es validación externa:

```bash
.venv/bin/python -B -m cancer_mama.confirma_bce_normal --prueba
.venv/bin/python -u -B -m cancer_mama.confirma_bce_normal
```

La repetición terminó y cumplió el criterio fijado antes de entrenar. El
[informe de confirmación](resultados/11_confirmacion_bce_normal_20261011/INFORME_RESULTADOS.md)
guarda métricas por semilla, intervalos y limitaciones. v004 conserva los diez
pesos originales 42/2026; las semillas 73/1701 se usaron solo para comprobar
estabilidad. Siete de los diez pesos originales seleccionan época 0: no se ha
demostrado una contribución visual independiente.

Para regenerar ensayos antiguos tras cambios de código hay que usar el commit
registrado en su evidencia y un entorno compatible (para la repetición BCE,
commit `41428a2`, anterior a la adaptación de la interfaz); los protocolos rechazan
reanudar con fuentes o datos modificados. La inferencia portable de cada versión
usa su código congelado.

## Evidencia conservada y pruebas

La base histórica v002 obtuvo AUC OOF 0,724568 y AP 0,531646 sobre 1.097 pacientes de
desarrollo. Es evaluación interna con selección de checkpoints y calibración
sobre OOF; no equivale a validación externa. No se vuelve a abrir el test reservado
para escoger modelos.

- [Historial de versiones](modelos/README.md): v001–v004 con pesos y código propios.
- [Registro de ensayos descartados](docs/EXPERIMENTOS_DESCARTADOS.md).
- [Protocolo multimodal original](docs/mejoras_cnn/PROTOCOLO_MULTIMODAL_CLINICO.md).
- `resultados/06_informe_multimodal_20261007/`: informe del multimodal.
- `resultados/04_entrenamiento/realce_temporal_20261009/`: evidencia del ensayo cerrado.
- Los generadores de informes de ensayos previos se conservan para auditoría;
  su reproducción exige el código y los hashes del ensayo correspondiente.

```bash
.venv/bin/python -B -m unittest discover -s tests -v
.venv/bin/python -B modelos/versionar.py verificar
```

Proyecto de uso educativo. Las estimaciones no son un diagnóstico ni se usan
para tomar decisiones de tratamiento.
