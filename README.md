# Caso cáncer de mama

Predicción educativa de respuesta patológica completa (pCR) a partir de resonancia
DCE anterior al tratamiento y variables clínicas. La base activa es el **multimodal
clínico v002**, ya entrenado: `pool_dropout_wd_clinical`, BCE ponderada por corte
y ensemble de diez modelos. Sus pesos se cargan directamente; no hace falta
volver a entrenarlos.

El ensayo de realce temporal, con y sin mayor regularización, queda cerrado como
**fracaso del objetivo de mejora**. Se conserva el multimodal anterior. Métricas,
decisión y límites están en [EXPERIMENTOS_DESCARTADOS.md](docs/EXPERIMENTOS_DESCARTADOS.md).
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
├── app.py                # aplicación Streamlit con el multimodal v002
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
`modelos/versiones/v002_multimodal_clinico_20261007/manifiesto.json`.
Conserva mediana entre cortes, calibración Platt y umbral 0,3543331693.
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
cortes antes de calibrar. `tum_vol` se transforma mediante `log1p`; los cuatro
valores se imputan y escalan con estadísticas del train de cada fold, junto con
cuatro indicadores de ausencia.

## Aplicación educativa

```bash
.venv/bin/python -m streamlit run app.py
```

La aplicación carga v002 y permite subir PRE/EARLY/LATE e introducir edad,
volumen tumoral, HR y HER2, indicando los desconocidos. Los ejemplos locales
solo aportan imágenes: sus datos clínicos deben introducirse expresamente.
La vista de realce EARLY−PRE es descriptiva; la CNN recibe las tres fases originales.

La interfaz estima un solo corte; la calibración de desarrollo se ajustó con
varios cortes por paciente. Para agregar todos los cortes disponibles usar la CLI.
El resultado guardado se invalida cuando cambian las imágenes o los datos clínicos.

## Único entrenamiento activo

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

## Evidencia conservada y pruebas

La base v002 obtuvo AUC OOF 0,724568 y AP 0,531646 sobre 1.097 pacientes de
desarrollo. Es evaluación interna con selección de checkpoints y calibración
sobre OOF; no equivale a validación externa. No se vuelve a abrir el test reservado
para escoger modelos.

- [Historial de versiones](modelos/README.md): v001 y v002 con pesos y código propios.
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
