# Caso cáncer de mama

Predicción educativa de respuesta patológica completa (pCR) a partir de RM DCE
y variables clínicas. El modelo activo vuelve a ser **v002_multimodal_clinico_20261007**:
CNN 2D propia, BCE ponderada y diez pesos ya entrenados. La aplicación y la CLI
cargan su calibración Platt y su umbral originales, sin necesidad de reentrenar.

Los ensayos posteriores no demostraron superar al v002 original y quedan
cerrados como **fracasos del objetivo de mejora**. Esto incluye v003 y v004.
La mejora de BCE normal frente a un control reentrenado no justificaba sustituir
v002. Se conservan los resultados y todas las versiones de pesos.
Véase el [registro de experimentos descartados](docs/EXPERIMENTOS_DESCARTADOS.md).

## CNN solo imagen por fases: entrenamiento terminado

El ensayo del 14/10/2026 completó **20 ejecuciones y 326 épocas** en 1 h 58 min.
BCE ponderada obtuvo AUC medio por fold **0,5486 ± 0,0504** y AUC OOF global
**0,5246**; BCE normal, **0,5330 ± 0,0253** y **0,5087**, respectivamente.
El resultado es insuficiente: persiste sobreajuste y la calibración final
ponderada devuelve una probabilidad constante. No se promocionan estos pesos.

Se conservan los **20 pesos de inferencia** en
[v005, BCE normal](modelos/versiones/v005_cnn_imagen_fases_normal_20261014/) y
[v006, BCE ponderada](modelos/versiones/v006_cnn_imagen_fases_ponderada_20261014/),
con código compatible y SHA-256. V002 continúa activa en la aplicación multimodal.

- [Informe completo, curvas y límites](resultados/14_informe_cnn_imagen_20261014/INFORME_RESULTADOS.md).
- [Métricas y predicciones](resultados/14_informe_cnn_imagen_20261014/).
- [Protocolo prefijado de CNN por fases](docs/mejoras_cnn/PROTOCOLO_CNN_IMAGEN_FASES.md).

El entrenamiento independiente de imagen permanece reproducible desde
`cancer_mama.entrenamiento_imagen`: CNN desde cero, PRE/EARLY/LATE, encoder
compartido, sin variables clínicas, media de cortes y selección interna por
paciente. El lanzador GPU se conserva en `scripts/entrenar_cnn_imagen_gpu.ps1`.
Los checkpoints de trabajo y estados de optimización permanecen locales.

## CNN cinética multiescala: ajuste terminado, resultado insuficiente

El ensayo del 21/10/2026 completó **60 entrenamientos, 948 épocas y 220.884 pasos
de AdamW** en 3 h 17 min. Se conservó la arquitectura de `39bd951`, con
593.521 parámetros, mapas de realce explícitos y recortes de 256/128/64 píxeles.
Solo se compararon tres pares de LR/weight decay, dos pérdidas y dos semillas.
Cada fold eligió los parámetros internamente antes de la evaluación exterior.

El candidato obtuvo **AUC global OOF cruda 0,5264**, media por fold
**0,5766 ± 0,0301** y AUC con calibración interna **0,5618**. La misma red de
referencia obtuvo 0,5360 global. No se alcanzó el mínimo 0,60 ni el objetivo
0,70; persiste el sobreajuste y el ajuste no demuestra mejora.

Se conservan los diez pesos seleccionados, configuración, calibración y código
en [v007](modelos/versiones/v007_cnn_cinetica_multiescala_20261021/), como
`ensayo_insuficiente_conservado`. **V002 sigue activo.** El test reservado no se
abrió y no se inicia otro ensayo antes de revisar estos resultados.

- [Informe completo y curvas](resultados/15_informe_cinetico_20261021/INFORME_RESULTADOS.md).
- [Métricas, intervalos, predicciones y evidencia](resultados/15_informe_cinetico_20261021/).

## Piloto anti-sobreajuste (preparado, no entrenado)

El siguiente ensayo vuelve a una CNN temporal simple de una escala y compara un
control con un paquete anti-sobreajuste: lotes por paciente, peso igual por
paciente, Dropout2d y aumentos conjuntos suaves. Son 16 ejecuciones en los folds
0 y 3; sirven para filtrar la dirección antes de pagar otra matriz completa. El
test permanece cerrado y el criterio de éxito está prefijado en el
[protocolo](docs/mejoras_cnn/PROTOCOLO_PILOTO_ANTIOVERFIT.md).

```powershell
powershell -ExecutionPolicy Bypass -File scripts/piloto_antioverfit_gpu.ps1
```
- [Protocolo original y ampliación del ajuste](docs/mejoras_cnn/PROTOCOLO_CNN_CINETICA.md).

El módulo `cancer_mama.ajuste_cinetico` conserva la búsqueda ejecutada en Linux.
`cancer_mama.entrenamiento_cinetico` y el lanzador PowerShell conservan la matriz
original de 20 ejecuciones. Ambos emplean la misma arquitectura, desde cero y
sin variables clínicas. Para comprobar la configuración del ajuste sin entrenar:

```bash
.venv/bin/python -B -m cancer_mama.ajuste_cinetico verificar
```

## Estructura

```text
src/cancer_mama/          # datos, entrenador multimodal y candidato solo imagen
tests/                   # pruebas del código activo
app.py                   # aplicación Streamlit con v002
scripts/                 # lanzador GPU y comprobación de inferencia
modelos/versiones/       # versiones congeladas con pesos y código compatible
resultados/              # evidencia y ejecuciones locales
docs/                    # guías, protocolos y decisiones
```

El dataset `breastdcedl/`, entornos virtuales y checkpoints de trabajo son locales.
Los pesos seleccionados de `modelos/versiones/` están en Git. El código de datos
conserva la base de Marta Navarro Guil, commit `69a44eb` (28/09/2026).

## Preparación

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
```

En Windows, sustituir `.venv/bin/python` por `.venv/Scripts/python.exe`.
Para GPU, instalar primero la compilación de PyTorch apropiada desde su
[selector oficial](https://pytorch.org/get-started/locally/).
No copiar el entorno virtual de otro ordenador. Las dependencias activas están
en `requirements.txt`; torchvision y los pesos preentrenados no son necesarios.

Para inferir imágenes nuevas no hace falta el dataset. Los módulos
`cancer_mama.descarga`, `auditoria`, `eda` y `preparacion` conservan las etapas
de preparación; consultar su `--help` antes de repetirlas sobre los datos existentes.

## Usar el multimodal original

```bash
.venv/bin/python -B -m cancer_mama.entrenamiento verificar
.venv/bin/python -m streamlit run app.py
```

Manifiesto activo:
`modelos/versiones/v002_multimodal_clinico_20261007/manifiesto.json`.
Agregación: mediana. Calibración: Platt. Umbral guardado: **0,3543331693**.
La verificación comprueba hashes y compatibilidad de pesos; no entrena ni evalúa test.

La aplicación acepta un corte PRE/EARLY/LATE e introduce edad, volumen tumoral,
HR y HER2. Los ejemplos locales solo aportan imágenes; las variables clínicas
se introducen expresamente o se dejan como desconocidas. El mapa EARLY−PRE es
descriptivo; la CNN recibe las tres fases originales. Al cambiar una entrada se
oculta el resultado anterior. La evaluación de desarrollo usa varios cortes por
paciente; para agregarlos todos puede utilizarse la CLI:

```bash
.venv/bin/python -B -m cancer_mama.entrenamiento predecir \
  --pre paciente_z000_PRE.png \
  --early paciente_z000_EARLY.png \
  --late paciente_z000_LATE.png \
  --edad 49 --volumen-tumoral 12 --hr 1 --her2 0
```

Los PNG deben ser monocromos de 8 bits, 256 × 256, de la misma paciente y corte.
Se admiten varios cortes en el mismo orden para las tres fases. HR y HER2 son
0/1. Una variable omitida se imputa con las estadísticas guardadas de cada modelo;
para declarar todas desconocidas, pasar `--edad nan`. El volumen usa `log1p`.
El ensemble promedia modelos por corte, aplica mediana entre cortes y después
la calibración y el umbral guardados.

## Entrenamiento activo

Un único perfil: `pool_dropout_wd_clinical`, CNN 2D desde cero con 551.921
parámetros, ocho convoluciones, BatchNorm, canales 24/48/96/160 y rama clínica.
Mantiene dropout 0,35, weight decay 0,001, AdamW LR 0,0008, calendario coseno,
lote 64, BCE ponderada por corte y aumentos geométricos compartidos entre fases.
La imputación, escala e inicialización clínica se ajustan solo con train del fold.
Se selecciona por AUC de validación, hasta 46 épocas, mínimo 12 y paciencia 10.

Para un nuevo entrenamiento, cuando sea necesario:

```bash
.venv/bin/python -B -m cancer_mama.entrenamiento entrenar \
  --dispositivo cuda --salida resultados/04_entrenamiento/multimodal_actual
.venv/bin/python -B -m cancer_mama.entrenamiento comparar \
  --salida resultados/04_entrenamiento/multimodal_actual
```

Son diez ejecuciones: semillas 42/2026 × cinco folds. Una ejecución nueva no
reemplaza los pesos activos. Reanudar exige el mismo código, datos y configuración;
si cambian, usar otra salida. La [guía GPU](docs/GUIA_GPU_MULTIMODAL.md) incluye
el lanzador Windows. Para comprobar CUDA con un lote y una época, excluidos de
comparación:

```bash
.venv/bin/python -B -m cancer_mama.entrenamiento entrenar --prueba \
  --semillas 42 --folds 0 --epocas 1 --workers 0 --lote 8 --dispositivo cuda
```

El proyecto mantiene una única rama, `main`, con el entrenador original v002.
El código de los ensayos descartados se conserva en commits anteriores, sin
ramas adicionales ni copias de archivo en el árbol actual. El
[registro de experimentos](docs/EXPERIMENTOS_DESCARTADOS.md) reúne resultados,
motivos del descarte y commits para recuperar las fuentes.
El PDF exige CNN 2D desde cero y prohíbe transfer learning y redes preentrenadas.

## Ensayo FiLM cerrado

Las 40 ejecuciones terminaron el 11/10/2026 sin cumplir el criterio de mejora.
AUC OOF de FiLM ponderado: **0,7014**, frente a **0,7020** del control equiparado.
Persistió el sobreajuste. Se mantiene **v002**; la comparación descriptiva con
sus pesos históricos (AUC 0,7246) también favorece conservarlo.

El [registro de experimentos descartados](docs/EXPERIMENTOS_DESCARTADOS.md)
reúne métricas, curvas, límites de la comparación y evidencia verificable.
El [protocolo prefijado](docs/mejoras_cnn/PROTOCOLO_FILM.md) queda como documento
histórico. El módulo FiLM y sus pruebas se conservan en el commit `9e2aa93` de
`main` y se retiran del árbol actual. Los pesos de trabajo se conservan localmente.

## Ensayo SAM cerrado

Las 40 ejecuciones terminaron el 11/10/2026. SAM ponderado sube el AUC de su
control de 0,7020 a 0,7071, pero el intervalo de confianza incluye cero y no cumple
el criterio prefijado. Su aporte visual es prácticamente nulo: al retirar el
término de imagen se mantiene el AUC. **V002 original sigue activo.**

El [registro de descartes](docs/EXPERIMENTOS_DESCARTADOS.md) contiene resultados,
curvas, ablación y comparación descriptiva con los pesos originales de v002.
El [protocolo SAM](docs/mejoras_cnn/PROTOCOLO_SAM.md) se conserva como documento
histórico; el módulo y sus pruebas están en el commit `4a2c959` de `main` y se
retiran del árbol actual. Los checkpoints de trabajo se conservan localmente.

## Evidencia y comprobaciones

V002 obtuvo ROC-AUC OOF **0,7246**. Con su calibración y umbral guardados:
log-loss 0,5367, Brier 0,1791, 187 falsos positivos y 128 falsos negativos.
Son métricas aparentes de desarrollo: las mismas OOF intervienen en selección,
calibración y umbral. No equivalen a evaluación independiente en pacientes nuevas.
El test reservado no se usa para escoger entre versiones.

```bash
.venv/bin/python -B -m unittest discover -s tests -v
.venv/bin/python -B modelos/versionar.py verificar
.venv/bin/python -B scripts/verifica_multimodal.py
```

La última comprobación usa un ejemplo de train y compara aplicación, CPU, CLI y
paquete archivado en FP32 (TF32 desactivado en el proceso de prueba); sus resultados se guardan en `.codex_tmp/verificacion_multimodal/`.
No necesita ejecutar nuevos entrenamientos.

- [Historial de versiones](modelos/README.md): v002 activa; v001, v003 y v004 conservadas.
- [Informe original](resultados/06_informe_multimodal_20261007/INFORME_RESULTADOS.md).
- [Registro de ensayos cerrados](docs/EXPERIMENTOS_DESCARTADOS.md).
- [Protocolo multimodal original](docs/mejoras_cnn/PROTOCOLO_MULTIMODAL_CLINICO.md).

Proyecto de uso educativo. Las estimaciones no se usan para tomar decisiones de tratamiento.
