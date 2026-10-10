# Caso cáncer de mama

Predicción educativa de respuesta patológica completa (pCR) a partir de resonancia
DCE anterior al tratamiento. Se conserva la base de Marta Navarro Guil,
commit `69a44eb` (28/09/2026), y se añade únicamente el paso de entrenamiento.
El código instalable está en `src/cancer_mama/` y los resultados reproducibles
se conservan por etapa.

## Estructura

```text
caso-cancer-mama/
├── src/cancer_mama/    # paquete: datos, EDA, entrenamiento e informes
├── tests/               # pruebas unitarias y contratos de reproducibilidad
├── docs/                # protocolos, guías e informes finales
├── resultados/          # evidencia ligera versionada por etapa
├── modelos/             # versiones completas con pesos y código compatible
├── pyproject.toml       # metadatos, instalación y comandos de consola
├── requirements.txt     # dependencias con rangos reproducibles
├── CONTRIBUTING.md      # flujo de ramas, commits y reproducibilidad
└── README.md
```

`breastdcedl/`, los entornos virtuales, checkpoints y ejecuciones pesadas son
locales y están excluidos mediante `.gitignore`.

## Preparación

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m cancer_mama.descarga
```

En Windows, utiliza `.venv\Scripts\python.exe`. Los datos se guardan en
`breastdcedl/` y no hace falta volver a descargarlos si ya están en el ordenador.
Los modelos seleccionados se conservan en Git, organizados por versiones en
`modelos/versiones/`, con sus pesos, manifiestos y código compatible. El dataset
y los checkpoints de trabajo siguen fuera de Git. Consulta el
[historial de modelos](modelos/README.md) para recuperar y usar cada versión.

## Abrir y entrenar en otro ordenador con GPU

El experimento nuevo de realce temporal ya tiene un lanzador que comprueba
tests y CUDA, ejecuta dos humos, entrena 20 runs reanudables y genera la
comparación OOF:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/entrenar_mejora_gpu.ps1
```

Compara `enhancement_clinical` (PRE/EARLY/LATE más tres diferencias temporales)
con `enhancement_regularized_clinical` (el mismo modelo con regularización más
fuerte). Ambos usan BCE ponderada, semillas 42/2026 y cinco folds. Para validar
la instalación sin lanzar la matriz completa, añadir `-SoloHumo`. La
[guía GPU](docs/GUIA_GPU_MULTIMODAL.md) detalla el protocolo y qué carpeta debe
volver al ordenador principal.

Git solo descarga archivos confirmados y enviados al remoto. Antes de cambiar de
ordenador hay que comprobar que el código multimodal está en un commit y se ha
hecho `git push`. `git clone` incluye las versiones archivadas de los modelos;
el dataset y los checkpoints de las ejecuciones en curso se copian por separado.

En el ordenador con GPU:

```powershell
git clone https://github.com/martanavarroguil5/caso-cancer-mama.git
cd caso-cancer-mama
```

La carpeta de datos que se copie por separado debe quedar exactamente así:

```text
caso-cancer-mama/
├── breastdcedl/
│   ├── dataset/
│   ├── metadata/
│   └── documentation/
├── src/cancer_mama/
├── pyproject.toml
└── requirements.txt
```

No se debe copiar ni reutilizar `.venv/` del primer ordenador: depende del sistema
y de su instalación de PyTorch. Crear un entorno nuevo, preferiblemente con
Python 3.12:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
```

`requirements.txt` contiene las dependencias de Python, pero no puede instalar
el controlador NVIDIA ni elegir por sí solo la compilación CUDA apropiada. Hay
que instalar primero PyTorch CUDA usando el comando generado por el
[selector oficial de PyTorch](https://pytorch.org/get-started/locally/) para el
sistema y controlador concretos. Después se instala el resto:

```powershell
.venv\Scripts\python.exe -m pip install -e .
```

La instalación debe verificarse antes de entrenar:

```powershell
nvidia-smi
.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'SIN CUDA')"
```

`torch.cuda.is_available()` debe devolver `True`. A continuación se ejecutan los
tests y un humo de un único lote, que no es seleccionable ni sirve como resultado:

```powershell
.venv\Scripts\python.exe -B -m unittest discover -s tests -v

.venv\Scripts\python.exe -B -m cancer_mama.entrenamiento entrenar `
  --prueba --configuraciones pool_dropout_wd_clinical `
  --perdidas ponderada --semillas 42 --folds 0 `
  --epocas 1 --lote 8 --workers 0 --dispositivo cuda `
  --salida resultados/04_entrenamiento/multimodal_clinico_20261007
```

Si ambas comprobaciones terminan bien, la matriz completa es una configuración ×
dos pérdidas × dos semillas × cinco folds: **20 entrenamientos secuenciales**.

```powershell
.venv\Scripts\python.exe -B -m cancer_mama.entrenamiento entrenar `
  --configuraciones pool_dropout_wd_clinical `
  --perdidas normal ponderada --semillas 42 2026 --folds 0 1 2 3 4 `
  --epocas 46 --lote 64 --workers 4 --dispositivo cuda `
  --salida resultados/04_entrenamiento/multimodal_clinico_20261007
```

Repetir exactamente el mismo comando y la misma salida reanuda cada run desde
su última época completa. Si Windows da errores con los procesos de lectura,
puede usarse `--workers 0` o `--workers 2`. Si se modifica el lote u otro
hiperparámetro, debe utilizarse una salida nueva para no mezclar configuraciones.

Cuando terminen los veinte runs:

```powershell
.venv\Scripts\python.exe -B -m cancer_mama.entrenamiento comparar `
  --salida resultados/04_entrenamiento/multimodal_clinico_20261007
```

Hay que conservar y copiar de vuelta toda esa carpeta de resultados: contiene
los checkpoints `.pt`, las curvas, predicciones OOF, métricas y el manifiesto del
ensemble. Para publicar el modelo seleccionado se usa
`modelos/versionar.py guardar`; los checkpoints intermedios no se suben automáticamente.
Se recomienda entrenar en un SSD local, evitar que el ordenador se suspenda y disponer de al menos 10 GiB
libres. La versión ampliada de esta lista está en
[guía GPU](docs/GUIA_GPU_MULTIMODAL.md).

## Aplicación web

La interfaz de demostración está en `app.py`. Valida de forma estricta las tres
fases, muestra PRE/EARLY/LATE y el mapa de realce, y conserva exactamente el
preprocesamiento, calibración y umbral del manifiesto. Para abrirla localmente:

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

La inferencia requiere los diez pesos referenciados por
`resultados/04_entrenamiento/modelo_final.json`. Deben copiarse a
`resultados/04_entrenamiento/modelos/` antes de desplegar; la interfaz detecta
su ausencia y nunca muestra una predicción simulada. El servidor limita cada
archivo a 5 MB y solo admite PNG monocromos de 256×256 con nombres terminados
en `_PRE`, `_EARLY` y `_LATE` para el mismo corte.

Para la defensa, desplegar `app.py` en Streamlit Community Cloud o un servicio
equivalente y comprobar la URL desde otro dispositivo. No deben publicarse la
validación privada, sus etiquetas ni rutas locales del sistema.

## Pasos

```bash
.venv/bin/python -m cancer_mama.auditoria
.venv/bin/python -m cancer_mama.eda
.venv/bin/python -m cancer_mama.preparacion --fold 0
.venv/bin/python -B modelos/versionar.py verificar v001_cnn_historica_20260930
```

Los pasos 01-03 y `src/cancer_mama/datos.py` son los originales de Marta. El paso 04
reutiliza su Dataset: convierte PRE/EARLY/LATE a float32 /255, conserva los cinco
folds originales y aplica las transformaciones conjuntamente. Usa la configuración
sin estandarización; las estadísticas del paso 03 siguen siendo descriptivas.
Los CSV fuente no se modifican y el entrenamiento no abre imágenes ni convierte
etiquetas del test reservado.

## Red y entrenamiento

`src/cancer_mama/entrenamiento.py` contiene toda la CNN, el aprendizaje, la comparación OOF y
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
.venv/bin/python -m cancer_mama.entrenamiento entrenar --prueba --folds 0 --semillas 42 --perdidas ponderada --dispositivo cpu --workers 0 --salida resultados/04_entrenamiento/entrenamiento_actual
```

Para una ejecución concreta en la RTX 3090:

```bash
.venv/bin/python -m cancer_mama.entrenamiento entrenar --folds 0 --semillas 42 --perdidas ponderada --dispositivo cuda --salida resultados/04_entrenamiento/entrenamiento_actual
```

Para reproducir las 40 ejecuciones del diseño del informe (dos configuraciones,
dos pérdidas, dos semillas y cinco folds):

```bash
.venv/bin/python -m cancer_mama.entrenamiento entrenar --configuraciones base_raw raw_rot90 --dispositivo cuda --salida resultados/04_entrenamiento/entrenamiento_actual
.venv/bin/python -m cancer_mama.entrenamiento comparar --salida resultados/04_entrenamiento/entrenamiento_actual
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
.venv/bin/python -B modelos/versionar.py predecir v001_cnn_historica_20260930 -- --pre paciente_z000_PRE.png --early paciente_z000_EARLY.png --late paciente_z000_LATE.png
```

Se pueden pasar varios archivos en cada opción, en el mismo orden y de una sola
paciente. Deben ser PNG monocromos de 8 bits y 256×256. Para usar otra versión,
indica su ID después de `predecir`. El [historial de modelos](modelos/README.md)
incluye un ejemplo completo con las variables clínicas del multimodal.

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
| cnn.py | CNN y bloques en `src/cancer_mama/entrenamiento.py` |
| datos.py | `src/cancer_mama/datos.py` y validaciones del entrenamiento |
| entrenar.py | entrenamiento y checkpoints en el paso 04 |
| evaluar.py | comparación OOF, calibración e inferencia en el paso 04 |
| configs/seleccion_final.json | modelo_final.json y historico/seleccion_original.json |

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m unittest discover -s tests -v
.venv/bin/python -B modelos/versionar.py verificar
```

Las pruebas verifican la arquitectura, escala y alineación de fases, separación
por paciente, OOF y reanudación idéntica, sin repetir el test reservado.
El respaldo de la limpieza, con el código, ramas y experimentos previos,
está fuera de la carpeta del proyecto. Para compartir código entre ordenadores,
usa `git pull` antes de trabajar y revisa `git status` antes de commit/push.

## Configuración ajustada conservada

Se mantiene `pool_dropout_wd`: la misma CNN de 551.913 parámetros con MaxPool
entre bloques, LR 0,0008, dropout 0,35 y weight decay 0,001. Conserva los cuatro
bloques, ocho convoluciones y salidas 64/32/16/8. Se inicializa desde cero y
utiliza BatchNorm y BCE por corte. Es una configuración directa del entrenamiento;
no lanza una búsqueda ni compara variantes descartadas.

La selección previa en tres folds obtuvo AUC media 0.6270 frente a 0.6231 de
referencia y F1 0.4770 frente a 0.4336. Fue exploratoria, con una semilla;
no demuestra una mejora general ni sustituye al modelo histórico del informe.
La arquitectura original continúa disponible para reproducir ese diseño y
cargar sus diez pesos.

```bash
.venv/bin/python -B -m cancer_mama.entrenamiento entrenar --configuraciones pool_dropout_wd --perdidas ponderada --folds 0 --semillas 42 --epocas 30 --revision-cada 10 --lote 64 --dispositivo cuda --salida resultados/04_entrenamiento/entrenamiento_pool
```

Cada revisión guarda pérdida, AUC, F1, accuracy y predicciones por paciente.
Las métricas train de las revisiones se calculan sin aumentos ni dropout.
El entrenamiento habitual sigue usando cortes barajados y los folds originales;
no utiliza los lotes de pacientes completas ni el holdout anidado de los ensayos
retirados. Se conserva la reanudación, el calendario coseno y el checkpoint de
mayor AUC de validación. `--hasta-epoca` permite revisar un bloque y continuar
con el mismo presupuesto y RNG; no reinicia el aprendizaje.

Para completar una evaluación OOF de esta configuración, entrenar los cinco
folds con semillas 42/2026 y pérdida ponderada, y después ejecutar `comparar`
con esa misma salida. Para `base_raw` y `raw_rot90` se mantienen ambas pérdidas
del diseño del informe. Esa comparación sigue siendo desarrollo interno y
usa validación para seleccionar checkpoints.

## Investigación y ensayos de mejora

### Candidato con realce temporal explícito

`enhancement_clinical` mantiene como entrada las tres fases originales y deriva
dentro de la red tres canales firmados: `EARLY-PRE`, `LATE-PRE` y
`LATE-EARLY`. Así el modelo ve a la vez anatomía e información dinámica sin
generar nuevos archivos ni ajustar el preprocesamiento con validación. El brazo
`enhancement_regularized_clinical` prueba si Dropout2d y una penalización mayor
reducen la divergencia observada entre las curvas de train y validación.

La comparación usa AUC por paciente; el umbral se decide después con predicciones
OOF. Por tanto, mover el umbral puede reducir falsos positivos o falsos negativos,
pero no modifica el ROC-AUC. No se promete alcanzar 0,80 y no se consulta el test
reservado para escoger entre los dos brazos.

[Investigación de mejoras de la CNN](docs/mejoras_cnn/INVESTIGACION_MEJORAS.md)
recoge la revisión del enunciado, curvas, metadatos y fuentes primarias realizada
el 05/10/2026. Las cuatro propuestas ya se probaron: Dropout2d, weight decay
0,003, aumentos afines suaves y EMA. Se completaron 80 entrenamientos, cinco
folds, semillas 42/2026 y ambas pérdidas, con selección interna independiente
del fold exterior. El [protocolo previo](docs/mejoras_cnn/PROTOCOLO_CUATRO_MEJORAS.md)
y los [resultados completos](docs/mejoras_cnn/RESULTADOS_CUATRO_MEJORAS.md)
registran las condiciones y decisiones.

Con BCE ponderada, Dropout2d dio la mayor señal: AUC media 0,5504 → 0,5729,
pero IC98,75% de la diferencia [−0,0126, +0,0584]. Ninguna propuesta cumplió
todos los criterios fijados de adopción. Los cuatro candidatos quedan
inconcluyentes y fuera del entrenador activo; no se sustituye el modelo histórico.
La evidencia y código congelado están en
`resultados/04_entrenamiento/cuatro_mejoras_20261005/`, separados del paso 04.
Se conservan los pasos 01–03 y la CNN 2D desde cero exigida por la práctica.

## Candidato multimodal clínico

`pool_dropout_wd_clinical` añade a la CNN ajustada cuatro variables disponibles
en `patients.csv`: edad, volumen tumoral, HR y HER2. La red sigue partiendo de
cero y termina en un único logit. La imputación y estandarización se ajustan solo
con las pacientes de aprendizaje de cada fold y se guardan dentro de sus pesos.
No se utilizan cohorte, raza, coordenadas de recorte ni indicadores del split.
La salida clínica se inicializa con una regresión logística ponderada ajustada
en ese mismo train y su rendimiento se conserva como checkpoint de época 0: la
optimización conjunta puede mejorarlo, pero no sustituirlo por un checkpoint peor.

Es un candidato pendiente, no sustituye al modelo histórico. Su uso en la defensa
depende de que el profesorado proporcione esas cuatro variables junto con cada
muestra privada. El
[protocolo multimodal](docs/mejoras_cnn/PROTOCOLO_MULTIMODAL_CLINICO.md) fija
antes del entrenamiento veinte jobs, el test cerrado y AUC media y OOF de al
menos 0,70 como criterios principales.
La [guía de traslado a GPU](docs/GUIA_GPU_MULTIMODAL.md) detalla qué copiar, cómo
instalar CUDA, ejecutar el humo, reanudar los veinte runs y recuperar los pesos.

```powershell
.venv\Scripts\python.exe -B -m cancer_mama.entrenamiento entrenar `
  --configuraciones pool_dropout_wd_clinical `
  --perdidas normal ponderada --semillas 42 2026 --folds 0 1 2 3 4 `
  --epocas 46 --lote 64 --dispositivo cuda `
  --salida resultados/04_entrenamiento/multimodal_clinico_20261007

.venv\Scripts\python.exe -B -m cancer_mama.entrenamiento comparar `
  --salida resultados/04_entrenamiento/multimodal_clinico_20261007
```

### Variante orientada a reducir falsos negativos

`patient_level_clinical` conserva la arquitectura multimodal, pero agrupa todos
los cortes de una paciente dentro del mismo lote y calcula una única BCE por
paciente. De esta forma, la unidad de optimización coincide con la unidad de
evaluación y cada paciente pesa una vez por época. La decisión final maximiza la
especificidad entre los umbrales que mantienen una sensibilidad aparente mínima
del 90 %. El modelo anterior no se sobrescribe.

La variante usa solo BCE ponderada, dos semillas y cinco folds: diez runs.

```powershell
.venv\Scripts\python.exe -B -m cancer_mama.entrenamiento entrenar `
  --configuraciones patient_level_clinical --perdidas ponderada `
  --semillas 42 2026 --folds 0 1 2 3 4 `
  --epocas 46 --lote 64 --workers 4 --dispositivo cuda `
  --salida resultados/04_entrenamiento/patient_level_clinical_20261007

.venv\Scripts\python.exe -B -m cancer_mama.entrenamiento comparar `
  --salida resultados/04_entrenamiento/patient_level_clinical_20261007
```

El 90 % es una restricción de desarrollo, no una garantía clínica. El umbral y
la matriz resultantes deben confirmarse en pacientes independientes.

Los diez entrenamientos del 07/10/2026 están completos. Obtienen AUC media por
fold **0,7155** y AUC OOF **0,7150**, frente a **0,7243** y **0,7246** de v002.
Se mantiene v002 como referencia y se conservan los checkpoints nuevos localmente.
El [informe por paciente](resultados/07_informe_paciente_20261007/INFORME_RESULTADOS.md)
incluye curvas, predicciones OOF, configuraciones y comparación de falsos positivos
al mismo objetivo de sensibilidad. Con los datos y pesos locales se regenera con:

```bash
.venv/bin/python -B -m cancer_mama.informe_paciente
```

Para un manifiesto multimodal, `predecir` recibe además `--edad`,
`--volumen-tumoral`, `--hr` y `--her2`. Una variable individual omitida se trata
como ausente mediante la imputación del fold; omitir las cuatro se rechaza para
evitar una predicción accidental sin la modalidad clínica.

La [justificación cuantitativa](resultados/05_informe_mejora/INFORME_JUSTIFICACION.md)
compara ROC, PR, loss de train/validación, folds, cohortes, calibración y el test
histórico. Se regenera, sin volver a evaluar el candidato en test, con:

```powershell
.venv\Scripts\python.exe -B -m cancer_mama.informe_mejora
```

## Experimentos descartados y limpieza

[experimentos descartados](docs/EXPERIMENTOS_DESCARTADOS.md) registra hipótesis,
protocolos, resultados, límites y decisiones. Los comparadores de tamaño,
composición de lotes, GroupNorm y búsqueda de ajustes se retiraron del código
activo. También se eliminaron los samplers, aumentos pareados, particiones y
calibración anidada específicos de esos ensayos y sus pruebas exclusivas.

El paso 04 ofrece únicamente `verificar`, `entrenar`, `comparar` y `predecir`.
No incorpora GroupNorm. La BCE por paciente está disponible con
`patient_level_clinical`. Las configuraciones de experimentos
retirados se rechazan; no se reinterpretan como entrenamiento habitual.
Las opciones originales del informe y el perfil ajustado conservado siguen
funcionando. El valor por defecto de entrenamiento continúa siendo `raw_rot90`.

Los informes, predicciones, curvas y pesos existentes permanecen como evidencia
en sus carpetas de resultados. La copia anterior del código y documentación
queda en `resultados/04_entrenamiento/limpieza_20261005/codigo_anterior/`, fuera
del código activo, y Git conserva su historial. No se ejecutan ni se reanudan
ensayos descartados desde el archivo actual. Para reconstruir un ensayo histórico
hay que usar su código, configuración y entorno registrados, no mezclarlos con
esta versión. Al cambiar el código, usar una salida nueva para futuros runs.

Los archivos numerados 01–03, `src/cancer_mama/datos.py`, datos fuente, manifiesto,
pesos, calibración y umbral históricos se conservan. La limpieza solo ejecuta
pruebas CPU y de humo; no inicia nuevos ensayos ni repite el test reservado. Sus comprobaciones quedan en
`resultados/04_entrenamiento/limpieza_20261005/`.

La regla de trabajo es conservar en el entrenador las opciones del proyecto y
los cambios adoptados. Un ensayo que no se adopta se retira del código activo
y se documenta en Markdown, conservando su evidencia fuera del entrenador.
