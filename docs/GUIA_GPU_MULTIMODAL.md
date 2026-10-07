# Traslado y entrenamiento multimodal en GPU

## Estado

El entrenador está preparado para ejecutar una configuración, dos pérdidas,
dos semillas y cinco folds: **20 entrenamientos**.

```text
pool_dropout_wd_clinical × (normal, ponderada) × (42, 2026) × (0,1,2,3,4)
```

Se ejecutan secuencialmente. Cada run guarda `last.pt`, `best.pt`,
`inference.pt`, historia, revisiones y predicciones OOF. Si el proceso se
interrumpe, repetir el mismo comando y la misma carpeta de salida reanuda desde
`last.pt`.

## Qué debe llegar al ordenador con GPU

1. El código actual del proyecto. Los cambios multimodales todavía deben viajar
   mediante una copia de la carpeta de trabajo o mediante un commit y `git push`.
   Clonar ahora mismo `main` sin guardar esos cambios recuperaría la versión
   anterior.
2. La carpeta `breastdcedl/` completa: 38.122 archivos y aproximadamente
   1,28 GiB. Está ignorada por Git y no aparece al clonar el repositorio.
3. No copiar ni reutilizar `.venv/`: depende del sistema y de la instalación de
   PyTorch. Hay que crear un entorno nuevo en el ordenador con GPU.
4. Reservar al menos 10 GiB libres para datos, entorno CUDA, checkpoints y resultados.
   Es preferible un SSD local; OneDrive puede bloquear archivos temporales y
   ralentizar la lectura de miles de PNG.

No hace falta copiar las ejecuciones de humo de
`resultados/04_entrenamiento/multimodal_clinico_20261007/`. Están ignoradas y no
son seleccionables. Los checkpoints de trabajo siguen ignorados. Los modelos
seleccionados se archivan con `modelos/versionar.py guardar` en
`modelos/versiones/`, donde sus pesos sí se guardan en Git. El
[historial de modelos](modelos/README.md) explica cómo publicar una versión nueva.

## Preparación del entorno

En PowerShell, dentro del repositorio:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
```

Instalar primero una compilación estable de PyTorch con CUDA usando el comando
que genere el selector oficial para el sistema, GPU y controlador del ordenador:
<https://pytorch.org/get-started/locally/>. Después instalar el resto:

```powershell
.venv\Scripts\python.exe -m pip install -e .
```

Si PyTorch CUDA ya satisface `torch>=2.4,<3`, este último comando no debe
reemplazarlo. Verificarlo explícitamente:

```powershell
nvidia-smi
.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'SIN CUDA')"
```

El tercer valor debe ser `True`. `src/cancer_mama/entrenamiento.py` también se detiene con un
error antes de empezar si se pide `--dispositivo cuda` y CUDA no está disponible.

## Comprobación corta obligatoria

```powershell
.venv\Scripts\python.exe -B -m unittest discover -s tests -v

.venv\Scripts\python.exe -B -m cancer_mama.entrenamiento entrenar `
  --prueba --configuraciones pool_dropout_wd_clinical `
  --perdidas ponderada --semillas 42 --folds 0 `
  --epocas 1 --lote 8 --workers 0 --dispositivo cuda `
  --salida resultados/04_entrenamiento/multimodal_clinico_20261007
```

El humo debe mostrar `Baseline clínico fold-train`, aproximadamente 551.921
parámetros y terminar una época. Su AUC no es una métrica válida porque carga
solo un lote de cada clase.

## Matriz completa

```powershell
.venv\Scripts\python.exe -B -m cancer_mama.entrenamiento entrenar `
  --configuraciones pool_dropout_wd_clinical `
  --perdidas normal ponderada --semillas 42 2026 --folds 0 1 2 3 4 `
  --epocas 46 --lote 64 --workers 4 --dispositivo cuda `
  --salida resultados/04_entrenamiento/multimodal_clinico_20261007
```

Si Windows presenta errores de procesos de carga, repetir con `--workers 0` o
`--workers 2`; no cambia el diseño estadístico. Si falta memoria CUDA, reducir
`--lote` exige usar una carpeta de salida nueva porque cambia la configuración.
No mezclar runs con batch sizes diferentes en el comparador.

Evitar suspensión, reinicios automáticos y sincronización de la carpeta mientras
entrena. Se puede parar con `Ctrl+C`; el último checkpoint de época completa se
conserva.

## Comparación al terminar

Solo cuando los 20 runs estén completos:

```powershell
.venv\Scripts\python.exe -B -m cancer_mama.entrenamiento comparar `
  --salida resultados/04_entrenamiento/multimodal_clinico_20261007
```

El comparador requiere los cinco folds y ambas semillas de las dos pérdidas.
Genera `comparacion/seleccion.json`, `comparacion/comparacion.csv`, las
predicciones OOF y `comparacion/modelo_desarrollo.json`. La adopción exige AUC
media por fold y AUC OOF agrupada de al menos 0,70. No ejecuta ni reabre el test
histórico.

## Archivos que deben volver

Copiar de vuelta la carpeta completa:

```text
resultados/04_entrenamiento/multimodal_clinico_20261007/
```

Incluye los checkpoints de entrenamiento, que no viajan automáticamente con Git.
Con esa carpeta se pueden revisar las curvas, reanudar el entrenamiento y
preparar el ensemble. Después se archiva la versión seleccionada en
`modelos/versiones/` y se hace commit y push; esa versión completa se obtiene
al clonar el repositorio.
