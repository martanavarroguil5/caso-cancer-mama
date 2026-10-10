# Guía GPU: base multimodal

Actualizado el 10/10/2026. La base activa es `pool_dropout_wd_clinical` con
BCE ponderada por corte. **v002 ya está entrenado**: para usarlo basta instalar
las dependencias y cargar sus pesos. El ensayo de realce temporal terminó y
queda [descartado](EXPERIMENTOS_DESCARTADOS.md).

## Preparar otro ordenador

Clonar la versión actual del repositorio y crear un entorno propio. En Windows:

```powershell
py -3.12 -m venv .venv
```

Instalar primero PyTorch CUDA con el comando del [selector oficial](https://pytorch.org/get-started/locally/)
para el ordenador y después el paquete:

```powershell
.venv/Scripts/python.exe -m pip install -e .
.venv/Scripts/python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
.venv/Scripts/python.exe -B -m cancer_mama.entrenamiento verificar
```

`git clone` incluye v002 y sus diez pesos. El dataset y checkpoints de trabajo
se copian aparte si se va a entrenar; no copiar `.venv` de otra máquina.
Los datos preparados deben quedar en `breastdcedl/dataset/` y
`breastdcedl/metadata/`, incluyendo `samples.csv` y las cuatro variables clínicas
en `patients.csv` o en los metadatos de muestras.

## Comprobar la GPU sin ejecutar la matriz

```powershell
powershell -ExecutionPolicy Bypass -File scripts/entrenar_multimodal_gpu.ps1 -SoloHumo
```

El lanzador verifica CUDA, ejecuta los tests y hace una prueba de un lote y una
época. Esa prueba se guarda en `pruebas/` y se excluye de comparación.
`-OmitirTests` permite omitir solo los tests, no la comprobación de CUDA ni el humo.

## Nuevo entrenamiento, cuando se decida

```powershell
powershell -ExecutionPolicy Bypass -File scripts/entrenar_multimodal_gpu.ps1
```

Son diez runs reanudables: semillas 42 y 2026 × folds 0–4. El único perfil
usa PRE/EARLY/LATE, edad, volumen tumoral, HR y HER2; pooling intermedio,
dropout 0,35, weight decay 0,001, LR 0,0008, lote 64 y hasta 46 épocas.
Después se evalúan OOF con mediana entre cortes, calibración Platt y umbral Youden.
Esa evaluación sigue siendo desarrollo interno y no utiliza imágenes de test.

La salida por defecto es `resultados/04_entrenamiento/multimodal_actual`.
Puede cambiarse con `-Salida`; también existen `-Datos`, `-Workers`, `-Lote`
y `-Epocas`. No sobrescribe ni promueve automáticamente la versión v002.

En Linux, desde la raíz:

```bash
.venv/bin/python -B -m unittest discover -s tests -v
.venv/bin/python -B -m cancer_mama.entrenamiento entrenar --prueba \
  --semillas 42 --folds 0 --epocas 1 --workers 0 --lote 8 --dispositivo cuda
.venv/bin/python -B -m cancer_mama.entrenamiento entrenar \
  --dispositivo cuda --salida resultados/04_entrenamiento/multimodal_actual
.venv/bin/python -B -m cancer_mama.entrenamiento comparar \
  --salida resultados/04_entrenamiento/multimodal_actual
```

Para continuar un run interrumpido, repetir el comando con el mismo código,
datos, presupuesto y salida. Para revisar un bloque, añadir `--hasta-epoca N`;
posteriormente continuar con la misma configuración. Si cambia el código,
usar otra salida: los hashes impiden reinterpretar checkpoints anteriores.

Copiar la carpeta completa de la ejecución para conservar estados de optimizador,
RNG, revisiones, OOF y curvas. Una futura versión aceptada se archiva mediante
[modelos/versionar.py](../modelos/README.md), conservando las anteriores.
