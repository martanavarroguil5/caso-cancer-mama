# Historial de versiones de los modelos

La base activa es **v002_multimodal_clinico_20261007**, ya entrenada. El módulo
actual de entrenamiento y la aplicación la usan por defecto. No hace falta
reentrenar para cargar sus diez pesos. El experimento de realce temporal queda
descartado en [EXPERIMENTOS_DESCARTADOS.md](../docs/EXPERIMENTOS_DESCARTADOS.md).

Cada versión guarda un ensemble completo, listo para predecir desde otro
ordenador después de clonar el repositorio e instalar sus dependencias. Los
pesos, la calibración, el preprocesado y el código compatible quedan juntos.
La herramienta de archivado comprueba las versiones anteriores y rechaza
sobrescribir una carpeta o reutilizar su número de versión.

## Versiones conservadas

| Versión | Entrada | Estado | AUC OOF cruda | Pesos |
| --- | --- | --- | --- | --- |
| `v001_cnn_historica_20260930` | PRE, EARLY y LATE | Histórico; test original conservado | 0,5945 | 10 archivos; 21,25 MiB |
| `v002_multimodal_clinico_20261007` | Imágenes, edad, volumen tumoral, HR y HER2 | Desarrollo; sin evaluación nueva de test | 0,7246 | 10 archivos; 21,27 MiB |

La segunda versión usa BCE ponderada y mediana de cortes por paciente. La
disponibilidad de las variables clínicas en la prueba privada es un supuesto
de trabajo. Las métricas OOF proceden de desarrollo interno y no equivalen al
rendimiento de la prueba privada.

`catalogo.json` resume las versiones. Cada carpeta contiene:

```text
versiones/v002_multimodal_clinico_20261007/
├── version.json                    # Identidad, estado, métricas y hashes
├── version.json.sha256
├── manifiesto.json                 # Rutas portables, configuración y calibración
├── manifiesto.json.sha256
├── pesos/                          # Dos semillas y cinco folds
├── codigo/                         # Copia compatible del entrenamiento y sus módulos
├── entorno.json                    # Python y paquetes usados para comprobarla
└── evidencia/                      # Manifiesto original, configuraciones y resultados
```

El manifiesto operativo usa únicamente archivos de su propia versión. La copia
del manifiesto de origen se conserva como evidencia con sus rutas originales.
El código incluido es el que se comprobó al archivar; el entorno registrado
describe esa comprobación y no afirma ser el entorno original del entrenamiento
histórico. Git conserva los bytes de las carpetas de versiones, también en
Windows, para mantener los checksums.

## Listar y comprobar

Desde la raíz del repositorio:

```bash
.venv/bin/python -B modelos/versionar.py listar
.venv/bin/python -B modelos/versionar.py verificar
```

La comprobación revisa todos los archivos y carga los diez pesos de cada versión
con su copia de la arquitectura. No vuelve a entrenar ni evalúa imágenes de
test. Para revisar solo integridad, sin instalar PyTorch:

```bash
python modelos/versionar.py verificar --solo-hashes
```

En Windows, sustituye `.venv/bin/python` por `.venv\Scripts\python.exe`. Las
dependencias de cada versión están en su carpeta `codigo/requirements.txt`;
`entorno.json` conserva las versiones exactas de los paquetes con los que se
verificó aquí. La inferencia puede ejecutarse en CPU.

## Predecir con una versión concreta

Para el histórico:

```bash
.venv/bin/python -B modelos/versionar.py predecir v001_cnn_historica_20260930 -- \
  --pre paciente_z000_PRE.png \
  --early paciente_z000_EARLY.png \
  --late paciente_z000_LATE.png
```

Para el multimodal, con valores clínicos de ejemplo:

```bash
.venv/bin/python -B modelos/versionar.py predecir v002_multimodal_clinico_20261007 -- \
  --pre paciente_z000_PRE.png \
  --early paciente_z000_EARLY.png \
  --late paciente_z000_LATE.png \
  --edad 49 --volumen-tumoral 12 --hr 1 --her2 0
```

Se admiten varios cortes de la misma paciente, en el mismo orden en las tres
fases. Deben ser PNG monocromos de 8 bits, de 256 × 256. No se necesita descargar
el dataset de entrenamiento para utilizar el modelo con nuevas imágenes.
La herramienta verifica la versión y usa su código conservado aunque el módulo
de entrenamiento cambie en el futuro. No hay un alias que sustituya automáticamente
el modelo histórico en este archivador: aquí se elige la versión de forma explícita.

## Guardar una actualización sin perder las anteriores

Después de completar el nuevo entrenamiento y ejecutar su comparación, archiva
el manifiesto elegido con el siguiente número disponible:

```bash
.venv/bin/python -B modelos/versionar.py guardar \
  --id v003_multimodal_clinico_20261008 \
  --manifest resultados/04_entrenamiento/nueva_ejecucion/comparacion/modelo_desarrollo.json \
  --descripcion "Nueva actualización del modelo multimodal"

.venv/bin/python -B modelos/versionar.py verificar
git add modelos
git commit -m "Conserva la tercera versión del modelo"
git push origin main
```

Las rutas y la fecha del ejemplo se sustituyen por las de la ejecución real.
`--evidencia archivo1 archivo2` permite añadir métricas o informes al paquete.
Las configuraciones, resúmenes, curvas numéricas y entorno disponibles junto a
los pesos de origen se incorporan automáticamente. La publicación de la carpeta
solo ocurre después de comprobar todas las copias y cargar los diez modelos.
El catálogo puede reconstruirse con `python modelos/versionar.py catalogar`.
La herramienta admite tanto las copias antiguas de `04_entrenamiento.py` y
`pipeline_datos.py` como la estructura actual `src/cancer_mama/`. Las versiones
ya archivadas mantienen su estructura original; una nueva versión incorpora
una copia del paquete actual para seguir funcionando de forma independiente.

Se guardan los pesos de inferencia y lo necesario para recuperar sus predicciones.
Los checkpoints `last.pt` con optimizador y RNG, las revisiones intermedias y el
dataset de entrenamiento siguen en las carpetas locales de cada ejecución. Para
reanudar exactamente un entrenamiento interrumpido se conserva también esa
carpeta local; el paquete de una versión no sustituye ese estado de entrenamiento.

Cada peso archivado ocupa aproximadamente 2,13 MiB, por lo que estas versiones
se guardan directamente en Git. Si las versiones futuras crecen, puede adoptarse
Git LFS; GitHub bloquea archivos individuales mayores de 100 MiB, según sus
[límites de archivos](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
