# Historial de versiones de los modelos

**Modelo activo: `v002_multimodal_clinico_20261007`.** Se restaura el 11/10/2026
por decisión del usuario, tras no demostrar los ensayos posteriores una mejora
sobre el multimodal original. La aplicación y la CLI cargan sus diez pesos,
calibración Platt, mediana y umbral 0,3543331693. No hace falta volver a entrenar.

## Versiones conservadas

| Versión | Estado actual | ROC-AUC OOF cruda | Pesos |
|---|---|---:|---|
| `v001_cnn_historica_20260930` | Histórica, solo imágenes; test original conservado | 0,5945 | 10 archivos |
| `v002_multimodal_clinico_20261007` | **Activa, multimodal original** | **0,7246** | 10 archivos |
| `v003_multimodal_loss_20261010` | Experimento descartado; mejora no concluyente | 0,7061 | 10 archivos |
| `v004_multimodal_bce_normal_20261011` | Experimento descartado; no supera al original | 0,7109 | 10 archivos |

Los protocolos de selección difieren: estas AUC describen cada desarrollo y
no constituyen una comparación independiente entre modelos. El cambio de decisión
no modifica las versiones congeladas. Su `status` interno y `catalogo.json`
conservan el estado en el momento de archivado; **no designan la versión activa**.
La selección de aplicación/CLI está en `src/cancer_mama/paths.py`.

V004 mejoró la pérdida de un control reentrenado con separación interna, pero
ese resultado no demuestra que mejore v002. Con las calibraciones y umbrales
guardados, el original obtiene log-loss 0,5367 y 187 FP/128 FN, frente a 0,5463
y 210 FP/126 FN de v004. Son métricas aparentes de desarrollo.
La [decisión de cierre](../docs/EXPERIMENTOS_DESCARTADOS.md) documenta la corrección,
las comparaciones y sus límites. El test reservado no se abrió para elegir.

## Paquetes congelados

Cada carpeta conserva pesos, manifiesto, preprocesado, calibración, código
compatible, entorno y evidencia, con SHA-256. No se permite sobrescribir una
versión ni reutilizar su número. V001, v002, v003 y v004 permanecen intactas.

```text
versiones/v002_multimodal_clinico_20261007/
├── version.json
├── version.json.sha256
├── manifiesto.json
├── manifiesto.json.sha256
├── pesos/
├── codigo/
├── entorno.json
└── evidencia/
```

## Listar y comprobar

```bash
.venv/bin/python -B modelos/versionar.py listar
.venv/bin/python -B modelos/versionar.py verificar
.venv/bin/python -B -m cancer_mama.entrenamiento verificar
```

El archivador verifica todas las versiones con su código compatible. La última
orden comprueba únicamente el modelo activo v002. Ninguna entrena ni evalúa test.
`verificar --solo-hashes` permite revisar integridad sin cargar PyTorch.

## Inferir con una versión concreta

```bash
.venv/bin/python -B modelos/versionar.py predecir v002_multimodal_clinico_20261007 -- \
  --pre paciente_z000_PRE.png \
  --early paciente_z000_EARLY.png \
  --late paciente_z000_LATE.png \
  --edad 49 --volumen-tumoral 12 --hr 1 --her2 0
```

Cambiar el identificador permite revisar otra versión sin activarla. Para v001
se omiten las variables clínicas. Las fases son PNG monocromos de 8 bits,
256 × 256, de la misma paciente; pueden pasarse varios cortes alineados.
Cada versión usa su código congelado aunque se retiren módulos experimentales
del paquete principal. El dataset no es necesario para inferir nuevas imágenes.

## Archivar una futura actualización

Solo después de justificar la selección frente a la referencia adecuada:

```bash
.venv/bin/python -B modelos/versionar.py guardar \
  --id v005_multimodal_clinico_AAAAMMDD \
  --manifest resultados/04_entrenamiento/nueva_ejecucion/comparacion/modelo_desarrollo.json \
  --descripcion "Descripción de la actualización y su evaluación"
.venv/bin/python -B modelos/versionar.py verificar
```

Sustituir las rutas y la fecha del ejemplo por las reales. `--evidencia` añade
informes. El archivador copia y verifica diez pesos, fuentes y evidencia antes
de publicar la carpeta; archivar no cambia automáticamente el modelo activo.
Los estados de optimizador/RNG y `last.pt` permanecen en las ejecuciones locales;
no se incluyen en el paquete de inferencia ni se confirman como runs pesados.
Los pesos congelados se añaden explícitamente a Git porque `*.pt` se ignora
por defecto. El catálogo puede reconstruirse con `versionar.py catalogar`.

En Windows, sustituir `.venv/bin/python` por `.venv/Scripts/python.exe`.
