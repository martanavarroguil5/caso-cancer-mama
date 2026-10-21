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
| `v005_cnn_imagen_fases_normal_20261014` | Solo imagen, BCE normal; resultado insuficiente conservado | 0,5087 | 10 archivos |
| `v006_cnn_imagen_fases_ponderada_20261014` | Solo imagen, BCE ponderada; resultado insuficiente conservado | 0,5246 | 10 archivos |
| `v007_cnn_cinetica_multiescala_20261021` | Solo imagen, ajuste anidado; resultado insuficiente conservado | 0,5264 | 10 archivos |

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

## CNN por fases del 14/10/2026

V005/v006 contienen los veinte pesos elegidos del ensayo de imagen, separados
por pérdida. La media de AUC por fold es 0,5330 ± 0,0253 y 0,5486 ± 0,0504;
la tabla anterior muestra AUC global OOF, una agregación distinta.
El [informe](../resultados/14_informe_cnn_imagen_20261014/INFORME_RESULTADOS.md)
recoge las predicciones y curvas de las veinte ejecuciones.

Se archivan como `ensayo_insuficiente_conservado`, sin promoción. La calibración
final de v006 tiene pendiente 0 y umbral 1: siempre predice no pCR. V005 no ganó
la selección de pérdida y se guarda con calibración identidad y umbral crudo
0,5 solo para diagnóstico, sin ajustar una nueva política tras ver el resultado.
Estos umbrales del paquete no son los umbrales internos por fold del informe.

## CNN cinética multiescala del 21/10/2026

V007 conserva los diez pesos elegidos mediante validación interna entre tres
pares LR/weight decay, BCE normal/ponderada y dos semillas por fold. Las
60 ejecuciones mantienen exactamente la red de 593.521 parámetros de `39bd951`.
El AUC exterior global es 0,5264; la media por fold, 0,5766 ± 0,0301; el AUC
global con calibración interna por fold, 0,5618. No alcanza el mínimo 0,60 ni
el objetivo 0,70. La referencia de la misma red obtiene 0,5360 global.

El [informe completo](../resultados/15_informe_cinetico_20261021/INFORME_RESULTADOS.md)
explica los intervalos, el sobreajuste, las comparaciones y los errores. La
versión guarda el candidato prefijado como `ensayo_insuficiente_conservado`,
sin promoción. La política global para una posible prueba futura tiene Platt
con pendiente 0,0982743656 y umbral 0,3087942921. No se confunde con los umbrales
internos por fold usados para obtener 74 FP, 281 FN y 41 TP en el informe.

## Paquetes congelados

Cada carpeta conserva pesos, manifiesto, preprocesado, calibración, código
compatible, entorno y evidencia, con SHA-256. No se permite sobrescribir una
versión ni reutilizar su número. Las versiones publicadas son inmutables;
archivar v007 no modifica v001–v006.

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

## Compatibilidad de las CNN solo imagen

El archivador distingue el multimodal de los manifiestos con
`model_class: CNNImagenFases` o `CNNMultiescalaCinetica`. Para estos conserva
y ejecuta `cancer_mama.inferencia_imagen`, junto con la arquitectura de
`entrenamiento_imagen.py` o `entrenamiento_cinetico.py`, respectivamente.
Comprueba la clase, la configuración y, en la cinética, los recortes y mapas
firmados; no intenta cargarlos como el multimodal.
La verificación comprueba diez pesos, sus configuraciones y todos los hashes.
La inferencia usa media de modelos y cortes, conserva exactamente la calibración
y el umbral archivados y no necesita dataset ni variables clínicas.
Una calibración constante se identifica expresamente en la salida y se conserva
como evidencia del resultado, sin modificarla al archivar.

## Archivar una futura actualización

Solo después de justificar la selección frente a la referencia adecuada:

```bash
.venv/bin/python -B modelos/versionar.py guardar \
  --id v008_multimodal_clinico_AAAAMMDD \
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
