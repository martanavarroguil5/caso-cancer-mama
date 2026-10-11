# Archivo de ensayos descartados · 11/10/2026

Los ensayos de pérdidas/LR, transferencia, características locales, ajuste parcial,
textura, BCE normal y mixup no demostraron superar al multimodal original v002.
Se cierran como fracasos del objetivo de mejora. V002 vuelve a ser el modelo activo.

Se trasladan aquí, sin modificar sus bytes, doce módulos experimentales, ocho
archivos de pruebas, las dependencias opcionales de transferencia y la comprobación
específica de la aplicación v004. `inventario.json` conserva rutas originales y SHA-256.
Estos archivos son históricos: no pertenecen al paquete instalado ni a las pruebas
activas. Los imports relativos y las rutas reflejan su ubicación original.

## Reproducción histórica

El commit `8c0600c` contiene el proyecto antes de esta limpieza. El commit `41428a2`
contiene las fuentes anteriores al cambio de interfaz/defaults de v004, apropiadas
para revisar la repetición BCE con los hashes originales. Cada ensayo exige el
commit, configuración, datos y entorno registrados en su propio protocolo.
Para reproducirlo se necesita una copia separada de ese commit y los datos/artefactos
locales; no ejecutar directamente estos archivos archivados contra el paquete activo.
Los checkpoints de reanudación siguen en `resultados/04_entrenamiento/` y no se borran.

Las versiones congeladas de [modelos](../../modelos/README.md) conservan pesos y
código compatible para inferencia, incluida v004 como experimento descartado.
No se modifican sus archivos ni sus checksums.

## Decisión y evidencia

- [Registro de experimentos descartados](../../docs/EXPERIMENTOS_DESCARTADOS.md).
- [Pérdida y LR](../../resultados/08_informe_loss_20261010/INFORME_RESULTADOS.md).
- [Transferencia y textura](../../resultados/09_informe_transferencia_20261010/INFORME_RESULTADOS.md).
- [BCE normal y mixup](../../resultados/10_informe_bce_mixup_20261011/INFORME_RESULTADOS.md).
- [Repetición BCE](../../resultados/11_confirmacion_bce_normal_20261011/INFORME_RESULTADOS.md).

La mejora contra un control reentrenado con selección interna sigue siendo el
resultado de ese experimento; no demuestra superioridad sobre el v002 original.
Las arquitecturas preentrenadas, además, están prohibidas por el enunciado docente.
Los informes originales conservan sus resultados y llevan una nota de cierre.
