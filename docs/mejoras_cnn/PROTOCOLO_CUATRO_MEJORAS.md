# Protocolo de los cuatro ensayos de generalización

Fijado el 05/10/2026 antes de ejecutar la matriz completa. La comparación usa exclusivamente las 1.097 pacientes de desarrollo. Test histórico y validación privada del profesor permanecen cerrados. Los archivos 01–03, pipeline de datos y entrenador activo se conservan.

## Brazos y presupuesto

Referencia: `pool_dropout_wd`, ocho convoluciones, canales 24/48/96/160, cabeza 64, 551.913 parámetros, BatchNorm, dropout de cabeza 0,35, AdamW, LR 0,0008, weight decay 0,001, lote 64 y espejo/rotaciones de 90° conjuntos. Máximo 30 épocas, coseno hasta 0,00001, mínimo 12 épocas, paciencia 10 y min_delta 0,001. AMP FP16 solo en aprendizaje; selección y evaluación en FP32 estricto, TF32 desactivado y algoritmos deterministas.

| Brazo | Cambio frente a referencia |
|---|---|
| control | Ninguno |
| dropout2d | Dropout2d p=0,10 tras el cuarto bloque y antes de pooling global |
| weight_decay | Weight decay 0,003 |
| afines | Añadir traslación conjunta ±3% y escala 0,95–1,05, interpolación bilineal y padding cero |
| ema | EMA de parámetros con decaimiento 0,99 tras cada paso efectivo del optimizador; recalcular BatchNorm con fit, sin aumentos ni dropout |
| control_bn | Diagnóstico secundario: control con la misma recalculación de BatchNorm que EMA |

Cada brazo se evalúa con cinco folds originales, semillas 42 y 2026, y BCE normal y ponderada. Hay 100 evaluaciones principales de modelos y 20 diagnósticos de BatchNorm. EMA y su control de BatchNorm comparten la trayectoria de aprendizaje de la referencia: se necesitan 80 jobs de entrenamiento, sin duplicar optimización para calcular EMA.

Cada una de esas tres salidas tiene su propio checkpoint y su propia parada, elegidos exclusivamente con selección interna. La trayectoria compartida continúa hasta que todas hayan parado o hasta la época 30; un brazo que ya haya parado no vuelve a seleccionar épocas. Se registra tanto su presupuesto lógico como las épocas físicas calculadas. No se promedian indiscriminadamente modelos de épocas finales ya sobreajustados.

Los jobs se ejecutan en dos procesos aislados sobre la RTX 3090. Las configuraciones registran código, datos, entorno, RNG, partición y pesos iniciales. Los procesos reanudan desde checkpoints completos de época. La implementación se conserva en el directorio de evidencia, separado del entrenador activo.

## Separación y comparación

El fold exterior se reserva para evaluación. Dentro de los cuatro folds restantes se separa 85% fit / 15% selección, por paciente y estratificación de cohorte × pCR, semilla fija 73421+fold. La partición interna no cambia entre los brazos ni las dos semillas. Aproximadamente 68% del desarrollo ajusta pesos. La ponderación BCE cuenta únicamente los cortes de fit.

La inicialización, orden de datos y decisiones de espejo/rotaciones se parean. Los aumentos afines usan otro RNG para conservar la secuencia de aumentos originales. Dropout2d añade su estocasticidad de entrenamiento. El promedio EMA se actualiza únicamente cuando GradScaler ejecuta el paso del optimizador. La selección se hace por AUC cruda de paciente, promediando probabilidades de sus cortes.

El fold exterior no elige checkpoint, umbral ni calibración. Sus predicciones se generan después de finalizar el entrenamiento. La agregación principal promedia primero probabilidades de ambas semillas por corte y después por paciente; usa los mismos cortes en todas las variantes.

## Criterios fijados antes de ver resultados

El criterio principal es la diferencia frente al control en la AUC cruda media de los cinco folds del ensemble de dos semillas, con BCE ponderada. Para adoptar un candidato se exige:

- Diferencia de AUC de al menos +0,01.
- Mejora en al menos cuatro de cinco folds.
- Diferencia de AUC media positiva con cada semilla por separado.
- AP media de folds sin empeoramiento.
- Límite inferior del intervalo pareado simultáneo mayor que cero.

Se calculan 4.000 bootstrap de pacientes, estratificados por fold/cohorte/clase, semilla 20261005, compartiendo remuestreos entre brazos. Se presentan IC95% y, para las cuatro comparaciones principales, intervalos de 98,75% con corrección Bonferroni. Condicionan a los modelos y predicciones: no capturan toda la incertidumbre de entrenamiento, ni hacen independientes los folds que comparten datos de aprendizaje.

BCE normal, AUC OOF agrupada, AP, Brier, F1, sensibilidad y especificidad a 0,5 son secundarios. El diagnóstico `control_bn` permite comprobar cuánto de una mejora de EMA coincide con recalcular BatchNorm; no forma parte de la búsqueda principal de cuatro candidatos. No se promueve una variante únicamente por mejorar el umbral o un solo fold.

No habrá selección por resultados preliminares de los primeros folds: se completan los cuatro candidatos y las dos pérdidas. No se combinan cambios antes de terminar estas comparaciones. Las 1.097 pacientes ya se han observado en el desarrollo previo; este procedimiento continúa siendo evaluación interna adaptativa, no validación externa.

## Verificación y conservación

Antes de la matriz se comprobaron con datos sintéticos las particiones, alineación de fases, pesos iniciales, ecuación EMA, estadísticas BatchNorm, reanudación y reproducción numérica del control. Se ejecutó una prueba de humo CUDA de cada ruta. Los aumentos afines se inspeccionan en seis pacientes train de las tres cohortes; sin máscaras de tumor en PNG esa inspección no certifica cobertura completa de toda lesión.

Se archivan resultados, configuraciones, historias, particiones, predicciones exteriores, checkpoints y código de cada ensayo. Un candidato que empeore o no cumpla el criterio queda documentado como descartado o inconcluyente y se retira del código activo. El modelo, calibración, umbral y resultados históricos permanecen intactos.
