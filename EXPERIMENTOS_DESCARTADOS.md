# Experimentos cerrados y retirados del entrenador

Actualizado el 05/10/2026. Estos ensayos no forman parte del código activo de
`04_entrenamiento.py` y sus comandos de comparación se han eliminado. No se
vuelven a ejecutar automáticamente. Se conservan sus informes, predicciones,
curvas, configuraciones y evidencia histórica en `resultados/04_entrenamiento/`.
El código y README anteriores a esta limpieza quedan en
`limpieza_20261005/codigo_anterior/`; los commits citados identifican cada ensayo.

## Decisiones

| Ensayo | Resultado observado | Decisión en el código actual |
|---|---|---|
| Búsqueda de seis ajustes | Se seleccionó `pool_dropout_wd`; confirmación exploratoria en tres folds | Conservar solo ese perfil y el diseño original del informe; retirar la búsqueda y sus otras configuraciones |
| CNN más pequeña | Sin mejora global de AUC; mejor F1 a cambio de especificidad y peor Brier | Retirar el comparador y la configuración específica pequeña |
| BCE por paciente | AUC media 0.5590 → 0.5404 | Variante retirada antes de esta limpieza; retirar ahora también sus auxiliares de bolsas y evaluación anidada |
| Cortes mezclados en lotes pareados | AUC media 0.5546 → 0.5213 | Retirar el ensayo, samplers y aumentos pareados; conservar el loader habitual original |
| GroupNorm con ocho grupos | AUC media 0.5546 → 0.4801 | Retirar GroupNorm, su configuración y comparador; conservar BatchNorm |

Las decisiones son específicas de este proyecto y las condiciones ensayadas.
No prueban que esos métodos sean inferiores con cualquier dato o hiperparámetro.
La limpieza retira código; no altera los resultados ni promueve otro modelo.

## Ajustes de pooling e hiperparámetros: búsqueda cerrada

El 03/10/2026 se probaron seis combinaciones con fold 0, semilla 42, BCE
ponderada, rotaciones, lote 16 y treinta épocas. Se seleccionó
`pool_dropout_wd`: MaxPool entre bloques, LR 0.0008, dropout 0.35 y weight
decay 0.001, con 551.913 parámetros. La referencia usó la arquitectura original.
La configuración seleccionada se comprobó también en folds 1 y 2.

| Fold | AUC original | AUC ajustada | F1 original | F1 ajustada |
|---|---:|---:|---:|---:|
| 0 | 0.6292 | 0.6498 | 0.4459 | 0.5114 |
| 1 | 0.5748 | 0.5903 | 0.3740 | 0.4683 |
| 2 | 0.6653 | 0.6408 | 0.4810 | 0.4512 |
| Media | 0.6231 | 0.6270 | 0.4336 | 0.4770 |

Es una elección exploratoria: tres folds, una semilla y selección usando
validación. No demuestra mejora general. El perfil elegido sigue disponible
como configuración normal de entrenamiento; se retiró `ajustar`, el listado
de seis candidatos y su comparador. Evidencia: `ajustes/resultado_ajuste.json`,
`resultado_confirmacion.json`, revisiones, curvas y predicciones.

## Red más pequeña: no adoptada (04/10/2026)

Código del ensayo: `726b704`. Dos tamaños × cinco folds × semillas 42/2026:
veinte runs de treinta épocas, lote 64 y presupuesto común. Solo cambiaban los
canales: 24/48/96/160 (551.913 parámetros) frente a 16/32/64/128 (310.513).

AUC media por run: 0.6115 / 0.5954 (actual / pequeña). AUC OOF agrupada:
0.5932 / 0.5922; diferencia −0.0010, IC95% condicionado [−0.0372, +0.0346].
F1 OOF mejoró 0.3571 → 0.4252, con sensibilidad 0.3261 → 0.4720 y
especificidad 0.7923 → 0.6890. Brier empeoró 0.2297 → 0.2415. El ahorro de
parámetros y el mayor F1 no demostraron una mejora global de generalización.

Se retiraron `comparar-tamano`, su configuración pequeña, bootstrap y gráficos
específicos. Evidencia y pesos: `comparacion_tamano/`. La selección del
checkpoint usaba el mismo fold de validación que las métricas OOF; no fue
validación anidada. No se compara directamente con los ensayos posteriores.

## BCE por paciente: descartada (04/10/2026)

Código: `b549915`. Veinte runs y 600 épocas. Se comparó BCE ponderada por corte
con BCE después de promediar las probabilidades de los cortes de una paciente.
Ambos brazos utilizaban lotes completos de seis pacientes y la misma CNN.

AUC media de folds del ensemble de dos semillas: 0.5590 / 0.5404; diferencia
−0.0186, IC95% condicionado [−0.0544, +0.0149]. AUC OOF agrupada cruda:
0.5480 / 0.5011; F1 a 0.5: 0.3943 / 0.3507; sensibilidad: 0.5155 / 0.4286.
El intervalo principal incluía cero, pero el ensayo no respaldó el cambio.

`BCEPaciente`, la selección de esa pérdida y `generalizar` se retiraron antes
de esta limpieza. Se habían eliminado los 60 pesos/checkpoints del candidato,
con registro en `retirada_variante.json`; sus resultados y los pesos de
referencia se conservaron. Esta limpieza no elimina más pesos. Se retiran ahora
las bolsas completas y auxiliares anidados que permanecían en el entrenador.
Evidencia: `generalizacion_paciente/`.

## Composición de lotes: cambio no adoptado (04/10/2026)

Código: `b40ac95`. Veinte runs y 600 épocas. Se compararon bolsas de seis
pacientes con cortes mezclados usando las mismas longitudes, cortes por época,
pasos efectivos del optimizador, parámetros iniciales y aumentos por muestra.
BCE ponderada por corte, CNN ajustada, LR 0.0008, dropout 0.35 y weight decay 0.001.

AUC media de folds: 0.5546 / 0.5213 (completas / mezclados); diferencia −0.0332,
IC95% condicionado [−0.0723, +0.0042]. AUC OOF agrupada: 0.5545 / 0.5128.
F1 a 0.5: 0.4190 / 0.3407; sensibilidad: 0.6025 / 0.3602; especificidad:
0.4710 / 0.6865. AUC train/selección en época 30: 0.6642/0.5380 frente a
0.9510/0.4979. La mezcla en ese protocolo produjo mayor sobreajuste.

Se retiran `comparar-lotes`, el sampler pareado y los aumentos asignados por
hash. El loader habitual del proyecto ya barajaba cortes: se conserva ese
comportamiento original, sin convertir el ensayo en un cambio del entrenamiento.
Evidencia y pesos de ambos brazos: `composicion_lotes/`.

## GroupNorm: descartada con esta configuración (05/10/2026)

Código: `4321b0b`. Veinte runs y 600 épocas. Solo cambiaron las ocho capas
BatchNorm por GroupNorm de ocho grupos, epsilon 1e-5 y transformación afín.
Ambos brazos tenían 551.913 parámetros y lotes completos de seis pacientes.
Los diez controles BatchNorm reprodujeron exactamente los pesos seleccionados
y predicciones del control del ensayo anterior.

AUC media de folds: 0.5546 / 0.4801 (BatchNorm / GroupNorm); diferencia −0.0744,
IC95% condicionado [−0.1232, −0.0260]. AP media de folds: 0.3505 / 0.2890.
AUC OOF agrupada: 0.5545 / 0.4702; F1 a 0.5: 0.4190 / 0.4120; sensibilidad:
0.6025 / 0.7671; especificidad: 0.4710 / 0.1871. GroupNorm mejoró AUC solo en
uno de cinco folds. La mayor sensibilidad produjo 220 falsos positivos más.

En nueve de diez modelos GroupNorm de época 30, el diagnóstico posterior
mostró predicciones constantes y las 64 unidades ReLU de la cabeza inactivas
para todos los cortes de selección. Se evaluaron los mejores checkpoints,
no necesariamente los finales. Esto indica falta de aprendizaje con esos
hiperparámetros y no demuestra que GroupNorm falle en cualquier configuración.

El entrenamiento/selección desactivaba TF32 y fijaba determinismo; el
coordinador de evaluación paralela mantenía los defaults de PyTorch. La auditoría
posterior con los mismos modelos congelados y FP32 estricto obtuvo AUC media
0.5546 / 0.4798, diferencia −0.0748: mantuvo la conclusión. No se ajustaron
modelos ni se sustituyó el resultado principal usando esa auditoría.

Se retiran GroupNorm y `comparar-normalizacion`, junto con el protocolo anidado
y sus coordinadores de entrenamiento, calibración y evaluación. Evidencia y
pesos de ambos brazos: `normalizacion/`, incluido `informe_groupnorm.txt`,
`diagnostico_activaciones.csv` y `auditoria_precision.json`.

## Límites de los ensayos por paciente, lotes y normalización

Cinco folds originales, semillas 42/2026 y separación interna 70/15/15 por
paciente y cohorte × pCR: fit, selección y calibración. Solo selección elegía
el checkpoint. Platt no negativo con C=1 y Youden se fijaban internamente antes
de inferir los folds exteriores. Criterio principal: AUC cruda media de los
cinco folds del ensemble de dos semillas. OOF agrupada y métricas de umbral
eran secundarias. Se aprende con aproximadamente el 56% del desarrollo.

Las 1.097 pacientes de desarrollo ya se habían observado. Los intervalos de
2.000 bootstrap pareados por paciente, estratificados por fold/cohorte/clase,
condicionan a los modelos y predicciones. No incluyen toda la incertidumbre del
entrenamiento o de la selección adaptativa. Los folds comparten aprendizaje;
no equivalen a cinco experimentos independientes ni a pacientes nuevos externos.
No se volvió a abrir el test reservado de 176 pacientes ya evaluado el 30/09/2026.

## Alcance de la limpieza

Se eliminan los drivers y ramas experimentales del entrenador y sus pruebas
exclusivas. Se mantienen las pruebas de arquitectura, fases, pacientes, OOF,
calibración, caché, reanudación e inferencia. El código activo conserva el diseño
del informe, BatchNorm, BCE por corte, pooling aceptado y las revisiones por época.

No se modifican datos, archivos 01–03, `pipeline_datos.py`, modelos históricos,
calibración, umbral ni evidencia previa. La verificación de conservación y las
pruebas de la limpieza se guardan en `limpieza_20261005/`. El README de uso muestra
solo comandos que existen en la versión actual. Para reconstruir los ensayos
retirados se necesita su código/commit y entorno registrados; no se deben ejecutar
sus antiguos comandos contra el entrenador actual.
