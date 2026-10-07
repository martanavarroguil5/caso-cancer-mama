# Experimentos cerrados y retirados del entrenador

Actualizado el 05/10/2026. Estos ensayos no forman parte del código activo de
`src/cancer_mama/entrenamiento.py` y sus comandos de comparación se han eliminado. No se
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
| Dropout2d p=0,10 | AUC media 0.5504 → 0.5729; IC pareado incluye cero | Inconcluyente; no incorporar al entrenador |
| Weight decay 0,003 | AUC media 0.5504 → 0.5542; Δ menor que +0,01 | Inconcluyente; mantener weight decay 0,001 del perfil ajustado |
| Aumentos afines suaves | AUC media 0.5504 → 0.5667; IC pareado incluye cero | Inconcluyente; no incorporar traslación/escala |
| EMA 0,99 con BN recalculada | AUC media 0.5504 → 0.5562; Δ menor que +0,01 | Inconcluyente; no incorporar EMA ni su recalculación de BN |

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

## Cuatro ensayos de generalización: no adoptados (05/10/2026)

Protocolo fijado en `3623227`, antes de ejecutar la matriz. Se compararon el
control `pool_dropout_wd` y cuatro cambios independientes: Dropout2d p=0,10
tras el último bloque, weight decay 0,003, traslaciones ±3%/escala 0,95–1,05,
y EMA de parámetros con decaimiento 0,99. La arquitectura conserva ocho
convoluciones, BatchNorm y 551.913 parámetros. No se combinaron candidatos.

Ochenta jobs y 1.431 épocas físicas: cinco folds originales, semillas 42/2026,
BCE normal y ponderada, máximo treinta épocas, lote 64 y LR 0,0008. Cien
evaluaciones principales y veinte diagnósticos de BN. EMA y el control con BN
recalculada comparten la trayectoria del control, con checkpoint y parada propios.
Dentro de los cuatro folds disponibles se separa fit/selección 85/15 por paciente
y cohorte × pCR. El fold exterior solo evalúa, sin seleccionar épocas ni umbral.

El criterio principal es AUC cruda media de folds del ensemble de dos semillas
con BCE ponderada. Los controles se reentrenaron con este protocolo.

| Cambio | AUC media | ΔAUC | IC98,75% pareado de ΔAUC | ΔAUC con BCE normal |
|---|---:|---:|---|---:|
| Control | 0.5504 | — | — | — |
| Dropout2d | 0.5729 | +0.0225 | [−0.0126, +0.0584] | −0.0075 |
| Weight decay 0,003 | 0.5542 | +0.0038 | [−0.0309, +0.0372] | −0.0064 |
| Afines | 0.5667 | +0.0163 | [−0.0152, +0.0506] | −0.0221 |
| EMA 0,99 | 0.5562 | +0.0058 | [−0.0262, +0.0362] | +0.0084 |

Todos mejoraron la estimación de AUC en cuatro de cinco folds, con diferencias
medias positivas en ambas semillas y mayor AP media ponderada. **Ninguno
cumplió el requisito de intervalo inferior positivo**; weight decay y EMA
tampoco alcanzaron la mejora práctica de +0,01. Dropout2d y afines dan señales
favorables que quedan inconcluyentes. Estos resultados no se describen como
un empeoramiento general de los cuatro métodos.

Los intervalos usan 4.000 bootstrap pareados por paciente, estratificados por
fold/cohorte/clase, y corrección Bonferroni para cuatro comparaciones. Condicionan
a esos modelos y no recogen toda la variabilidad del entrenamiento. Las pacientes
de desarrollo ya se habían observado antes; sigue siendo evaluación interna
adaptativa y no una confirmación externa.

La recalculación de BN del control obtuvo AUC 0.5558; EMA frente a ese diagnóstico
dio Δ +0.0004, IC95% [−0.0241, +0.0251]. No se demuestra un beneficio adicional
del promedio de parámetros frente a recalcular BN. Ese control era diagnóstico,
sin convertirlo después en un quinto candidato de búsqueda.

Se conserva `src/cancer_mama/entrenamiento.py` sin modificaciones: los candidatos se ejecutaron
en código aislado y nunca se añadieron al entrenador activo. Los 3.334 archivos
protegidos previos mantienen sus SHA-256. Se verificaron los 120 pesos de
inferencia, su correspondencia con el mejor checkpoint y su compatibilidad exacta
con la CNN original en evaluación CPU sintética. No se abrieron imágenes de test
ni de validación privada y no se sustituyen pesos, calibración o umbral históricos.

Evidencia y scripts congelados: `resultados/04_entrenamiento/cuatro_mejoras_20261005/`.
Detalles, métricas de ambas pérdidas y reconstrucción:
[RESULTADOS_CUATRO_MEJORAS.md](mejoras_cnn/RESULTADOS_CUATRO_MEJORAS.md).

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

No se modifican datos, archivos 01–03, `src/cancer_mama/datos.py`, modelos históricos,
calibración, umbral ni evidencia previa. La verificación de conservación y las
pruebas de la limpieza se guardan en `limpieza_20261005/`. El README de uso muestra
solo comandos que existen en la versión actual. Para reconstruir los ensayos
retirados se necesita su código/commit y entorno registrados; no se deben ejecutar
sus antiguos comandos contra el entrenador actual.
