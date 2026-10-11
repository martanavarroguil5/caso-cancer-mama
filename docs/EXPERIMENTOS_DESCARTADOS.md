# Experimentos cerrados y retirados del entrenador

Actualizado el 11/10/2026. Estos ensayos no forman parte del código activo de
`src/cancer_mama/entrenamiento.py` y sus comandos de comparación se han eliminado. No se
vuelven a ejecutar automáticamente. Se conservan sus informes, predicciones,
curvas, configuraciones y evidencia histórica en `resultados/04_entrenamiento/`.
El código y README anteriores a esta limpieza quedan en
`limpieza_20261005/codigo_anterior/`; los commits citados identifican cada ensayo.

## Cierre de los ensayos posteriores al multimodal original (11/10/2026)

**Decisión: se restaura `v002_multimodal_clinico_20261007` como modelo activo.
Los ensayos posteriores quedan registrados como fracasos del objetivo de superar
al multimodal original. V003 y v004 se conservan como experimentos descartados.**

| Ensayo | Resultado y motivo del cierre |
|---|---|
| LR 0,0008/0,0002 y selección por log-loss; candidata v003 | Reducción de pérdida frente al control equiparado no concluyente después de calibrar. No demuestra superioridad sobre v002. |
| ResNet50 congelada, PCA y fusión clínica; AlexNet local y SVM/logística; ResNet18 parcialmente ajustada con L2-SP | No cumplieron el criterio de mejora; además usan redes preentrenadas prohibidas por el enunciado. Fuera de la entrega. |
| Textura y diferencias entre fases PNG | No cumplieron el criterio de mejora. Comparador auxiliar archivado. |
| BCE normal con inicialización clínica sin ponderar y selección por pérdida; candidata v004 | Mejoró un control reentrenado con separación interna, pero no demostró superar al v002 original. Su promoción se revoca. |
| BCE normal con mixup | No mejoró la variante BCE normal y no cumplió el criterio fijado frente al control. Descartado. |
| Repetición BCE normal, semillas 73/1701 | Repitió la reducción de pérdida frente al control reentrenado, sobre las mismas pacientes. No fue una prueba independiente frente al v002 original ni validación externa. |

### Comparación directa de las versiones guardadas

Sobre 1.097 pacientes de desarrollo (322 pCR y 775 no pCR), usando las predicciones
OOF históricas, la calibración y el umbral guardados de cada versión:

| Métrica | Original v002 | Candidata v004 |
|---|---:|---:|
| ROC-AUC | 0,724568 | 0,710888 |
| Average precision | 0,531646 | 0,508693 |
| Log-loss | 0,536656 | 0,546255 |
| Brier | 0,179060 | 0,182626 |
| Falsos positivos | 187 | 210 |
| Falsos negativos | 128 | 126 |
| Sensibilidad | 60,25 % | 60,87 % |
| Especificidad | 75,87 % | 72,90 % |
| Exactitud | 71,29 % | 69,37 % |
| F1 | 0,551920 | 0,538462 |
| Umbral | 0,3543331693 | 0,3489952150 |

V004 evita dos falsos negativos a costa de 23 falsos positivos adicionales.
El v002 original conserva mejores AUC, AP, pérdida calibrada, Brier, exactitud y F1
en estas predicciones. Las diferencias de entrenamiento/selección impiden tratar
esta tabla como un ensayo controlado de superioridad; son métricas aparentes de
desarrollo, con calibración y umbrales ajustados sobre OOF, no métricas de test.
El original tampoco queda validado externamente por esta comparación.

**Corrección de la interpretación anterior:** la reducción de log-loss
0,5615 → 0,5463 y su repetición 0,5678 → 0,5467 pertenecen a controles reentrenados
con otro protocolo. No deben presentarse como mejora sobre los pesos originales
de v002. Comparar la pérdida cruda 0,6098 de v002 con v004 ignoraba que v002
ya disponía de calibración Platt; su pérdida calibrada guardada es 0,5367.
La selección temporal de v004 por defecto fue prematura y queda revocada.

### Código, evidencia y restauración

- V002 recupera sus diez pesos, Platt, mediana y umbral originales, sin reentrenar.
- El entrenador activo conserva únicamente el perfil multimodal original con
  BCE ponderada por corte. La aplicación y CLI usan su manifiesto.
- Doce módulos, ocho archivos de pruebas, la comprobación v004 y las dependencias
  de transferencia se archivan, sin cambiar sus bytes, en
  [experimentos/descartados_20261011](../experimentos/descartados_20261011/README.md).
- Las versiones congeladas v001–v004, los datasets y checkpoints de trabajo se conservan.
  Los informes 08–11 mantienen sus resultados originales con una nota de cierre.
- El test reservado no se abrió para esta decisión ni para la limpieza.

Evidencia: [métricas directas CSV](mejoras_cnn/v002_vs_v004_20261011.csv),
[métricas y hashes JSON](mejoras_cnn/v002_vs_v004_20261011.json),
[ensayo de pérdidas](../resultados/08_informe_loss_20261010/INFORME_RESULTADOS.md),
[investigación visual](../resultados/09_informe_transferencia_20261010/INFORME_RESULTADOS.md),
[BCE/mixup](../resultados/10_informe_bce_mixup_20261011/INFORME_RESULTADOS.md) y
[repetición](../resultados/11_confirmacion_bce_normal_20261011/INFORME_RESULTADOS.md).
El commit `8c0600c` identifica el estado anterior a esta limpieza; `41428a2`
conserva las fuentes originales de la comparación antes de cambiar interfaz y versión.

La [verificación de la restauración](mejoras_cnn/verificacion_restauracion_v002_20261011.json)
registra 39 pruebas del código activo, una prueba de un lote/época en GPU,
inferencia aplicación/CLI/paquete en FP32 y 302 archivos protegidos sin cambios.
Los 22 archivos archivados conservan exactamente sus bytes originales.

«Fallido» describe el objetivo de mejorar el original en estas condiciones.
No significa que los entrenamientos fallaran técnicamente ni que las técnicas
sean inservibles para otros datos o configuraciones.


## Decisiones históricas anteriores

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
| Realce temporal explícito, con y sin mayor regularización | AUC OOF 0.724568 → 0.726087 / 0.721992; sin mejora útil demostrada | **Fracaso del objetivo de mejora; descartado. Conservar el multimodal v002** |

Las decisiones son específicas de este proyecto y las condiciones ensayadas.
No prueban que esos métodos sean inferiores con cualquier dato o hiperparámetro.
La base activa vuelve a ser el multimodal clínico v002 ya entrenado (11/10/2026).
La limpieza retira código; conserva los pesos, calibración y umbral de esa versión.

## Realce temporal: fracaso del objetivo de mejora (10/10/2026)

**Decisión: experimento cerrado y descartado; nos quedamos con
`v002_multimodal_clinico_20261007`.** No se adopta ninguno de los dos brazos.
«Fracaso» significa que el ensayo no demostró una mejora útil sobre la base
multimodal, no que el entrenamiento fallara ni que toda regularización sea inferior.

Código ejecutado: `c0f23167fd465d7929e868629a7e9d693c566541`.
Se completaron 20 ejecuciones (dos variantes × semillas 42/2026 × cinco folds),
293 épocas en total, en 61 minutos y 10 segundos. BCE ponderada por corte,
máximo 46 épocas, lote 64 y parada temprana. Solo se entrenaron las variantes
nuevas; los modelos previos se reutilizaron para comparar.

`enhancement_clinical` añadía a PRE/EARLY/LATE tres diferencias firmadas:
EARLY−PRE, LATE−PRE y LATE−EARLY. `enhancement_regularized_clinical`
añadía además dropout 0,45, Dropout2d 0,10 y weight decay 0,003, frente a
dropout 0,35 y weight decay 0,001. La rama clínica usaba edad, volumen tumoral,
HR y HER2; imputación, escala e inicialización se ajustaban con train de cada fold.

Comparación interna sobre las mismas 1.097 pacientes de desarrollo (322 pCR
y 775 no pCR), promediando las dos semillas por corte. La versión conservada
agrega cortes mediante mediana; los mejores candidatos nuevos usaron media.

| Modelo | AUC OOF | AUC media de folds | AP | Brier crudo | FP a 0,5 crudo | FN a 0,5 crudo |
|---|---:|---:|---:|---:|---:|---:|
| Multimodal v002 (conservado) | 0.724568 | 0.724311 | 0.531646 | 0.210852 | 272 | 95 |
| Realce temporal | 0.726087 | 0.728979 | 0.536726 | 0.213788 | 275 | 97 |
| Realce con mayor regularización | 0.721992 | 0.724584 | 0.524162 | 0.208895 | 261 | 104 |

El realce simple mejora la estimación de AUC OOF en solo +0,001519, con peor
Brier crudo y tres FP y dos FN adicionales a ese umbral. El brazo regularizado
reduce once FP a cambio de nueve FN adicionales y una AUC OOF menor (−0,002577).
La AUC media de folds del regularizado sube ligeramente: no se afirma que
empeore todas las métricas.

El análisis posterior pareado (2.000 bootstrap estratificados por clase,
semilla 20261010) dio IC95% de ΔAUC OOF [−0,009415, +0,011369] para realce
y [−0,012555, +0,007150] para realce regularizado. Ambos incluyen cero.
Estos intervalos condicionan a los modelos y a sus predicciones OOF: no corrigen
el optimismo de selección de checkpoints/candidatos ni equivalen a validación externa.

Con la calibración y el umbral Youden propios de cada modelo, ajustados sobre
esas mismas OOF, el realce simple produjo 211 FP y 116 FN, frente a 187 FP y
128 FN del multimodal. Son resultados aparentes con umbrales distintos;
la mayor sensibilidad se paga con más falsos positivos. Tampoco respaldan
sustituir la base actual. No se reabrió el test reservado.

Evidencia conservada:

- `resultados/04_entrenamiento/realce_temporal_20261009/comparacion/seleccion.json`
- `resultados/04_entrenamiento/realce_temporal_20261009/comparacion/oof_pacientes.csv`
- Configuraciones, resúmenes, checkpoints y curvas de los 20 runs en esa carpeta.
- `seguimiento/estado.json`, log de entrenamiento y comprobación de pesos anteriores.
- Referencia conservada: `modelos/versiones/v002_multimodal_clinico_20261007/`.

Se retiran del entrenador activo ambos perfiles de realce, sus seis canales,
Dropout2d y el lanzador del experimento. También se retiran las otras alternativas
de entrenamiento, incluida la BCE por paciente multimodal. Sus informes y
predicciones anteriores siguen disponibles; su código se recupera del commit
correspondiente, sin reinterpretar los comandos antiguos en el entrenador actual.

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

## Alcance de la limpieza anterior (10/10/2026)

Se eliminan los drivers y ramas experimentales del entrenador y sus pruebas
exclusivas. Se mantienen las pruebas de arquitectura, fases, pacientes, OOF,
calibración, caché, reanudación e inferencia. El código activo conserva solo la
CNN multimodal con BatchNorm, BCE ponderada por corte, pooling intermedio,
dropout 0,35 y weight decay 0,001. La inferencia por defecto y la aplicación
usan el manifiesto de v002. El ensemble promedia modelos por corte y después
aplica mediana entre los cortes de una paciente, con su calibración y umbral guardados.

No se modifican datos, archivos 01–03, `src/cancer_mama/datos.py`, modelos históricos,
calibración, umbral ni evidencia previa. La verificación de conservación y las
pruebas de esta limpieza se guardan en `.codex_tmp/limpieza_multimodal_20261010/`;
la evidencia de la limpieza anterior sigue en `limpieza_20261005/`. El README de uso muestra
solo comandos que existen en la versión actual. Para reconstruir los ensayos
retirados se necesita su código/commit y entorno registrados; no se deben ejecutar
sus antiguos comandos contra el entrenador actual.
