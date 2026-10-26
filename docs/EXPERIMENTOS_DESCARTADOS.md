# Experimentos cerrados y retirados del entrenador

Actualizado el 21/10/2026. Estos ensayos no forman parte del código activo de
`src/cancer_mama/entrenamiento.py` y sus comandos de comparación se han eliminado. No se
vuelven a ejecutar automáticamente. Se conservan sus informes, predicciones,
curvas, configuraciones y evidencia histórica en `resultados/04_entrenamiento/`.
El código y los README anteriores se conservan en el historial de `main`;
los commits citados identifican cada ensayo. Las copias locales de limpiezas
antiguas no forman parte del código activo ni son necesarias para recuperarlo.

## CNN cinética multiescala: ajuste insuficiente (21/10/2026)

**No alcanza el mínimo AUC 0,60 ni el objetivo 0,70; no sustituye V002.**
Se mantiene exactamente la arquitectura de `39bd951`, con 593.521 parámetros.
El ajuste compara LR/WD 0,0008/0,001, 0,0003/0,001 y 0,0003/0,01, ambas BCE,
semillas 42/2026 y cinco folds. Son 60 ejecuciones, 948 épocas y 220.884 pasos
de AdamW, sin pasos omitidos; duración 3 h 17 min. No se abre test.

| Métrica exterior | Referencia de la misma red | Ajustada |
|---|---:|---:|
| AUC OOF global cruda | 0,535981 | 0,526404 |
| Media de AUC por fold | 0,581014 | 0,576593 |
| AUC con calibración interna por fold | 0,578982 | 0,561803 |
| Log-loss con calibración interna | 0,604072 | 0,613787 |
| Brier con calibración interna | 0,205838 | 0,209418 |
| FP / FN / TP | 55 / 281 / 41 | 74 / 281 / 41 |

La diferencia de AUC global candidato − referencia es −0,009577, IC95 %
pareado [−0,028690; +0,008952]. Frente a la CNN por fases v006 la mejora
puntual es solo +0,001831 y su intervalo también incluye cero. El AUC histórico
solo imagen de v001 (0,594510) y el multimodal v002 (0,724568) no se superan;
esas comparaciones son descriptivas por sus protocolos e información distintos.

El máximo interno 0,710 de un fold cae a 0,577 en su evaluación exterior.
En los diez modelos elegidos, la mediana de brechas fit − selección crece de
0,058252 en el checkpoint seleccionado a 0,268942 en la última época. Los diez
terminan con peor pérdida interna que en su checkpoint elegido. Más épocas
no resuelven el sobreajuste observado y los ajustes probados no aportan una
mejora exterior respaldada.

Se conservan los diez pesos, fuentes, calibración y hashes en
`v007_cnn_cinetica_multiescala_20261021`, como `ensayo_insuficiente_conservado`.
El código de esta preparación docente permanece disponible para revisión y
reproducción; no forma parte del modelo activo de la aplicación ni se ejecuta
automáticamente. La arquitectura no se retira ni se rehace antes de que el
usuario revise el resultado. No se inicia otro entrenamiento.

Código del ajuste: `b0d4ea2`; soporte de carga y archivo: `6a50a77`.
[Informe completo, curvas y verificación](../resultados/15_informe_cinetico_20261021/INFORME_RESULTADOS.md).

## CNN solo imagen por fases: discriminación insuficiente (14/10/2026)

Las 20 ejecuciones de BCE normal/ponderada, dos semillas y cinco folds
terminaron correctamente en 1 h 58 min, con 326 épocas y el test sin abrir.
La CNN desde cero emplea un encoder monocanal compartido para PRE/EARLY/LATE
y fusión de diferencias temporales, sin clínica. Código de entrenamiento:
commit `3eec1f0`; entrenamiento y arquitectura no se modifican al archivar.

| Métrica | Normal | Ponderada |
|---|---:|---:|
| AUC medio por fold ± desviación | 0,5330 ± 0,0253 | 0,5486 ± 0,0504 |
| AUC global OOF cruda | 0,5087 | 0,5246 |
| Sensibilidad con umbral interno conservador | 6,52 % | 11,18 % |
| Especificidad | 92,13 % | 90,58 % |
| FP / FN | 61 / 301 | 73 / 286 |

**Resultado insuficiente; no se promociona a la aplicación.** Persiste una
brecha creciente entre fit y selección interna. La ponderada gana por el
criterio prefijado de media de AUC por fold, pero su calibración final acaba
en pendiente 0 y umbral 1, con probabilidad constante 0,293528. No se confunde
esa política final con los umbrales por fold de la tabla ni se corrige después
sin tratarlo como otro experimento.

Se conservan todos los pesos de inferencia en v005 (normal) y v006 (ponderada),
con código compatible y hashes. El entrenador independiente de imagen se
mantiene para reproducibilidad de la nueva preparación docente; el entrenador
y los pesos activos del multimodal v002 no cambian. No se lanza otra prueba.
La comparación con v002 es descriptiva por sus entradas y selección distintas.

[Informe, evidencia y límites](../resultados/14_informe_cnn_imagen_20261014/INFORME_RESULTADOS.md).
No repetir esta misma configuración esperando que más épocas eliminen el
sobreajuste. Este resultado no demuestra que todas las arquitecturas visuales
fallen ni que las imágenes carezcan de señal.

## SAM: señal interna insuficiente y aporte visual prácticamente nulo (12/10/2026)

**Decisión: mantener activo el multimodal original v002 y descartar esta
configuración SAM como sustituta.** El entrenamiento no falló técnicamente:
se completaron 40/40 ejecuciones, 542 épocas y 63.414 pasos de AdamW, sin pasos
omitidos por AMP. Duración: 82 min 57 s, del 11/10/2026 a las 20:40 a las 22:03
(Europe/Madrid). Cierre y análisis el 12/10/2026.

Se conserva el [protocolo prefijado](mejoras_cnn/PROTOCOLO_SAM.md): arquitectura
v002, SAM L2 rho=0,05 sobre todos los parámetros con AdamW, BCE normal/ponderada,
dos semillas y cinco folds. Se usaron solo pacientes de train. No se modificaron
los pesos, calibración ni umbral de v002 y no se inició otro entrenamiento.
El código experimental y sus pruebas quedan en el commit `4a2c959` de `main`;
se retiran del árbol actual sin nuevas ramas ni copias de archivo.

### Comparación descriptiva con el original y comparación equiparada

1.097 pacientes, 322 pCR y 775 no pCR. AUC con probabilidades OOF sin calibrar;
pérdida/Brier después de calibrar; FP/FN con el umbral guardado de v002 o Youden
interno por fold en el ensayo. **El original usa selección histórica distinta
y calibración/umbral ajustados sobre las mismas OOF.** Por ello, original frente
a SAM es una comparación descriptiva, no un ensayo controlado ni validación externa.
El control reentrenado es quien permite aislar el efecto de SAM bajo este protocolo.

| Métrica | V002 original | Control ponderado | SAM ponderado |
|---|---:|---:|---:|
| ROC-AUC sin calibrar | 0,724568 | 0,701968 | 0,707053 |
| Average precision sin calibrar | 0,531646 | 0,494831 | 0,495531 |
| Log-loss calibrada | 0,536656 | 0,561517 | 0,558266 |
| Brier calibrado | 0,179060 | 0,187723 | 0,187611 |
| Falsos positivos | 187 | 297 | 319 |
| Falsos negativos | 128 | 112 | 103 |
| Sensibilidad | 60,25 % | 65,22 % | 68,01 % |
| Especificidad | 75,87 % | 61,68 % | 58,84 % |
| Exactitud | 71,29 % | 62,72 % | 61,53 % |
| F1 | 0,551920 | 0,506634 | 0,509302 |

SAM ponderado evita 25 falsos negativos respecto al original a costa de 132
falsos positivos adicionales: 107 errores más con estos umbrales en estas OOF.
«Positivo» significa respuesta patológica completa (pCR), no presencia de cáncer.
No se extrapolan estos recuentos a pacientes nuevas.

### Hipótesis principal y brazo secundario

SAM ponderado gana +0,005085 de AUC frente a su control, supera por poco el
margen prefijado +0,005 y mejora ambas semillas: 42, 0,700441 → 0,706940;
2026, 0,701274 → 0,707117. También reduce ligeramente pérdida y Brier.
Sin embargo, **no cumple el requisito de intervalo de AUC completamente positivo**.

| Diferencia SAM ponderado − control | Estimación | IC pareado del 95 % |
|---|---:|---:|
| AUC OOF agrupada sin calibrar | +0,005085 | [−0,006067; +0,015778] |
| Media de AUC por fold | +0,001094 | [−0,008714; +0,010623] |
| Log-loss calibrada | −0,003251 | [−0,010017; +0,003228] |
| Brier calibrado | −0,000112 | [−0,002367; +0,002082] |

Los 2.000 remuestreos pareados y estratificados por fold/cohorte/clase incluyen
cero en todos los intervalos. Hay una señal pequeña favorable, insuficiente
para afirmar superioridad. Los intervalos no corrigen toda la búsqueda histórica.
Los controles reproducen las métricas de los controles del ensayo FiLM.

| Métrica secundaria | Control BCE normal | SAM BCE normal |
|---|---:|---:|
| AUC OOF agrupada sin calibrar | 0,681667 | 0,694879 |
| Media de AUC por fold | 0,694701 | 0,707139 |
| Log-loss calibrada | 0,567060 | 0,560038 |
| Brier calibrado | 0,189359 | 0,187211 |
| Falsos positivos | 311 | 299 |
| Falsos negativos | 104 | 102 |

SAM normal también mejora sus estimaciones puntuales frente a su control,
pero no supera las métricas descriptivas del v002 original. Esta comparación
es secundaria y no activa una selección automática de pérdida.

### Sobreajuste: menor brecha, sin nuevo beneficio visual

En la última época, la mediana de AUC de fit/selección interna pasa de
0,8359/0,6407 en el control ponderado a 0,7207/0,7113 con SAM ponderado.
La mediana de las brechas por ejecución cae de 0,1970 a 0,0125. Estos valores
describen la última época; no son las métricas exteriores de los checkpoints
seleccionados. SAM ponderado elige época 0 en 2/10 ejecuciones (control: 6/10).
Una brecha menor por sí sola no demuestra un mejor multimodal.

Se hizo una **ablación posterior sin reentrenar**: quitar el término visual del
logit de cada checkpoint elegido, manteniendo sus pesos clínicos finales y sesgo.
Se promedian las dos semillas por corte y se agrega por mediana igual que en el
modelo completo. No se introducen imágenes negras, no se ajustan umbrales ni se
calibra de nuevo. Es un diagnóstico interno exploratorio, no una nueva selección.

| Brazo | AUC completo | AUC sin término visual | Cambio absoluto medio de probabilidad |
|---|---:|---:|---:|
| Control ponderado | 0,701968 | 0,707946 | 0,027935 |
| SAM ponderado | 0,707053 | 0,707053 | 0,000000421 |
| Control normal | 0,681667 | 0,706540 | 0,082211 |
| SAM normal | 0,694879 | 0,701106 | 0,039169 |

En SAM ponderado, la media de AUC por fold también es idéntica con y sin imagen
(0,705778). Las predicciones seleccionadas dependen prácticamente de la rama
clínica. La menor brecha no equivale a haber aprendido mejores características
visuales: este ensayo no alcanza el objetivo de aprovechar mejor las imágenes.
En SAM normal, quitar imagen mejora el AUC agrupado pero baja la media por fold
de 0,707139 a 0,705023; no deben confundirse ambas agregaciones.

No se concluye que SAM sea inútil en general ni que todas las configuraciones de
radio/optimización deban fallar. No se prueba otro radio tras ver estos resultados.

### Evidencia verificable

Los hashes de los artefactos de las 40 ejecuciones coinciden; los 40 checkpoints
cargan estrictamente. Una muestra real de train por checkpoint reproduce en CPU
su predicción FP32 guardada con diferencia máxima 9,24 × 10⁻⁸ (tolerancia 2 × 10⁻⁶).
Los 299 archivos protegidos siguen intactos. Test no se evaluó. Tras retirar
el módulo SAM y sus pruebas, pasan las 39 pruebas del código activo y la
verificación de v002. Se recalcularon las métricas de las cuatro tablas OOF y
se comprobaron los enlaces: [verificación del cierre](../resultados/13_informe_sam_20261012/verificacion_cierre.json).

- [Resumen y calibraciones](../resultados/13_informe_sam_20261012/resumen.json) y [métricas CSV](../resultados/13_informe_sam_20261012/comparacion_metricas.csv).
- [ROC comparadas](../resultados/13_informe_sam_20261012/01_roc.png) y [curvas de pérdida/AUC](../resultados/13_informe_sam_20261012/02_curvas.png).
- [Ablación visual](../resultados/13_informe_sam_20261012/ablacion_visual.json) y [diagnóstico por ejecución](../resultados/13_informe_sam_20261012/diagnostico_final.csv).
- [Verificación de pesos](../resultados/13_informe_sam_20261012/verificacion_final.json), [procedencia y hashes](../resultados/13_informe_sam_20261012/procedencia.json), [configuraciones y resúmenes](../resultados/13_informe_sam_20261012/ejecuciones.json) e [historias](../resultados/13_informe_sam_20261012/historias.csv).

Las tablas OOF y las de ablación están en la misma carpeta de evidencia. Los
checkpoints completos permanecen localmente en
`resultados/04_entrenamiento/sam_multimodal_20261011/`, fuera de Git. Fuentes:

```bash
git show 4a2c959:src/cancer_mama/experimento_sam.py
git show 4a2c959:tests/test_pasos_sam.py
```


## FiLM: ensayo terminado sin mejora (11/10/2026)

**Decisión: descartar esta configuración de FiLM y mantener activo el multimodal
original v002 con sus pesos, calibración y umbral intactos.** No se inicia otro
entrenamiento. FiLM no satisface el criterio fijado antes del ensayo y no resuelve
el sobreajuste. El código experimental se retira del árbol actual; se recupera
del commit `9e2aa93` en la historia de `main`, sin ramas adicionales.

Terminado correctamente a las 19:55 del 11/10/2026 (Europe/Madrid): **40/40
ejecuciones, 549 épocas, 78 min 7 s de tiempo total**. Se probaron CNN control y
CNN + FiLM, cada una con BCE ponderada y normal, semillas 42/2026 y cinco folds.
La capa FiLM añadía 1.152 parámetros para modular las características visuales
según las variables clínicas. No se usaron redes preentrenadas ni test reservado.
El [protocolo prefijado](mejoras_cnn/PROTOCOLO_FILM.md) conserva todos los detalles.

### Comparación solicitada con los pesos originales v002

1.097 pacientes de desarrollo: 322 con pCR y 775 sin pCR. AUC y AP se calculan
con probabilidades OOF sin calibrar; pérdida y Brier después de calibrar; errores
con los umbrales declarados de cada método. **La comparación es descriptiva:**
v002 utiliza su selección histórica y una calibración/umbral ajustados a las
mismas OOF. FiLM utiliza selección, calibración y umbrales internos por fold,
aplicados al exterior. Esta diferencia impide atribuir toda la diferencia a FiLM
o considerar la tabla una validación externa de v002.

| Métrica | V002 original | FiLM ponderado | Cambio FiLM − v002 |
|---|---:|---:|---:|
| ROC-AUC, mayor mejor | 0,724568 | 0,701439 | −0,023130 |
| Average precision, mayor mejor | 0,531646 | 0,502409 | −0,029237 |
| Log-loss calibrada, menor mejor | 0,536656 | 0,560865 | +0,024209 |
| Brier calibrado, menor mejor | 0,179060 | 0,187753 | +0,008693 |
| Falsos positivos | 187 | 254 | +67 |
| Falsos negativos | 128 | 130 | +2 |
| Verdaderos positivos | 194 | 192 | −2 |
| Verdaderos negativos | 588 | 521 | −67 |
| Sensibilidad | 60,25 % | 59,63 % | −0,62 puntos |
| Especificidad | 75,87 % | 67,23 % | −8,65 puntos |
| Exactitud | 71,29 % | 65,00 % | −6,29 puntos |
| Precisión positiva | 50,92 % | 43,05 % | −7,87 puntos |
| F1 | 0,551920 | 0,500000 | −0,051920 |

Aquí «positivo» significa pCR. Un falso positivo predice respuesta completa
cuando no la hubo; un falso negativo omite una respuesta completa real.
Son 69 errores adicionales en estas predicciones y con estos umbrales, no una
estimación de 69 errores adicionales en futuras pacientes.

### Comparación controlada para aislar el efecto de FiLM

El control es la arquitectura de v002 **reentrenada bajo el protocolo de este
ensayo**, no los pesos originales. En cada par se mantienen particiones,
semillas, inicialización común, optimizador y selección por AUC interna.

| Métrica | Control ponderado | FiLM ponderado | Control normal | FiLM normal |
|---|---:|---:|---:|---:|
| ROC-AUC OOF agrupada sin calibrar | 0,701968 | 0,701439 | 0,681667 | 0,671753 |
| Media de AUC de los cinco folds | 0,704684 | 0,701576 | 0,694701 | 0,695303 |
| Average precision sin calibrar | 0,494831 | 0,502409 | 0,476552 | 0,470817 |
| Log-loss calibrada | 0,561517 | 0,560865 | 0,567060 | 0,571364 |
| Brier calibrado | 0,187723 | 0,187753 | 0,189359 | 0,190299 |
| Falsos positivos con Youden interno | 297 | 254 | 311 | 276 |
| Falsos negativos con Youden interno | 112 | 130 | 104 | 134 |

Hipótesis principal, FiLM ponderado menos control ponderado:

| Diferencia | Estimación | IC pareado del 95 % |
|---|---:|---:|
| ROC-AUC sin calibrar | −0,000529 | [−0,011273; +0,010331] |
| Media de AUC por fold | −0,003108 | [−0,012926; +0,006739] |
| Log-loss calibrada | −0,000653 | [−0,005991; +0,004524] |
| Brier calibrado | +0,000030 | [−0,001746; +0,001729] |

Bootstrap de 2.000 remuestreos pareados, estratificados por fold/cohorte/clase.
Los intervalos incluyen cero: no se demuestra mejora ni equivalencia exacta.
Son intervalos internos exploratorios; no corrigen toda la búsqueda histórica
ni constituyen validación externa. No se alcanzan +0,005 de AUC, límite inferior
positivo, Brier no peor ni ausencia de deterioro por semilla:

| Semilla | Control ponderado AUC | FiLM ponderado AUC |
|---|---:|---:|
| 42 | 0,700441 | 0,690795 |
| 2026 | 0,701274 | 0,695524 |

FiLM reduce 43 falsos positivos frente a ese control a costa de 18 falsos
negativos más, con umbrales ajustados por separado. La pequeña ganancia de AP
y pérdida no permite concluir superioridad global. AUC por cohorte:
Duke 0,7244 → 0,7205; I-SPY1 0,8013 → 0,7692; I-SPY2 0,6833 → 0,6868.
El brazo BCE normal tampoco rescata la hipótesis: empeoran AUC agrupada, pérdida
y Brier, aunque mejora ligeramente la media de AUC por fold (+0,000603).

### Qué ocurrió con el sobreajuste

FiLM ponderado conserva una brecha clara al **final** de los entrenamientos:
mediana de AUC de fit 0,8315 frente a 0,6436 en selección interna; mediana de las
brechas por ejecución 0,2032. En el control: 0,8359 y 0,6408, con brecha mediana
0,1970. Son diagnósticos de la última época, **no** resultados del checkpoint
elegido ni del exterior OOF. La selección temprana evita conservar esos pesos
finales deteriorados; no demuestra haber eliminado el sobreajuste.

En 6 de las 10 ejecuciones de FiLM ponderado se seleccionó la época 0, igual
que en el control ponderado. Esa época conserva el inicializador clínico y
FiLM identidad; solo 4 checkpoints ponderados elegidos tienen FiLM aprendido.
La pérdida por paciente de selección empeora respecto al inicio en 8/10
ejecuciones FiLM ponderadas (9/10 controles). Añadir capacidad de interacción
clínica-imagen no produjo un beneficio consistente con esta receta y estos datos.
No permite concluir que cualquier arquitectura FiLM vaya a fallar.

### Evidencia y cierre del código

Se verificaron los hashes de resultados y la carga estricta de los 40 checkpoints.
Una predicción de train por checkpoint en CPU coincide con la OOF FP32 guardada
con diferencia máxima 2,04 × 10⁻⁷ (tolerancia 2 × 10⁻⁶). Se mantienen sin cambios
los 299 archivos protegidos por el protocolo, incluidos los paquetes congelados.
La verificación no reentrena ni evalúa test. Tras retirar el módulo experimental,
pasaron las 39 pruebas del código activo y la verificación del manifiesto v002.
También se recalcularon las métricas de las cuatro tablas OOF y se comprobaron
los enlaces de documentación; véase la [verificación del cierre](../resultados/12_informe_film_20261011/verificacion_cierre.json).

- [Resumen completo y calibraciones](../resultados/12_informe_film_20261011/resumen.json).
- [Tabla de métricas CSV](../resultados/12_informe_film_20261011/comparacion_metricas.csv).
- [ROC comparadas](../resultados/12_informe_film_20261011/01_roc.png).
- [Curvas de pérdida y AUC](../resultados/12_informe_film_20261011/02_curvas.png): mediana e IQR de las diez ejecuciones, épocas 0–12 comunes a todas, evaluación FP32 sin aumentos.
- [Diagnóstico por ejecución](../resultados/12_informe_film_20261011/diagnostico_final.csv).
- [Verificación de pesos y predicciones](../resultados/12_informe_film_20261011/verificacion_final.json).
- [Procedencia y hashes](../resultados/12_informe_film_20261011/procedencia.json), [configuraciones, entornos y resúmenes](../resultados/12_informe_film_20261011/ejecuciones.json), [historias completas](../resultados/12_informe_film_20261011/historias.csv) y [protocolo original y hashes](../resultados/12_informe_film_20261011/protocolo.json).

La carpeta de evidencia contiene también las cuatro tablas OOF y un script que
regenera las figuras sin entrenar. Los checkpoints y ejecuciones completas
permanecen locales en `resultados/04_entrenamiento/film_multimodal_20261011/`,
fuera de Git. No se crea una versión de modelo activa ni una rama para este ensayo.
Para consultar la implementación retirada:

```bash
git show 9e2aa93:src/cancer_mama/experimento_film.py
git show 9e2aa93:tests/test_experimento_film.py
```


## Cierre de los ensayos posteriores al multimodal original (11/10/2026)

**Decisión: se restaura `v002_multimodal_clinico_20261007` como modelo activo.
Los ensayos posteriores quedan registrados como fracasos del objetivo de superar
al multimodal original. V003 y v004 se conservan como experimentos descartados.**

| Ensayo | Resultado y motivo del cierre |
|---|---|
| LR 0,0008/0,0002 y selección por log-loss; candidata v003 | Reducción de pérdida frente al control equiparado no concluyente después de calibrar. No demuestra superioridad sobre v002. |
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
  no utilizados se retiran del árbol actual.
- Las versiones congeladas v001–v004, los datasets y checkpoints de trabajo se conservan.
  Los informes 08–11 mantienen sus resultados originales con una nota de cierre.
- El test reservado no se abrió para esta decisión ni para la limpieza.

Evidencia: [métricas directas CSV](mejoras_cnn/v002_vs_v004_20261011.csv),
[métricas y hashes JSON](mejoras_cnn/v002_vs_v004_20261011.json),
[ensayo de pérdidas](../resultados/08_informe_loss_20261010/INFORME_RESULTADOS.md),
[BCE/mixup](../resultados/10_informe_bce_mixup_20261011/INFORME_RESULTADOS.md) y
[repetición](../resultados/11_confirmacion_bce_normal_20261011/INFORME_RESULTADOS.md).
El commit `8c0600c` identifica el estado anterior a esta limpieza; `41428a2`
conserva las fuentes originales de la comparación antes de cambiar interfaz y versión.

La [verificación de la restauración](mejoras_cnn/verificacion_restauracion_v002_20261011.json)
registra 39 pruebas del código activo, una prueba de un lote/época en GPU,
inferencia aplicación/CLI/paquete en FP32 y 302 archivos protegidos sin cambios.
En esa verificación previa se comprobó que los 22 archivos archivados conservaban
exactamente sus bytes originales. Esa copia temporal se conserva en el commit
`bbd769f`; posteriormente se retiró del árbol actual al consolidar todo en `main`.

«Fallido» describe el objetivo de mejorar el original en estas condiciones.
No significa que los entrenamientos fallaran técnicamente ni que las técnicas
sean inservibles para otros datos o configuraciones.

## Historial único en main

Por decisión del usuario, todo el desarrollo queda en `main`. Las cuatro ramas
auxiliares se integraron por avance directo, sin reescribir ni compactar commits,
y se eliminan una vez comprobado que sus puntas pertenecen a esta historia.
El código actual mantiene v002. Los pesos y paquetes congelados v001–v004
siguen versionados; v003 y v004 no se activan.

| Commit | Estado recuperable |
|---|---|
| `4a2c959` | Ensayo SAM con controles; descartado por señal insuficiente y aporte visual prácticamente nulo |
| `9e2aa93` | Implementación, pruebas y protocolo prefijado de FiLM; descartado sin mejora |
| `d1d61b8` | Informe del entrenamiento multimodal original |
| `7116b8e` | Multimodal v002 como única base activa antes de los últimos ensayos |
| `2cde016` | Ensayo de pérdidas/LR y candidata v003 |
| `497abd6` | Investigación visual descartada y restricciones del enunciado |
| `847466e` | Resultados de BCE normal y mixup |
| `41428a2` | Repetición de BCE normal con semillas nuevas |
| `8c0600c` | Fuentes experimentales completas y selección posteriormente revocada de v004 |
| `bbd769f` | Restauración de v002 y copia temporal de las fuentes descartadas |

Para leer un archivo histórico sin cambiar el modelo activo ni crear una rama:

```bash
git show 8c0600c:src/cancer_mama/experimento_bce_mixup.py
git show 41428a2:src/cancer_mama/confirma_bce_normal.py
```

Reproducir un ensayo requiere también su configuración, entorno y datos registrados;
no se deben ejecutar sus comandos históricos contra el entrenador actual.
Este Markdown concentra la decisión y enlaza las evidencias conservadas.

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

## Campo suave de adquisición: piloto cerrado sin mejora (25/10/2026)

Código revisable en el commit local 2c48ab2f533550e62853758dd43ef90af7a673cb; se retira del código activo y se conserva en la historia de main. Protocolo y [informe completo](../resultados/25_robustez_adquisicion_20261025/INFORME_RESULTADOS.md).

Misma CNNMultiescalaCinetica de593.521 parámetros: control/campo común entre fases, BCE normal/ponderada, semillas42/2026 y folds0/3. Dieciséis entrenamientos;439 pacientes exteriores de train,139 pCR. Fit, selección, calibración y comprobación separados por paciente para cada modelo. Épocas y pérdidas congeladas antes de calibración; umbrales congelados antes de comprobar. Media de todos los cortes por paciente, sin clínica.

| Brazo | AUC fold0 | AUC fold3 | Media | Brecha mediana seleccionada |
|---|---:|---:|---:|---:|
| Control |0,5617|0,5590|0,5603|0,0080|
| Campo común |0,5338|0,5664|0,5501|0,0547|

Δmedia−0,0102, IC95 % pareado[-0,0492;+0,0283],4.000 bootstrap por fold/cohorte/clase. No se demuestra beneficio ni empeoramiento concluyente; no pasa el filtro prefijado y no se amplía a cinco folds. Con ambas BCE fijas tampoco hay mejora media del campo. Una brecha pequeña del control no implica aprendizaje útil.

La política principal, máxima sensibilidad con especificidad≥95 % en calibración, produjo en comprobación: control TN300/FP0/FN139/TP0; campo TN297/FP3/FN135/TP4. El candidato sigue sin sensibilidad útil. No se elige otro umbral con estas pacientes.

Las101 pruebas del piloto/informe pasaron, los37.632 pasos AdamW se verificaron sin omisiones AMP y todos los pesos fueron finitos. Test y validation privada no se cargaron; V001–V008 y las aplicaciones permanecen intactas. Los pesos y checkpoints de trabajo se conservan solo localmente, fuera de Git; configuraciones, historias, hashes, curvas y límites están en el informe. Esta comprobación es desarrollo adaptativo; no es test independiente.

## Selección por pérdida y criterio de curvas: insuficiente (25/10/2026)

CNN cinética593.521 parámetros sin cambios de arquitectura/LR/WD,20 runs,
BCE normal/ponderada, dos semillas, cinco folds previamente auditados.
Fit745–746, selección132, exterior219–220; media de todos los cortes por paciente.
Checkpoint principal por BCE interna del régimen correspondiente; elección de
pérdida por AUC interna de la pareja, políticas congeladas antes de exterior.

AUC media0,556270±0,035413 frente a solo cohorte0,549012; delta+0,007258,
IC95% pareado[-0,032640;+0,047542],2000 bootstrap fold/clase. OOF agrupada cruda
0,512731. Superar0,54 por estimación puntual no demuestra señal superior a cohorte.
La regla secundaria por máximoAUC da media0,552695 y la ponderada fija0,570209;
no se sustituye retrospectivamente el objetivo principal por el mejor secundario.

Cero de20 trayectorias cumple descenso conjunto1–14 según el diagnóstico prefijado.
Mediana de época elegida en las diez redes finales2; una llega a12. Persiste el
sobreajuste. Política principal: TN739/FP36/FN300/TP22, sensibilidad6,83%,
especificidad95,35%, F1pCR0,1158. No es una mejora suficiente ni nueva confirmación
independiente: train tiene antecedentes adaptativos. Test/privada cerrados.

106 pruebas previas;289 épocas;67.337 pasos AdamW sin omisiones; pesos y archivos
históricos verificados. Código retirado y conservado en db845626bfca69276217074256fc131c0bffbc01.
No se inicia otra tanda ni se activa o publica un modelo.
[Informe completo](../resultados/27_criterio_docente_20261025/INFORME_RESULTADOS.md).

## CNN espacial compacta por corte: no adoptada (25/10/2026)

Una variante prefijada de105.569 parámetros: PRE/EARLY/LATE y diferencias,
resolución global128, tres bloques16/32/64, cuadrícula4×4, cabeza32, un logit
por corte y media de probabilidades por paciente. Sin atención entre cortes,
clínica, EMA, label smoothing ni mixup. Entrenador idéntico al control salvo
fábrica de modelos; roles, semillas y parámetros conservados.

20 runs, BCE normal/ponderada, dos semillas, cinco folds:346 épocas y80.618
pasos AdamW sin omisiones. AUC media0,546094±0,032469 frente a cohorte0,549012
y control0,556270. Delta cohorte−0,002918, IC95%[-0,042034;+0,037720];
delta control−0,010176, IC95% secundario[-0,041030;+0,021731]. No hay
mejora demostrada ni empeoramiento concluyente. OOF agrupada cruda0,510214.

Descenso conjunto1–14:4/20 trayectorias y2/10 redes elegidas. La mejora de
algunas curvas no se traslada a AUC exterior. Política principal: TN750/FP25/
FN308/TP14, sensibilidad4,35%, especificidad96,77%, F1pCR0,0776.
El secundario bestAUC da media0,556638; su ponderada fija0,564831, sin
redefinir después el criterio principal. Train adaptativo, no confirmación
externa. Test/privada cerrados; datos, versiones y control sin cambios.

109 pruebas previas. Código retirado, conservado en main en
ea01a35ba68f396870973225733fd927ea89ab40. Pesos/checkpoints de trabajo
locales e ignorados; no versión productiva ni push ni entrenamiento adicional.
[Informe completo](../resultados/28_cnn_espacial_por_corte_20261025/INFORME_RESULTADOS.md).


## MixStyle en CNN cinética: sin mejora convincente (26/10/2026)

Investigación de [MixStyle, Zhou et al., ICLR2021](https://arxiv.org/html/2104.02008v1)
y contraste único con el control cerrado del ensayo27. CNN propia desde cero,
593.521 parámetros, PRE/EARLY/LATE PNG/255, diferencias firmadas, mismo entrenamiento
y media de probabilidades de todos los cortes por paciente. Solo se añaden durante
fit interpolaciones de estadísticas tras los dos primeros bloques: p0,5,
alpha0,1, epsilon1e-6, pareja/coeficiente compartidos entre tres vistas y escala
emparejada. Sin parámetros adicionales, pesos preentrenados ni datos externos.
Identidad en inferencia. No confundir MixStyle con mixup de imágenes/etiquetas.

20/20 ejecuciones (cinco folds, dos semillas, normal/ponderada), 342 épocas,
79,686 pasos AdamW sin omisiones; 20.6 minutos.
Checkpoint por mínima BCE interna por paciente; pérdida por AUC de pareja en
selección; cinco elecciones congeladas antes de inferir exterior.

| Métrica de desarrollo | Control equiparado | MixStyle |
|---|---:|---:|
| Media AUC por fold | 0.556270 | 0.557581 |
| AUC agrupada cruda | 0,512731 | 0.528888 |

Delta medio +0.001311, IC95% pareado
[-0.031875976136917616, 0.033473692147917664]; mejora puntual en3/5 folds.
No supera el gate de réplica (media≥0,60, delta≥0,01, límite inferiorIC>0, mejora3/5).
No se demuestra superioridad ni equivalencia; configuración descartada como
candidata. El comparador de cohorte tiene media0,549012 y su delta también incluye0.
V007 histórica sigue en0,576593 de media con otra selección interna; comparación
descriptiva. No se reabre su test histórico0,608376.

Mediana de checkpoint época8.0; brecha AUC fit−selección
0.0690 en los elegidos frente0.2530 al final.
4/10 elegidos cumplen tendencia conjunta1–14; mejores curvas
parciales no eliminan sobreajuste ni garantizan AUC. Las reglas secundarias por
AUC y BCE fija tampoco alcanzan0,60; no se cambia ganador después de ver exterior.

[Informe, métricas, curvas y verificaciones](../resultados/30_regularizacion_estilo_20261026/INFORME_RESULTADOS.md).
[Protocolo](mejoras_cnn/PROTOCOLO_REGULARIZACION_ESTILO.md).
Código/pruebas en commit local7f9b3f2f96c58e3261f344d25d52f77a6804c199, retirados del árbol activo sin nuevas ramas.
Pesos de trabajo solo locales, sin versión productiva. Archivos históricos intactos,
no se usan test/privada, no se activa modelo, no se publica ni se lanza otra tanda.
La CV es desarrollo adaptativo y no una nueva validación externa.
