# Ensayo FiLM sobre el multimodal v002

> **Cerrado el 11/10/2026, sin mejora.** Completadas 40/40 ejecuciones.
> Se conserva v002 activo. Véase el [resultado y decisión](../EXPERIMENTOS_DESCARTADOS.md).
> El texto prefijado se conserva a continuación. Sus comandos son históricos:
> el código experimental se retiró del árbol actual y está en el commit `9e2aa93`
> de `main`. Los hashes del protocolo de ejecución corresponden a aquel estado,
> anterior a esta nota de cierre.


Fijado el 11/10/2026 antes del entrenamiento. Petición: probar la fusión
condicionada, sin SAM ni un ensayo nuevo de modelo exclusivamente clínico.
Todo el código se mantiene en `main`. V002 continúa activo y sus archivos
congelados, calibración y umbral no se modifican.

## Hipótesis y funcionamiento

V002 concatena 64 características visuales con ocho entradas clínicas (edad,
log1p(volumen), HR, HER2 y cuatro indicadores de ausencia) antes de una capa
lineal. Su logit es aditivo: la contribución visual no depende directamente
del contexto clínico de esa paciente.

FiLM añade una transformación afín condicionada sobre las 64 características:

`visual_modulada = (1 + gamma(clinica)) * visual + beta(clinica)`

Una capa lineal propia de ocho entradas y 128 salidas calcula gamma y beta.
Se inicializan sus pesos y sesgos a cero: gamma=0, beta=0 y FiLM comienza como
identidad. La rama clínica directa original se conserva y sigue entrenándose.
La nueva vía multiplicativa permite interacciones imagen-clínica. Beta es lineal
en la clínica; no se añade una red clínica no lineal independiente.

Posición: después de la proyección visual de 64 dimensiones y su dropout,
antes de concatenar con clínica y producir el logit. No se modifican las ocho
convoluciones, pooling, BatchNorm, imágenes, aumentos ni inicialización común.
La construcción del nuevo generador conserva el estado RNG del control.

Parámetros: 551.921 en control y 553.073 con FiLM, 1.152 adicionales (~0,21%).
Ningún peso procede de redes preentrenadas ni de los checkpoints históricos:
la CNN se vuelve a entrenar desde cero y la logística clínica se ajusta solo
en pacientes de fit, como en la receta original.

Referencia: [Perez et al., FiLM, AAAI 2018](https://arxiv.org/abs/1709.07871).
El artículo describe el mecanismo; no demuestra una mejora en este dataset ni
que desaparezca el sobreajuste. La posición y el tamaño elegidos son hipótesis
de ingeniería. FiLM no transforma los PNG ni produce explicaciones clínicas.

## Restricciones del PDF

El apartado 7 exige CNN 2D propia desde cero, un logit, BCEWithLogitsLoss y
comparación normal/ponderada (pos_weight=N0/N1). Permite justificar arquitectura
y optimizador; la pequeña capa de modulación propia es compatible con esos
requisitos. No se usan ResNet, VGG, EfficientNet, DenseNet ni transferencia.
Se conservan PRE/EARLY/LATE alineadas, entrada [3,256,256], normalización de v002
y aumentos geométricos sincronizados. La inferencia sigue admitiendo un corte.

El PDF no garantiza el suministro de variables clínicas en la evaluación
privada. Se mantiene el supuesto previo del proyecto de que estarán disponibles;
esta limitación afecta también a v002. El test y la validación privada no se
utilizan para elegir el modelo.

## Comparación: un cambio de arquitectura

Cuatro brazos, semillas 42/2026 y cinco folds: 40 entrenamientos.

| Brazo | Arquitectura | BCE |
|---|---|---|
| control_weighted | CNN v002 reentrenada | Ponderada |
| film_weighted | Misma CNN + FiLM | Ponderada |
| control_normal | CNN v002 reentrenada | Normal |
| film_normal | Misma CNN + FiLM | Normal |

La hipótesis principal prefijada es film_weighted frente a control_weighted.
La comparación normal/ponderada cumple el enunciado; el brazo normal es
secundario y no se seleccionará automáticamente la pérdida con mejor exterior.
La inicialización logística equilibrada de v002 se conserva en ambos tipos de
BCE; no se vuelve a introducir el inicializador distinto de v004.

Se mantienen los cinco folds exteriores. Dentro de los cuatro restantes se
reserva 15% de pacientes para selección, estratificado por cohorte y pCR,
semilla 20261010+fold, como en los ensayos anteriores. Todos los brazos y
semillas comparten las particiones. Imputación, escala, inicialización clínica
y pos_weight se ajustan únicamente en fit. Los cortes de una paciente nunca
cruzan particiones. El exterior se predice después de congelar cada checkpoint.

AdamW, LR 0,0008, weight decay 0,001, dropout 0,35, batch 64, AMP en CUDA,
coseno de 46 épocas hasta LR 0,00001; mínimo 12 épocas y paciencia 10.
Selección por AUC de paciente interna, mediana de cortes, incluyendo época 0.
Toda mejora estricta se guarda; min_delta 0,001 controla la paciencia acumulada.
La arquitectura y optimización comunes arrancan de la misma semilla; el orden
de muestras, generadores de aumentos e inicialización común están equiparados.

Las curvas incluyen época 0, BCE por corte, log-loss, Brier y AUC por paciente.
Fit y selección se miden en eval/FP32 sin aumentos ni dropout. La pérdida de
optimización aumentada se registra aparte. Se guardan tiempos y memoria GPU.

## Análisis prefijado

Promedio de dos semillas por corte, después mediana por paciente. En cada fold,
Platt y Youden se ajustan a las predicciones del ensemble sobre selección
interna; se aplican congelados al exterior. Las métricas exteriores no intervienen
en épocas, calibración, umbral, combinación de semillas o configuración.

Se muestran AUC cruda agrupada y media de folds, AP, log-loss y Brier calibrados,
FP/FN y demás métricas al umbral interno. Se desglosa por semilla, fold y cohorte.
La AUC cruda es la principal; no se usan cambios de umbral para afirmar mejora.

Criterio de señal interna favorable, fijado antes de entrenar:

- Diferencia de AUC OOF cruda >=0,005 frente al control ponderado equiparado.
- Límite inferior de su IC95% pareado >0.
- Log-loss y Brier calibrados no peores en estimación puntual.
- Ninguna de las dos semillas empeora la AUC frente a su control correspondiente.

Bootstrap: 2.000 repeticiones, semilla 20261011, estratos fold/cohorte/clase,
paciente como unidad. Hay una sola hipótesis principal. Los intervalos son
condicionados a modelos ya elegidos y no corrigen toda la búsqueda histórica.
Las pacientes de desarrollo ya se han observado: sigue siendo evidencia interna
adaptativa, no validación externa ni una confirmación independiente nueva.

Además se muestra explícitamente el V002 ORIGINAL guardado (AUC 0,724568 y
log-loss con su calibración 0,536656), usando sus mismas pacientes OOF. Su
protocolo histórico de selección y calibración es distinto: esa tabla es
descriptiva y se separa del control reentrenado. Mejorar este control no bastará
para declarar que se ha superado al original. No hay promoción automática.

## Ejecución y conservación

```bash
.venv/bin/python -B -m unittest discover -s tests -v
.venv/bin/python -u -B -m cancer_mama.experimento_film ejecutar --prueba
.venv/bin/python -u -B -m cancer_mama.experimento_film ejecutar
```

El mismo comando reanuda por época. Se fijan hashes de código, protocolo, datos,
PNG de train, entorno y versiones protegidas. Cambiar cualquiera invalida la
reanudación en esa salida. Se guardan modelo, optimizador, scheduler, AMP, RNG,
particiones, predicciones internas/OOF y curvas; un lock evita dos escritores.

Salida local ignorada: `resultados/04_entrenamiento/film_multimodal_20261011/`.
La prueba de humo usa cuatro brazos con un lote y una época y queda excluida
de comparación. Al terminar, se genera `comparacion/INFORME_RESULTADOS.md` y
`resumen.json` automáticamente. Se verifican los 40 runs y su integridad.

Si el ensayo se descarta, se documenta en `docs/EXPERIMENTOS_DESCARTADOS.md` y
se retira el código experimental del árbol actual; se conserva en commits de
`main`, sin nuevas ramas ni copias duplicadas. Una versión futura de modelo
solo se publica tras revisar la evidencia frente a la referencia correcta.
