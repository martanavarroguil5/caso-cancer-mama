# Ensayo SAM sobre el multimodal original v002

> **Cerrado el 12/10/2026, sin cumplir el criterio de mejora.** Terminadas 40/40
> ejecuciones. SAM ponderado mostró una señal pequeña frente al control, con IC
> que incluye cero y contribución visual prácticamente nula. V002 sigue activo.
> [Resultados y decisión](../EXPERIMENTOS_DESCARTADOS.md). El texto prefijado
> se conserva debajo; sus comandos son históricos. Código en `4a2c959` de `main`.
> Los hashes de ejecución corresponden al protocolo anterior a esta nota.


Fijado el 11/10/2026 antes del entrenamiento completo. Petición del usuario:
probar SAM tras descartar FiLM. Se trabaja exclusivamente en `main`; v002 sigue
activo con sus pesos, calibración y umbral originales. No hay promoción automática.

## Hipótesis

SAM (Sharpness-Aware Minimization) busca reducir la sensibilidad de la pérdida
a pequeñas perturbaciones de los pesos. Se prueba si mejora la generalización
de la CNN multimodal original sin modificar su arquitectura. No se promete
eliminar el sobreajuste ni superar a v002.

Referencias primarias consultadas:

- [Foret et al., ICLR 2021](https://arxiv.org/abs/2010.01412).
- [Implementación oficial y ejemplo rho=0,05](https://github.com/google-research/sam).
- [Gradiente SAM, mismo RNG y estado de la primera pasada](https://github.com/google-research/sam/blob/main/sam_jax/training_utils/flax_training.py).
- [PyTorch 2.14: AMP y gradientes calculados con autograd.grad](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html#gradient-penalty).

El artículo presenta evidencia en otros conjuntos y arquitecturas. El uso de
SAM + AdamW, BCE y esta CNN es una hipótesis para este problema; no reproduce
exactamente sus experimentos con SGD. No se incorpora ninguna red del artículo.

## Restricciones del enunciado y antecedentes

Página 6, apartados 7 y 8: CNN 2D propia desde cero, un logit, BCEWithLogitsLoss
normal/ponderada y documentación del optimizador, semillas, parada y GPU.
Cambiar la actualización de pesos mantiene estos requisitos. No se emplean
ResNet, EfficientNet, VGG, DenseNet, pesos preentrenados ni transfer learning.
El PDF permite particiones internas por paciente y reserva test para la
evaluación final, no para ajustar hiperparámetros. Este ensayo utiliza solo train.
PDF revisado SHA256: `1ec66c7e6402d9e9e1a4788626c3539d43247a88b7dce789cc47d520ab5b0901`.

Se mantiene el supuesto previo de disponibilidad de edad, volumen, HR y HER2;
el enunciado no garantiza su suministro en la evaluación privada. No se cambia
el contrato de inferencia ni se activa un modelo exclusivamente clínico.

El [registro de experimentos descartados](../EXPERIMENTOS_DESCARTADOS.md) no
contiene un ensayo previo de SAM. FiLM, mixup, cambios de LR/selección y BCE con
otro inicializador clínico no demostraron superar al v002 original. No se
reintroducen esas modificaciones: se prueba únicamente SAM, con controles nuevos.

## Diseño prefijado

Cuatro brazos × semillas 42/2026 × cinco folds: **40 entrenamientos desde cero**.

| Brazo | Actualización | BCE | Radio SAM |
|---|---|---|---:|
| control_weighted | AdamW | Ponderada por corte | 0 |
| sam_weighted | SAM + AdamW | Ponderada por corte | 0,05 |
| control_normal | AdamW | Normal | 0 |
| sam_normal | SAM + AdamW | Normal | 0,05 |

Hipótesis principal: `sam_weighted` frente a `control_weighted`. Los brazos con
BCE normal son secundarios y no seleccionan automáticamente la pérdida ganadora.
Un único radio 0,05, tomado como punto de partida documentado; no se busca rho
usando los resultados exteriores. SAM L2 estándar, no ASAM, sobre todos los
parámetros entrenables. Controles y candidatos tienen 551.921 parámetros.

CNN v002: ocho convoluciones, canales 24/48/96/160, BatchNorm, proyección visual
64, concatenación clínica y salida lineal. PRE/EARLY/LATE [3,256,256], misma
normalización y aumentos sincronizados. No se añade FiLM. La logística clínica
equilibrada se ajusta solo con fit y se usa como inicializador en todos los brazos,
incluidos los de BCE normal. No se cargan pesos históricos para entrenar.

AdamW LR 0,0008, weight decay 0,001, dropout 0,35, lote 64, calendario coseno
hasta LR 0,00001, máximo 46 épocas, mínimo 12, paciencia 10, min_delta 0,001.
AMP FP16 para entrenamiento, evaluación FP32 sin aumentos ni dropout, TF32
desactivado, ejecución determinista. La semilla común fija pesos iniciales,
orden de lotes, aumentos y dropout. Se conserva el número de lotes/épocas como
presupuesto; SAM calcula dos gradientes por lote y consume más tiempo.

## Actualización SAM y comprobaciones numéricas

1. Calcular el gradiente g de BCE en los pesos actuales w con el lote aumentado.
2. Evaluar el segundo gradiente en w + 0,05 g/(||g||2 + 1e-12), reutilizando
   exactamente las mismas imágenes, etiquetas y máscaras de dropout.
3. Restaurar por copia los pesos originales y actualizar AdamW una sola vez
   con el segundo gradiente. El clipping global 5 se aplica al gradiente final;
   weight decay sigue desacoplado en AdamW y no se añade al objetivo perturbado.

BatchNorm usa estadísticas del lote en ambas pasadas, pero se conservan solo
las estadísticas acumuladas de la primera, incluido num_batches_tracked.
El RNG tras la segunda pasada vuelve al estado posterior a la primera. Los
pesos y buffers se restauran también si falla la segunda pasada.

AMP: el primer gradiente se obtiene con autograd.grad sobre la pérdida escalada
y se desescala fuera del optimizador. Solo se llama unscale_ una vez por paso.
Si los gradientes desbordan en cualquiera de las pasadas, GradScaler omite la
actualización y reduce su escala; se registran los pasos omitidos. Una pérdida
no finita o una época sin actualizaciones válidas invalida la ejecución.
Se guardan por separado BCE aumentada inicial, BCE perturbada, escala AMP,
pasos efectivos/omitidos y desbordamientos de cada pasada.

Antes de lanzar: pruebas del gradiente frente a una función analítica, igualdad
rho=0 con AdamW, restauración exacta, BatchNorm/dropout, overflow AMP de ambas
pasadas, separación por paciente y reanudación idéntica. Prueba GPU de un lote
y una época por brazo, marcada explícitamente fuera de comparación.

## Selección y evaluación

Los cinco folds exteriores son los originales. En los otros cuatro folds se
reserva un 15 % interno por paciente, estratificado por cohorte × pCR, con
semilla 20261010 + fold. Mismas pacientes en los cuatro brazos y ambas semillas.
Imputación, escala clínica, inicializador y pos_weight se ajustan solo con fit.

Se elige checkpoint exclusivamente por AUC de pacientes de selección interna,
incluida la época 0. Mediana entre cortes. La paciencia usa mejora acumulada
desde la última referencia significativa; el mejor checkpoint conserva también
mejoras pequeñas. Nunca se observa el fold exterior para elegir época o parada.
El exterior se evalúa una vez fijado el checkpoint de cada ejecución.

Por brazo se promedian las dos semillas por corte y luego se aplica mediana
entre cortes de la paciente. Platt y umbral Youden se ajustan al ensemble de
selección interna de cada fold y se aplican al exterior. Se guardan ROC-AUC
sin calibrar y calibrada, media de AUC por fold, AP, log-loss, Brier, FP/FN,
sensibilidad, especificidad y resultados por semilla/cohorte/fold.

Una señal interna favorable requiere, conjuntamente:

- Incremento de AUC OOF agrupada sin calibrar de al menos 0,005 frente al control.
- Límite inferior del IC95 % pareado de esa diferencia mayor que cero.
- Log-loss y Brier calibrados no peores.
- Ninguna de las dos semillas con AUC inferior a la de su control.

IC mediante 2.000 remuestreos por paciente, pareados y estratificados por
fold/cohorte/clase, semilla 20261011. Estos intervalos no corrigen toda la búsqueda
histórica sobre train; no equivalen a validación externa.

El informe incluye aparte el **v002 original**: AUC 0,724568, pérdida con Platt
guardado 0,536656, Brier 0,179060, FP=187, FN=128. Su selección histórica y
calibración/umbral ajustados a las mismas OOF son distintos. Mejorar un control
reentrenado no autoriza afirmar que se ha superado a los pesos originales.
Ninguna métrica exterior decide un reentrenamiento automático ni una promoción.

## Ejecución, archivos y recuperación

```bash
.venv/bin/python -u -B -m cancer_mama.experimento_sam ejecutar --prueba
.venv/bin/python -u -B -m cancer_mama.experimento_sam ejecutar
```

Salida local: `resultados/04_entrenamiento/sam_multimodal_20261011/`, fuera de
Git. Lock para impedir ejecuciones simultáneas. Estado actualizado en estado.json,
log, curvas y checkpoint atómico por época con AdamW, scheduler, AMP, RNG y
mejor estado. Reanudación solo si coinciden código, protocolo, entorno y datos
(incluidos hashes de los PNG de train). Cambios exigen otra salida explícita.

Al finalizar se genera comparacion/resumen.json, informe Markdown y tablas OOF.
Se comprueban los hashes de los archivos protegidos después de cada ejecución.
V002 y los paquetes congelados permanecen intactos. Si no mejora, se documenta
el cierre en el registro de descartes y se retira este módulo, conservando su
código en los commits de `main`, sin ramas nuevas ni copias de archivo.
