# Protocolo del candidato multimodal clínico

Fecha de fijación: 07/10/2026. El candidato se prepara antes de entrenarlo y no
abre de nuevo el test histórico ni la validación privada. Parte de la hipótesis de
que edad, volumen tumoral, HR y HER2 estarán disponibles para cada muestra de la
evaluación privada. Si el profesorado no las proporciona, el candidato no es
desplegable bajo las condiciones de la defensa y no se adoptará.

## Motivación

La CNN de imagen presenta sobreajuste y generalización limitada. El baseline
exacto que inicializa la rama clínica, evaluado sin usar test, obtuvo AUC media
0,707584 y AUC OOF agrupada 0,707073 sobre edad, `log1p(tum_vol)`, HR y HER2.
Los resúmenes visuales agregados quedaron alrededor de 0,55 y no mejoraron el
modelo clínico. Por ello no se sigue aumentando la capacidad de la CNN: se prueba
si una rama clínica pequeña aporta la señal biológica que falta en la imagen.

| Fold | Pacientes train | Pacientes validación | ROC-AUC | AP |
|---:|---:|---:|---:|---:|
| 0 | 878 | 219 | 0,758165 | 0,543922 |
| 1 | 878 | 219 | 0,711629 | 0,465642 |
| 2 | 877 | 220 | 0,677019 | 0,503201 |
| 3 | 877 | 220 | 0,715126 | 0,553873 |
| 4 | 878 | 219 | 0,675981 | 0,485999 |

Estas cifras son el suelo reproducible del candidato, no su resultado final. La
variación entre folds impide afirmar todavía que el modelo sea robusto.

## Candidato y control

- Candidato: `pool_dropout_wd_clinical`, con la CNN 2D desde cero conservada,
  pooling intermedio, dropout 0,35 y weight decay 0,001.
- Entrada de imagen: PRE, EARLY y LATE, sin pesos preentrenados.
- Entrada clínica: edad, `log1p(tum_vol)`, HR y HER2, más cuatro indicadores
  fijos de ausencia.
- Fusión: representación visual de 64 dimensiones concatenada con las ocho
  entradas clínicas; una capa lineal produce un único logit.
- Comparación obligatoria: BCE normal frente a BCE ponderada por cortes.
- Referencia externa al ensayo: el baseline clínico OOF de 0,707073. No se usa
  para elegir checkpoints del candidato.

La imputación por mediana y la estandarización se ajustan con una fila por
paciente y exclusivamente dentro del train de cada fold. Las estadísticas se
guardan en el checkpoint de ese fold. Los valores ausentes se imputan y activan
su indicador; no se convierten silenciosamente en cero.

La capa final se inicializa con una regresión logística ponderada ajustada sobre
esas mismas pacientes del train; los pesos de la representación visual empiezan
en cero. Antes de entrenar la CNN se evalúa y guarda esta época 0. La selección
por AUC puede conservarla si el entrenamiento conjunto no mejora el fold, por lo
que la optimización de imagen no puede degradar el baseline clínico.

## Evaluación fijada

Se ejecutarán cinco folds, semillas 42 y 2026 y ambas pérdidas: veinte jobs. El
checkpoint se selecciona por AUC de paciente del fold correspondiente, igual que
en el entrenador existente. Para cada pérdida se promedian primero las dos
semillas por corte y después se comparan media, máximo y mediana por paciente.

El criterio principal es la AUC cruda media de los cinco folds del ensemble de
dos semillas, seguida por la AUC OOF agrupada en caso de empate. El comparador
guarda las cinco AUC, su media y su mínimo. Para adoptar el candidato se exige:

- AUC media mayor o igual que 0,70.
- AUC OOF agrupada mayor o igual que 0,70.
- AP superior a la prevalencia y ausencia de un deterioro extremo en algún fold.
- Preprocesado reproducible en inferencia y disponibilidad confirmada de las
  cuatro variables en la prueba privada.

F1, sensibilidad, especificidad, Brier, calibración y umbral son secundarios.
Cambiar el umbral no puede utilizarse para afirmar que mejoró la AUC.

## Ejecución

```powershell
.venv\Scripts\python.exe -B -m cancer_mama.entrenamiento entrenar `
  --configuraciones pool_dropout_wd_clinical `
  --perdidas normal ponderada --semillas 42 2026 --folds 0 1 2 3 4 `
  --epocas 46 --lote 64 --dispositivo cuda `
  --salida resultados/04_entrenamiento/multimodal_clinico_20261007

.venv\Scripts\python.exe -B -m cancer_mama.entrenamiento comparar `
  --salida resultados/04_entrenamiento/multimodal_clinico_20261007
```

Antes de la matriz completa se permite un humo con `--prueba`, que no es
seleccionable. No se ejecutará `predecir` contra test ni se modificará el modelo
histórico. Si no se cumplen los criterios, el candidato se documentará como no
adoptado y se retirará del entrenador activo.
