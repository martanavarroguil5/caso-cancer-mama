# Investigación de mejoras de la CNN para predecir pCR

Revisión del 05/10/2026 sobre el código posterior a la limpieza. La siguiente prueba recomendada es añadir `Dropout2d(0.10)` después del último bloque convolucional. La hipótesis es reducir la dependencia de mapas de características concretos y mejorar la generalización, conservando la CNN sencilla. Es una propuesta pendiente de ensayo; no se ha demostrado una mejora.

Se revisaron el enunciado docente, README, entrenador, ensayos cerrados, curvas y metadatos de desarrollo. El [diagnóstico reproducible](diagnostico.json) identifica el commit, los archivos fuente y sus SHA-256. Esta investigación no entrena modelos ni abre imágenes de test o de la validación privada.

## Condiciones de la práctica

El enunciado exige una CNN 2D propia, inicializada desde cero, entrada PRE/EARLY/LATE y salida de un logit con BCEWithLogitsLoss. Deben compararse BCE normal y ponderada. Prohíbe modelos preentrenados y transfer learning. El plan conserva los pasos 01–03 y `pipeline_datos.py`, y mantiene las tres imágenes como entrada de inferencia.

HR y HER2 se usan aquí para describir heterogeneidad y errores; el plan principal no añade entradas clínicas. El enunciado no formula una prohibición explícita de usar metadatos, pero incorporar esos predictores cambiaría la entrada de la aplicación y exigiría verificar su disponibilidad en la evaluación docente. Esa ampliación queda fuera de esta propuesta.

Los PNG ya están centrados en el entorno tumoral. El PDF docente describe recorte a 256×256 y una ventana conjunta de percentiles 1–99 para las tres fases. Volver a normalizar cada fase por separado modificaría la información relativa de realce. La publicación del conjunto original describe volúmenes NIfTI, anotaciones y diferencias de adquisición entre cohortes; no constituye una validación del rendimiento de esta versión docente en PNG. [BreastDCEDL](https://www.nature.com/articles/s41597-026-06589-6).

## Hallazgos en el proyecto

Hay 1.097 pacientes y 10.945 cortes de desarrollo. La media es 9,98 cortes por paciente, mediana 10 y rango 5–10. La prevalencia por paciente es 29,35%; la auditoría existente calcula 29,38% por corte. El peso desigual derivado del número de cortes parece pequeño en este conjunto; esta observación no resuelve todas las limitaciones de asignar una etiqueta de paciente a cada corte.

La comparación de tamaños proporciona un diagnóstico claro de sobreajuste. En los diez runs de la CNN ajustada de 551.913 parámetros, la media a la época 30 fue AUC train **0,937** y validación **0,532**. Los checkpoints seleccionados obtuvieron AUC media por run **0,611** y OOF agrupada **0,593**. Las cifras de época 30 describen modelos finales; no sustituyen las métricas de los mejores checkpoints. Reducir canales no mejoró la AUC global.

La tasa de aprendizaje 0,0003 ya se ensayó con pooling, dropout y weight decay en distintas combinaciones. En fold 0, semilla 42 y lote 16, `pool_lr_dropout_wd` obtuvo AUC **0,5967**, frente a **0,6498** de `pool_dropout_wd` con LR 0,0008. No procede presentar esa misma bajada de LR como una solución nueva.

La mejora anterior de pooling, dropout 0,35 y weight decay 0,001 fue pequeña: AUC media **0,6231 → 0,6270** en tres folds con una semilla. Se observó con lote 16. El valor actual por defecto es 64; esa evidencia no confirma automáticamente el rendimiento de la configuración con otro tamaño de lote.

Las cohortes tienen prevalencias diferentes: Duke 21,1%, I-SPY1 25,0% e I-SPY2 32,1%. En la comparación de tamaños, la CNN ajustada obtuvo AUC OOF 0,632 / 0,619 / 0,567, respectivamente. Estas cifras aconsejan analizar resultados por cohorte; no prueban por sí solas que la red utilice artefactos.

También hay heterogeneidad por subtipo. En las predicciones OOF históricas, HR+/HER2− presentó AUC **0,509** en 453 pacientes, y HR−/HER2+ **0,717** en 108. Es un análisis exploratorio de modelos ya seleccionados, con incertidumbre distinta por grupo. Sirve para localizar dificultades; no justifica entrenar cuatro CNN con menos pacientes cada una.

## Cambios propuestos por prioridad

Cada fila es un ensayo independiente. Los valores son elecciones de ingeniería para iniciar pruebas pequeñas, no valores óptimos publicados para este dataset.

| Prioridad | Propuesta | Motivo y límite |
|---|---|---|
| 1 | `Dropout2d(0.10)` tras el cuarto bloque y antes del pooling global | Regulariza mapas completos. Conserva ocho convoluciones, BatchNorm, tamaños y 551.913 parámetros. Su beneficio en pCR está pendiente de comprobar. |
| 2 | Weight decay 0,001 → 0,003, manteniendo LR 0,0008 | Aumenta una regularización que ya dio una señal favorable. Puede producir subajuste; se ensaya separada del dropout nuevo. |
| 3 | Aumentos conjuntos de traslación hasta 3% y escala 0,95–1,05 | Hipótesis de robustez a pequeñas diferencias de centrado y tamaño. Mantener inicialmente los aumentos actuales y comprobar que el tumor permanece dentro del recorte. |
| 4 | Promedio exponencial de pesos durante el entrenamiento | Hipótesis de mayor estabilidad con la misma arquitectura de inferencia. Requiere tratar correctamente las estadísticas de BatchNorm y la reanudación. |

El dropout actual actúa sobre el vector de la cabeza, después del pooling. `Dropout2d` elimina mapas completos durante aprendizaje. PyTorch explica su utilidad cuando los píxeles vecinos de las características están correlacionados. Propongo empezar por el último bloque para introducir una sola posición y evitar otra BatchNorm después del nuevo dropout. La ubicación y p=0,10 son hipótesis propias. [Documentación de Dropout2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Dropout2d.html).

El artículo de DropBlock respalda estudiar regularización estructurada en convoluciones. Sus mejoras se midieron en clasificación y detección de imágenes naturales, no en esta tarea. Dropout2d y DropBlock son métodos diferentes; se propone el primero por sencillez. [DropBlock](https://arxiv.org/abs/1810.12890).

AdamW ya desacopla el weight decay de la actualización adaptativa. Esto permite estudiarlo como otro control del sobreajuste, aunque su efecto efectivo también depende del presupuesto de actualizaciones. El artículo no prescribe 0,003 para resonancia de mama. [AdamW](https://arxiv.org/abs/1711.05101).

Torchvision documenta cómo definir traslaciones, escalado e interpolación. Esa documentación respalda la implementación, no demuestra que los rangos propuestos mejoren pCR. La misma transformación debe aplicarse a PRE/EARLY/LATE, con interpolación bilineal y sin aumentos independientes de color. La instalación actual no tiene torchvision; una prueba futura puede implementarse con PyTorch o justificar esa dependencia antes de incorporarla. [RandomAffine](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomAffine.html).

PyTorch proporciona `AveragedModel` para EMA y SWA. EMA suaviza la trayectoria de pesos; no garantiza evitar memorizar el desarrollo. Una prueba futura debe fijar la frecuencia y el decaimiento y recalcular las estadísticas de BatchNorm usando solo las pacientes de aprendizaje. El loader actual devuelve diccionarios, por lo que `update_bn` necesita un adaptador. Los resultados de SWA no demuestran automáticamente el efecto de EMA con AdamW en este proyecto. [AveragedModel](https://docs.pytorch.org/docs/2.14/generated/torch.optim.swa_utils.AveragedModel.html), [artículo de SWA](https://arxiv.org/abs/1803.05407).

El tamaño de lote 16 frente a 64 merece una comparación específica si el primer ensayo es inconcluyente. Mantener épocas iguales cambia el número de pasos del optimizador; mantener pasos iguales cambia las pasadas por los datos. Hay que declarar qué presupuesto se iguala. Es una cuestión distinta de la composición de lotes pareados ya descartada.

## Primer ensayo propuesto

Comparar `pool_dropout_wd` con la misma configuración más `Dropout2d(0.10)` en la posición indicada. Mantener canales 24/48/96/160, cabeza 64, BatchNorm, dropout de cabeza 0,35, AdamW, LR 0,0008, weight decay 0,001, lote 64, aumentos actuales y máximo 30 épocas. La elección de lote 64 fija una referencia actual; no declara que sea mejor que 16.

El protocolo necesita distinguir selección de checkpoints de evaluación del candidato. Conservar los cinco folds originales como exteriores; dentro de los cuatro folds de aprendizaje, reservar 15% de pacientes para selección, estratificando por cohorte y pCR. Aprender con el 85% restante. Ambos brazos deben usar las mismas pacientes, inicialización de parámetros, orden de datos y aumentos. Este diseño usa aproximadamente 68% del desarrollo para ajustar pesos, frente al 56% de los últimos ensayos con partición interna 70/15/15.

Seleccionar checkpoints con AUC del subconjunto interno; paciencia 10, mínimo 12 épocas y min_delta 0,001. El criterio principal será AUC cruda media de los cinco folds exteriores, promediando las predicciones de las semillas 42 y 2026 por paciente. El fold exterior no elegirá épocas, hiperparámetros, calibración ni umbral. AP, Brier y métricas a umbral 0,5 serán secundarias; un futuro umbral se ajustará con datos internos.

Una revisión inicial en folds 0 y 3, con ambas semillas y BCE ponderada, supone ocho runs contando control y candidato. Solo sirve para detectar fallos o un deterioro evidente. La comparación completa requiere los cinco folds y ambas semillas; comprobar después también BCE normal para cumplir el diseño de la práctica. Completar ambos brazos, dos pérdidas, dos semillas y cinco folds equivale a cuarenta runs en total, contando los iniciales.

Antes de ejecutar, fijar una mejora práctica objetivo de **0,01 de AUC**, revisar consistencia entre folds y semillas y calcular un intervalo pareado por paciente. No adoptar el cambio por una ganancia aislada en F1 o en un solo fold. Un intervalo que incluya cero deja la superioridad sin resolver. El bootstrap de predicciones condiciona a esos modelos y no recoge toda la incertidumbre del entrenamiento.

Los controles deben reentrenarse bajo el protocolo elegido. Las AUC de ensayos antiguos con otras particiones y lotes no son un comparador válido. Aunque separar selección y evaluación reduce el optimismo dentro del nuevo ensayo, las 1.097 pacientes ya se observaron en desarrollo: seguirá siendo evaluación interna adaptativa. La validación privada del profesor y el test histórico permanecen cerrados. [TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378).

## Gestión del código y decisiones

La investigación deja el entrenador sin cambios. El ensayo futuro debe desarrollarse de forma aislada, con una configuración fija y un registro del protocolo, versiones, semillas, pesos y predicciones. Al terminar, se incorpora al entrenador únicamente si se adopta. Si empeora o queda inconcluyente, se retira el código candidato y se documentan resultados y decisión en `EXPERIMENTOS_DESCARTADOS.md`.

El modelo histórico, su calibración y umbral siguen siendo los del informe. GroupNorm, BCE por paciente y la reducción de canales ya ensayada permanecen descartados. Una CNN más grande no es la primera prioridad cuando la actual alcanza AUC de entrenamiento cercana a 0,94 y muestra este desfase de validación. Modificar umbrales puede cambiar F1 y sensibilidad, pero no mejora la ordenación que mide la AUC.

Esta propuesta busca un avance medible con un cambio pequeño. Ninguna de las fuentes permite prometer una AUC concreta para la CNN del proyecto.
