# Transferencia visual y fusión regularizada del multimodal

Protocolo fijado el 10/10/2026 antes de examinar resultados de este ensayo.

## Evidencia y adaptación

- [Scientific Reports, 2021](https://www.nature.com/articles/s41598-021-93592-z): extracción con una CNN preentrenada y clasificador SVM en I-SPY1. Apoya probar representaciones fijas con pocos pacientes. Sus mejores resultados combinan exploraciones antes y durante tratamiento; no son comparables con nuestra entrada exclusivamente previa al tratamiento.
- [AACR/Clinical Cancer Research, 2026, resumen de congreso](https://aacrjournals.org/clincancerres/article/32/4_Supplement/PS2-09-01/773092/Abstract-PS2-09-01-Optimizing-Transfer-Learning): compara ResNet50 ImageNet/RadImageNet en 1.357 RM previas al tratamiento. El resultado global comunicado es AUC 0,63 con ImageNet y 0,68 al añadir clínica. Evidencia preliminar; no demuestra que alcance nuestra AUC histórica.
- [Cancers, 2025](https://www.mdpi.com/2072-6694/17/20/3394): características de ResNet MedicalNet y clínica, con selección de variables y clasificación regularizada, en 177 pacientes con RM/TC. Apoya la estrategia general; utiliza modalidades y preentrenamiento distintos.
- [BreastDCEDL, Scientific Data](https://www.nature.com/articles/s41597-026-06589-6): volúmenes centrados en tumor y fases PRE/EARLY/LATE. El resultado de ViT difundido para HR+/HER2− corresponde a un subgrupo y no se usa como objetivo comparable para todas nuestras pacientes.
- [PyTorch: transferencia](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) y [ResNet50](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html): implementación oficial y pesos ImageNet1K V2.

Hipótesis de ingeniería: una representación visual preentrenada y congelada evita aprender millones de asociaciones desde unas mil pacientes. Clasificadores pequeños, reducción de dimensión y fusión con clínica pueden reducir el sobreajuste observado. No se presupone una mejora.

## Primera tanda fijada

ResNet50 ImageNet1K V2 totalmente congelada, incluida BatchNorm en eval, FP32. Las tres fases son los canales de una imagen. Dos vistas fijas: imagen completa y recorte central de 128×128; transformación oficial de los pesos a 224×224. No se usan etiquetas para extraer características. Promedio de características por paciente, con el mismo peso por paciente independientemente de sus cortes.

Los cinco folds exteriores y el 15 % de selección interna son exactamente los del ensayo anterior. Imputación, escalado, PCA y clasificadores se ajustan solo con fit. Ningún fold exterior selecciona hiperparámetros. No se reentrena con la selección después de elegir parámetros. ResNet se preentrenó con ImageNet, sin pacientes de BreastDCEDL.

Clínica: edad, log1p(volumen), HR, HER2, indicadores de ausencia e interacción HR×HER2. La interacción usa los valores imputados del fit; la imputación no ve selección/exterior. Se comparan tres familias:

1. **Fusión lineal:** PCA de 16/64 componentes de las características de las dos vistas; logística L2 C=0,001/0,01/0,1/1; peso relativo de clínica 1/4. 16 combinaciones.
2. **Fusión no lineal:** 16/64 componentes y clínica; boosting de árboles pequeños, 4 hojas, mínimo 30 pacientes por hoja, L2=5, LR=0,03/0,1 y 100/250 iteraciones. 8 combinaciones. Sin early stopping que vuelva a dividir datos.
3. **Fusión tardía:** logística clínica C=1, logística visual sobre 16/64 componentes C=0,001/0,01/0,1/1 y mezcla de probabilidades con peso visual 0/0,25/0,5/0,75. 32 combinaciones; el peso cero permite rechazar una contribución visual perjudicial.

Se elige dentro de cada familia por mínima log-loss de selección interna; empate resuelto por orden fijo de la rejilla. BCE/log-loss sin ponderación de clase: la prevalencia real ya está en fit. Los modelos son deterministas, semilla 20261010; no se interpreta repetir una semilla de un clasificador determinista como evidencia independiente.

Cada candidato conserva probabilidades crudas y calibradas. Para evaluación, Platt/Youden se ajustan únicamente con selección interna, igual que el control; no se usan etiquetas exteriores. Se registra un baseline exclusivamente clínico, pero no cuenta como mejora multimodal si el candidato acaba descartando la imagen.

## Comparación y condición de mejora

Control principal: multimodal original reentrenado con LR 0,0008 y selección por AUC bajo el protocolo interno anterior. Se reutilizan sus OOF y calibradores verificados. Control secundario: selección por log-loss (candidata v003). La AUC histórica 0,724568 es descriptiva, no un control equiparado.

Sobre las mismas 1.097 pacientes se informa AUC media de folds, AUC agrupada, AP, log-loss, Brier y desglose por fold/cohorte. Bootstrap pareado estratificado por fold/cohorte/clase, 2.000 réplicas, intervalos Bonferroni del 98,33 % por las tres familias. Criterio primario: límite superior de delta log-loss calibrada <0, límite inferior de delta AUC media ≥−0,01 y Brier no peor. Se exige además contribución visual en el modelo para describirlo como mejora multimodal. Se informa también comparación con v003.

Un aumento puntual se describe como señal, no como mejora confirmada. Estas OOF ya han participado en investigación adaptativa: los intervalos son condicionales a predicciones fijas y no corrigen toda la búsqueda histórica. No se abre test ni validación privada. No se cambia v002 automáticamente por ganar una comparación exploratoria.

## Continuación, trazabilidad y recursos

La primera tanda tiene tres familias, con resultados completos incluidos aunque fallen. Si falla, la siguiente hipótesis se documentará antes de entrenarla; no se prolongará una búsqueda ciega hasta conseguir una cifra favorable. Es posible que los datos disponibles no permitan una mejora demostrable.

Código en módulo independiente, protocolo, hashes de datos/PNG/pesos/fuentes, entorno, selección de parámetros y predicciones quedan conservados. Los artefactos pesados se excluyen de Git. Se protegen pesos históricos, entrenador original y aplicación. CUDA RTX 3090 para extracción; CPU para clasificadores. Instalación adicional: torchvision 0.29.0, compatible con torch 2.14.0 instalado; no se actualiza torch.
