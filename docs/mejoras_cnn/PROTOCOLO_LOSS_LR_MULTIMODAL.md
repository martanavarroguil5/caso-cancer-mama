# Ensayo de pérdidas y tasa de aprendizaje del multimodal

Protocolo fijado el 10/10/2026 antes de entrenar. Parte de v002 y de la evidencia
de sobreajuste de sus historiales. No sustituye los pesos de esa versión ni cambia
la aplicación. El ensayo tiene su propio módulo `cancer_mama.experimento_loss`.

## Pregunta y diseño

¿Reducir el LR o seleccionar por log-loss de paciente mejora las probabilidades
en pacientes no utilizadas para elegir la época, conservando discriminación?

Se aplican los tres cambios acordados:

1. Registrar desde época 0 y en cada época BCE ponderada por corte, log-loss sin
   ponderar por paciente, Brier, AP y AUC. Fit y selección se evalúan en FP32,
   `eval()`, sin aumentos ni dropout. La pérdida de optimización con aumentos se
   guarda aparte como `train_augmented_loss`; no es la curva principal comparable.
2. Comparar LR inicial 0,0008 y 0,0002 manteniendo arquitectura, inicialización,
   datos, semillas, orden de lotes, aumentos, optimizador y demás hiperparámetros.
3. Conservar selectores independientes de máxima AUC y mínima log-loss por
   paciente, ambos calculados con mediana de cortes y ambos incluyendo época 0.

Son veinte trayectorias: dos LR × semillas 42/2026 × cinco folds exteriores.
Cada trayectoria conserva dos checkpoints, uno por criterio, sin duplicar el
entrenamiento. Se comparan cuatro candidatos. El control es LR 0,0008/AUC
reentrenado con este protocolo; la AUC histórica 0,724568 de v002 no es un
control directamente equiparable, porque utilizó otra selección de épocas.

## Separación por paciente

Se conservan los cinco folds originales como evaluación exterior. De las
pacientes de los otros cuatro folds se reserva el 15 % para selección interna,
estratificando por cohorte × pCR, con semilla 20261010 + fold. La partición es
idéntica para ambos LR y ambas semillas de entrenamiento.

Imputación, escala clínica, regresión logística de inicialización y peso positivo
de BCE se ajustan exclusivamente con fit. La selección interna decide épocas
y parada. El fold exterior se infiere solo después de congelar ambos modelos;
no decide la parada, el LR ni los checkpoints. Sus métricas no aparecen en las
curvas durante aprendizaje. La caché puede contener imágenes de todo desarrollo,
pero el loader de aprendizaje solo indexa fit.

No se leen imágenes ni se convierten etiquetas del test reservado. Tampoco se usa
la validación privada. Las pacientes de desarrollo ya se han observado en otros
ensayos: la separación nueva reduce el sesgo de selección de este entrenamiento,
pero sigue siendo desarrollo interno adaptativo, no validación externa nueva.

## Optimización y parada

Se mantiene CNN multimodal de 551.921 parámetros, PRE/EARLY/LATE, edad,
log1p(volumen), HR y HER2 e indicadores de ausencia. BatchNorm, pooling
intermedio, dropout 0,35, weight decay 0,001, AdamW, lote 64, AMP en CUDA,
flip/rotaciones conjuntas y BCE ponderada por corte. No se cambia a entrenamiento
con BCE por paciente, que ya se ensayó anteriormente.

Calendario coseno de 46 épocas y LR mínimo 0,00001 en ambos brazos. Cada selector
tiene mínimo 12 épocas y paciencia 10; mejora significativa 0,001 en su propia
métrica. Guarda toda mejora estricta, pero reinicia la paciencia respecto al
último progreso significativo para permitir acumular mejoras pequeñas.

Al agotarse la paciencia de un selector, su mejor checkpoint queda congelado.
El otro puede seguir aprendiendo y seleccionando sobre la trayectoria compartida.
La trayectoria termina cuando paran ambos o alcanza 46 épocas. Las épocas
posteriores a la parada de un selector no pueden alterar su checkpoint.

La log-loss de selección es cruda, no ponderada, por paciente y sin calibración;
no debe confundirse con la BCE ponderada usada para optimizar. Puede favorecer
otro compromiso entre ordenación y probabilidades. No se reajustan aquí Platt
ni el umbral de v002 para mejorar artificialmente el resultado del ensayo.

## Comparación fijada

Para cada candidato se promedian primero las dos semillas por corte de su fold
exterior y después se agregan cortes de cada paciente mediante mediana. Se
exigen exactamente veinte runs completos, con cobertura OOF idéntica y hashes
válidos. Se informa log-loss, Brier, AUC media de folds, AUC agrupada y AP.

Se comparan los otros tres candidatos contra LR actual/AUC. Dos mil bootstrap
pareados por paciente, estratificados por fold/cohorte/clase, semilla 20261010.
Los intervalos Bonferroni del 98,33 % contemplan las tres comparaciones. Son
condicionados a las predicciones congeladas y no recogen toda la incertidumbre
del aprendizaje o de la búsqueda histórica.

Como señal interna favorable se exige intervalo superior de delta log-loss menor
que cero, intervalo inferior de delta AUC media no inferior a -0,01 y Brier no
peor en estimación puntual. La tolerancia de AUC es una decisión de ingeniería
fijada para este ensayo, no una garantía clínica. Se presentan las cuatro
comparaciones aunque ninguna cumpla los criterios. No hay promoción automática;
los pesos, calibración, umbral y aplicación de v002 permanecen conservados.

## Reanudación y evidencia

Cada época guarda atómicamente modelo, optimizador, scheduler, AMP, generadores
aleatorios, historial, estados de selección y ambos mejores pesos. Reanudar
requiere el mismo código, datos, imágenes, configuración y entorno. Un bloqueo
de proceso impide dos escritores en la misma salida. El estado se actualiza en
`estado.json` cada época y al terminar cada run. Un fallo se registra explícitamente.

Se verifica la conservación por SHA-256 de ambas versiones archivadas y del
entrenador/app actuales al finalizar cada run. Las ejecuciones y la caché de
imágenes son locales y están excluidas de Git; código, protocolo y pruebas sí
se versionan. El comparador genera automáticamente métricas, OOF e informe.

## Ejecución en el servidor Linux

Prueba CUDA de los dos LR, un fold y una época, excluida de comparación:

```bash
.venv/bin/python -u -B -m cancer_mama.experimento_loss ejecutar --prueba
```

Matriz completa y comparación automática al terminar:

```bash
.venv/bin/python -u -B -m cancer_mama.experimento_loss ejecutar
```

El mismo comando reanuda una ejecución compatible; no sobrescribe un protocolo
diferente. La salida predeterminada es
`resultados/04_entrenamiento/loss_lr_multimodal_20261010/`.
Para repetir solo la comparación con el mismo código/entorno:

```bash
.venv/bin/python -u -B -m cancer_mama.experimento_loss comparar
```

El entrenamiento habitual sigue disponible en `cancer_mama.entrenamiento`.
