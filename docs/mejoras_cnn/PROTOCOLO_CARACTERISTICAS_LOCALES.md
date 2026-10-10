# Segunda tanda: características locales y selección supervisada

Fijada el 10/10/2026 después de observar el fracaso de las tres familias ResNet50/PCA. Los resultados anteriores se conservan íntegros. Esta tanda es adaptativa y no constituye confirmación independiente.

## Motivo y referencia

En la primera tanda, PCA puede priorizar variaciones dominantes de imagen que no predicen respuesta, y el promedio espacial de la última capa puede perder estructuras locales. Es una hipótesis, no una explicación demostrada del resultado negativo.

El [estudio de I-SPY1 de Scientific Reports (2021)](https://www.nature.com/articles/s41598-021-93592-z) emplea características de la capa pool2 de AlexNet preentrenada, selección de características y SVM. Su resultado más alto necesita exploraciones durante tratamiento; aquí solo adaptamos el mecanismo de representación y clasificación, usando exclusivamente RM previa al tratamiento. Nuestra selección ANOVA es más sencilla y no reproduce su procedimiento de estabilidad.

## Diseño fijado antes de entrenar esta tanda

- Misma separación exterior/fit/selección interna que el control anterior. Mismas 1.097 pacientes, sin acceder al test.
- **alexnet_svm:** AlexNet ImageNet congelada hasta pool2; se conservan posiciones espaciales (192×13×13). Las tres fases ocupan los tres canales y se aplica la transformación oficial. Promedio de características por paciente.
- **resnet_svm:** reutiliza las características ResNet50 de la primera tanda; sustituye PCA por selección supervisada y logística por SVM.
- **alexnet_lineal:** misma representación local, selección supervisada y logística L2 como control de complejidad.
- ANOVA univariante ajustada exclusivamente con fit, 32 o 128 características. Columnas constantes reciben puntuación cero; orden de empates estable. Escalado ajustado con fit, división por raíz del número de características visuales y concatenación con clínica estandarizada.
- Clínica original: edad, log1p(volumen), HR, HER2 e indicadores de ausencia, sin interacción adicional. Una fila por paciente.
- SVM RBF C=0,01/0,1/1/10; gamma=0,1/1; clases equilibradas. Selección por máxima AUC interna, empate por mínima log-loss de sigmoid(margen), después orden fijo. Platt final se ajusta solo con selección interna.
- Logística L2 C=0,01/0,1/1/10 sin ponderación, selección por mínima log-loss interna. No se usa selección externa para elegir parámetros ni se reentrena con selección interna.
- Guardar predicciones, modelos y transformaciones de cada fold, configuraciones y hashes. Incluir todos los resultados.

## Comparación

Métricas crudas y calibradas contra los mismos controles, intervalos por bootstrap pareado de 2.000 réplicas. Para la decisión de esta tanda se usa alfa 0,05/6, contando las seis familias investigadas en ambas tandas, sin afirmar que esto corrija toda la investigación histórica. Mejora interna: intervalo superior de delta log-loss calibrada <0, intervalo inferior de delta AUC media ≥−0,01 y Brier no peor. Se conservarán señales puntuales sin calificarlas de mejora concluyente ni reemplazar v002 automáticamente.
