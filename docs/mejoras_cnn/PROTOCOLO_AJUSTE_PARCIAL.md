# Tercera tanda: adaptar la representación visual preservando clínica y preentrenamiento

Fijada el 10/10/2026 tras dos tandas sin mejora sobre el control. Se conserva toda la evidencia negativa y se reconoce la selección adaptativa de esta hipótesis.

## Fundamento

El [estudio multicéntrico presentado en AACR](https://aacrjournals.org/clincancerres/article/32/4_Supplement/PS2-09-01/773092/Abstract-PS2-09-01-Optimizing-Transfer-Learning) ajusta parcialmente una ResNet preentrenada para RM de mama previa a tratamiento. El trabajo de [Li, Grandvalet y Davoine, ICML 2018](https://proceedings.mlr.press/v80/li18a.html) propone L2-SP: penalizar la distancia a los pesos preentrenados en lugar de solo acercarlos a cero. No es evidencia de eficacia de nuestra configuración exacta.

Las seis familias congeladas anteriores no mejoraron al control. Ahora se permite adaptar el último bloque de ResNet18 ImageNet1K V1. La elección de ResNet18, el tamaño de la corrección y los hiperparámetros son decisiones de ingeniería fijadas antes del entrenamiento.

## Modelo y aprendizaje

- ResNet18: prefijo hasta layer3 congelado, incluida BatchNorm. Se cachean activaciones de imágenes originales y horizontalmente reflejadas, con transformación oficial ImageNet, FP32. Ningún ajuste ni etiqueta se usa para el prefijo.
- Layer4 se inicializa desde ImageNet y se entrena con LR 0,00001; BatchNorm permanece en eval. L2-SP: 0,005 × suma de cuadrados de la diferencia de sus parámetros respecto a ImageNet; sin weight decay adicional en ese bloque.
- Pooling espacial medio; promedio de embeddings por paciente; cabeza 512→32→1 con ReLU y dropout 0,5, LR 0,001 y weight decay 0,01. Última capa inicializada a cero.
- Rama clínica fija: logística L2 C=1 sin equilibrio de clases, ajustada exclusivamente con pacientes de fit, con las cuatro variables originales y sus indicadores de ausencia. La imagen añade como máximo ±0,5 al logit clínico, mediante 0,5×tanh(cabeza). Ese límite busca evitar que una señal visual débil deshaga una predicción clínica razonable.
- Entrenamiento por paciente, BCE sin ponderar; lote 16 pacientes, cuatro cortes aleatorios por paciente (con reposición solo si faltan cortes), transformación horizontal compartida dentro del paciente. Cada paciente pesa lo mismo. Inferencia: todos los cortes disponibles, sin reflejo ni dropout.
- AdamW, LR constante, hasta 24 épocas, mínimo 8, paciencia 6, min_delta 0,001 en log-loss de selección interna. Época 0 incluida y toda mejora estricta conservada. Semillas 42/2026, cinco folds exteriores. Diez trayectorias.
- Misma separación fit/selección/exterior que antes. Escalado e imputación clínica ajustados únicamente en fit. Exterior inferido tras congelar cada checkpoint.
- Pérdidas de fit y selección medidas en eval, sin dropout ni aumentos y usando todos los cortes. Pérdida optimizada y penalización L2-SP guardadas aparte.

## Evaluación y conservación

Se promedian las dos semillas por paciente antes de calibrar. Platt/Youden se ajustan solo sobre pacientes de selección interna. Se informa AUC, AP, log-loss, Brier, curvas y desglose por fold. Comparación pareada con control AUC y candidata log-loss. Bootstrap de 2.000 réplicas e intervalo alfa 0,05/7 para las siete familias de estas tres tandas. Criterio: límite superior de delta log-loss calibrada <0, límite inferior de delta AUC media ≥−0,01 y Brier no peor.

Las métricas son de desarrollo interno adaptativo, no confirmación externa. Test y validación privada permanecen reservados. v002 no se sustituye automáticamente. Código, configuraciones, estados de optimizador/RNG, checkpoints, hashes y predicciones se conservan; caché y runs pesados no entran en Git.
