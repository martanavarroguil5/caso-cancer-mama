# Cuarta tanda: descriptores de textura y cambios entre fases

Fijada el 10/10/2026 después de tres tandas sin mejora concluyente. La selección de esta hipótesis es adaptativa. No se usan pacientes reservadas para test ni validación privada.

El [estudio sobre radiómica de DCE previa al tratamiento, PMID 35636571](https://pubmed.ncbi.nlm.nih.gov/35636571/) combina texturas, información clínica y hallazgos radiológicos; comunica AUC 0,77 en su mejor modelo. El [trabajo de Frontiers in Oncology (2022)](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2022.846775/full) también explora imagen, parámetros cinéticos y marcadores moleculares. Sus cohortes, máscaras, adquisición y validación son distintos. Aquí se adapta la idea de descriptores explícitos y fusión clínica; no se replica su protocolo.

## Representación fijada

PNG PRE/EARLY/LATE y diferencias EARLY−PRE, LATE−PRE, LATE−EARLY (las diferencias se trasladan a [0,1]). Tres ventanas centradas: 256, 128 y 64 píxeles, cada una reducida a 64×64 por promedio de área. Sin segmentación aprendida y sin usar etiquetas para extraer.

Por mapa y escala: media, desviación, gradiente horizontal/vertical absoluto, energía de laplaciano, entropía de histograma de 16 bins y cuantiles aproximados 10/25/50/75/90 derivados de ese histograma; textura GLCM de 8 niveles, simétrica y sumando vecinos horizontales/verticales, con contraste, homogeneidad, energía, entropía y correlación. Por paciente se guarda media y desviación entre cortes. Todos los pacientes pesan lo mismo.

Son descriptores de los PNG normalizados, no medidas cuantitativas de perfusión, volumen o parámetros de Tofts. La resolución original y la normalización a 8 bits limitan su interpretación. No se añaden variables nuevas a la clínica original.

## Entrenamiento y selección

Se mantienen las particiones previas: fit, 15 % de selección interna y cinco folds exteriores. ANOVA univariante elige 8/32 características dentro de fit; clínica original estandarizada con indicadores de ausencia. Escalado y selección de variables solo usan fit. Se comparan tres familias:

1. Logística L2 C=0,01/0,1/1/10, seleccionada por log-loss interna.
2. SVM RBF C=0,01/0,1/1/10 y gamma=0,1/1, clases equilibradas, seleccionado por AUC interna y desempate por log-loss de sigmoid(margen).
3. Boosting: 4 hojas, mínimo 30 pacientes por hoja, L2=5, LR=0,03/0,1 y 100/250 iteraciones, seleccionado por log-loss interna.

Platt/Youden se ajustan solo con selección interna después de elegir el clasificador. Se informa también la probabilidad sin calibración adicional, etiquetando sigmoid(margen) de SVM como puntuación. Exterior no interviene en decisiones.

## Comparación

Mismo control AUC calibrado, mismas 1.097 pacientes; bootstrap pareado 2.000 réplicas, alfa 0,05/10 para las diez familias de estas cuatro tandas, sin afirmar corrección de toda la investigación histórica. Mejora interna si el intervalo superior de delta log-loss calibrada es <0, el inferior de delta AUC media ≥−0,01 y Brier no empeora. Se guardan todos los resultados y hashes, sin sustitución automática de v002.
