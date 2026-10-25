# CNN espacial compacta por corte

25/10/2026. El usuario autoriza una única variante espacial, comprobación de
funcionamiento y comparación completa con el ensayo de selección por pérdida.
Solo main local, sin push ni cambio de modelo activo.

## Hipótesis y arquitectura prefijada

Una red de menor capacidad que conserve una cuadrícula espacial puede mejorar
el equilibrio entre aprendizaje y sobreajuste. Se prueba el conjunto de esta
arquitectura, sin atribuir un eventual resultado a un componente aislado.

- PNG originales PRE/EARLY/LATE,3×256×256, divididos entre255.
- Vista global remuestreada a128×128 por bilinear, sin seleccionar regiones.
- Seis canales explícitos: PRE, EARLY, LATE, EARLY−PRE, LATE−PRE, EARLY−LATE.
  No se interpretan las fases como colores RGB.
- Tres bloques de dos convoluciones3×3, canales16/32/64. La primera convolución
  de cada bloque tiene stride2; BatchNorm y ReLU tras cada convolución.
- Dropout espacial0,10; mapas64×16×16, media local4×4/stride4 para conservar
  una cuadrícula64×4×4. Es exactamente equivalente al pooling adaptativo4×4
  para estas dimensiones fijas, con backward CUDA determinista.
- Cabeza1024→32→1, ReLU y dropout0,35. Un logit por corte.
- **105.569 parámetros**, desde cero, frente a593.521 del control.
- La probabilidad de paciente es la media de todos sus cortes, fuera de la red.
  No hay atención, max entre cortes, clínica, radiómica, EMA ni label smoothing.

El piloto espacial histórico mezclaba cortes dentro de la red y usaba EMA.
Sus AUC internas0,595/0,611 no validan este candidato. Se adapta únicamente la
idea de conservar información espacial con capacidad reducida.

## Comparación y entrenamiento sin otros cambios

20 ejecuciones: cinco folds originales × semillas42/2026 × BCE normal/ponderada.
Los roles son idénticos al control cerrado de resultados/27:745–746 fit,
132 selección y219–220 exterior. Todas las pacientes aparecen una vez en OOF.
Se conservan todos los PNG y las incidencias auditadas, sin exclusiones inferidas.

LR0,0003, WD0,01, lote32, coseno a1e−5, máximo40 épocas, observar al menos14,
paciencia8, mejora mínimaBCE0,001, clip5, AMP y determinismo estricto. Hflip y
rot90 compartidos entre fases; mismas semillas. BCEWithLogitsLoss por corte,
peso1/n_cortes y normalizador constante; pos_weight=N0/N1 por pacientes de fit.
No se usa la composición de cada lote para redefinir el peso de una paciente.

Checkpoint principal: mínima BCE por paciente de selección, con el pos_weight
correspondiente. Fit/selección se miden en eval, sin aumentos/dropout, después
de promediar probabilidades. Un checkpoint anterior a14 sigue siendo elegible.
La BCE normal y la ponderada no se comparan por magnitud: se elige el régimen
por AUC de selección de la pareja de semillas, con empate a favor de normal.
La regla bestAUC de las mismas trayectorias es un diagnóstico secundario.

Se congelan las cinco elecciones y políticas antes de inferir cada fold exterior
con su pareja. Selection comparte checkpoint, pérdida, calibración Platt no
negativa y umbrales (principal especificidad≥95%, secundario macroF1); sus
métricas son optimistas. Cada exterior evalúa el procedimiento completo.

El código de train_run se contrasta por AST con el commit db845626bfca69276217074256fc131c0bffbc01:
solo cambia la fábrica de modelos. Los hiperparámetros se comparan también con
el protocolo del control. No se reentrena ni se modifica ese control.

## Métricas y límites

Principal: media±DE de AUC cruda por paciente de cinco folds. Comparador
principal: solo cohorte, prevalencias aprendidas en fit, media0,549012. Se
informa si supera puntualmente0,54 y la referencia real, junto con IC95%
pareado frente a cohorte. Se compara secundariamente con el control0,556270,
por las mismas pacientes. Bootstrap2000, estratificado fold/clase, condicionado
a estos modelos; no representa toda variabilidad de entrenamiento ni selección.
No se presentan varias pruebas secundarias como una confirmación conjunta.

Curvas sin suavizar, tendencias1–14, época elegida y brecha fit−selección.
Describir descenso conjunto no garantiza discriminación: puede ser ajuste de
prevalencia. AUC agrupada cruda/calibrada, normal/ponderada fijas, por cohorte
y fold, AP, logloss/Brier, sensibilidad/especificidad, precisión/F1 y matrices
se conservan como resultados secundarios, sin sustituir el objetivo principal.

Test y validación privada permanecen cerrados. El train tiene un historial
adaptativo; esta CV no constituye una confirmación externa nueva ni garantiza
0,60. No se cambia ningún parámetro/código de una ejecución sana. No se
retoca el candidato ni se lanza otra arquitectura después de observar exterior.
No se activa modelo, no se entrena refit final y no se hace push.

## Ejecución

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -B -m cancer_mama.cnn_espacial comprobar
.venv/bin/python -B -m cancer_mama.cnn_espacial preparar
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -u -B -m cancer_mama.cnn_espacial entrenar
```

La primera comprobación sintética detectó que AdaptiveAvgPool2d no disponía de
backward CUDA determinista. Antes de usar pacientes o congelar el protocolo se
sustituyó por su media fija equivalente; la igualdad de salidas se comprueba en
tests. Sin desactivar determinismo. El test sintético corregido reduce BCE
0,690557→0,013537 en60 pasos con LR/WD de producción y precisión1,0, sin imágenes
de pacientes. Es comprobación funcional, no métrica de generalización.

109 pruebas previas. Salida resultados/28_cnn_espacial_por_corte_20261025/.
Pesos/checkpoints de trabajo solo locales e ignorados en runs/. Al terminar se
conservan resultados, curvas, configuraciones, fuentes/hashes y se decide con el
usuario; no se publica ni se crea una versión productiva automáticamente.
