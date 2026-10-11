# Comprobación de estabilidad de BCE normal

La búsqueda anterior encontró una señal favorable para BCE normal, con LR 0,0008,
selección por log-loss interna, inicializador clínico no ponderado y sin Platt
posterior: log-loss 0,546255 frente a 0,561517 del control ponderado elegido por
AUC; AUC media 0,710686 frente a 0,704684. El IC pareado ajustado de delta log-loss
fue [−0,028157; −0,001348]. La subida de AUC no fue concluyente. Mixup no superó
el criterio fijado. Siete de diez checkpoints de BCE normal eligieron época 0:
la señal no demuestra que se haya resuelto el sobreajuste de la rama visual.

Antes de entrenar esta confirmación se fijan estas decisiones:

- Semillas nuevas 73 y 1701, cinco folds, exactamente la misma separación
  fit/selección/exterior por paciente. Diez entrenamientos nuevos de candidata y
  diez de control. No se seleccionan nuevas configuraciones ni se ensaya FiLM.
- Candidata: ejecutar el código inmutable `experimento_bce_mixup.run_one` en el
  brazo `bce_normal`. Arquitectura, inicialización, LR, regularización, tamaño de
  lote, coseno, paciencia, selección y agregación se mantienen.
- Control: ejecutar el código inmutable `experimento_loss.run_one` con LR 0,0008.
  Se conservan los checkpoints de selección AUC y log-loss; el primero sigue
  siendo el comparador principal. Pos_weight=N0/N1; Platt se ajusta solo con la
  selección interna después de elegir los checkpoints.
- Promediar las dos semillas por corte y después mediana por paciente. El umbral
  de cada fold se obtiene solo de la selección interna. BCE normal conserva
  probabilidades crudas; no se escoge calibración por el resultado exterior.
- Repetir el criterio principal: límite superior del IC de delta log-loss < 0,
  límite inferior de delta AUC media ≥ −0,01 y Brier no mayor; 2.000 bootstrap
  pareados estratificados por fold/cohorte/clase, alpha=0,05/12, como en la búsqueda.
- Además, exigir menor log-loss en cada una de las dos semillas respecto a su
  control de la misma semilla, y delta AUC media ≥ −0,01 en ambas. La calibración
  individual del control también usa exclusivamente su selección interna.

Es una comprobación de estabilidad frente a inicialización/optimización, usando
las mismas pacientes de desarrollo. No es validación externa ni un conjunto nuevo
independiente. No elimina el sesgo de la búsqueda adaptativa. No se accede a
imágenes del test reservado ni a validación privada. No se cambia v002 durante
la confirmación y ningún script promueve el candidato automáticamente. Si se
confirma la señal, el paquete candidato conservará los diez pesos de las semillas
originales 42/2026; las nuevas semillas comprobarán estabilidad, sin escoger después
el par que tenga mejores métricas.

Se verifican hashes de fuentes, particiones, PNG, caché y resultados originales.
Se conserva el estado de entrenamiento reanudable, entorno, curvas, tiempos,
checkpoints y predicciones de cada semilla. El comparador rechaza pruebas y
resultados incompletos, desalineados o modificados.

```bash
.venv/bin/python -B -m cancer_mama.confirma_bce_normal --prueba
.venv/bin/python -u -B -m cancer_mama.confirma_bce_normal
```
