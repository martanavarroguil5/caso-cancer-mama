# Piloto de reducción de cortes — 21/10/2026

## Pregunta

¿Entrenar con menos cortes redundantes mejora la AUC exterior por paciente?
Tres brazos: hasta 10 (todos los disponibles), 3 y 1 corte por paciente en fit.
Cada corte conserva PRE/EARLY/LATE. No se utilizan etiquetas para seleccionarlo.

## Selección fijada antes del entrenamiento

No hay máscaras verificadas ni mapa fiable entre slice_index y coordenadas del tumor.
Se selecciona representatividad y diversidad, no área tumoral ni relevancia clínica.
Los PNG se reducen a 32×32 y se concatenan las tres fases, con centrado y escala
conjuntos por corte. El corte único es el medoide (mínima suma de distancias
cuadráticas). Para tres se parte del mismo medoide y se añade el corte de máxima
distancia al seleccionado más próximo, con separación mínima de rangos axiales
floor(n/3), relajada si no hay candidatos. Empates: orden axial y sample_id.
Se usan únicamente los cortes disponibles; no se reconstruye el volumen original.
La regla es fija, por paciente, y no se aprende en otras pacientes ni usa pCR.

## Comparación controlada

- CNN temporal simple anti-overfit, una escala 256, sin clínica.
- BCE ponderada; peso positivo calculado por pacientes de fit (idéntico entre brazos).
- Lotes de seis pacientes y contribución igual por paciente en todos los brazos.
- Mismos parámetros, aumentos, folds 0/3, selección interna e inicializaciones 42/2026.
- Máximo 15 épocas, mínimo 4, paciencia 4. Doce ejecuciones nuevas; no reutilizar
  como control el piloto anterior porque usaba peso por corte.
- Solo fit pierde cortes: selección interna y exterior conservan todos los disponibles.
  Así se mantiene constante la agregación de inferencia y se estudia el entrenamiento.
- Igual número de lotes de pacientes por época, distinto número de imágenes por lote.
  BatchNorm y la variabilidad del gradiente pueden verse afectadas: son parte del efecto
  de reducir cortes, no una intervención exclusiva sobre correlación.
- Split interno por cohorte y etiqueta antes de filtrar cortes, conservando pacientes.
- Checkpoint por AUC interna; exterior consultado tras selección. Test cerrado.

## Resultados previstos

Manifiesto seleccion.csv, preparación y firma, correlación media por pares de
descriptores por paciente (no se define para k=1), historias por época, curvas por
run, AUC exterior por fold, ROC y matrices a umbral 0,5. La reducción de correlación
no se presupone: se mide y puede no producirse. Correlación de píxeles no es
independencia estadística ni valor predictivo.

Métrica principal: media de AUC exterior de ambos folds, ensemble de dos semillas.
Filtro exploratorio prefijado frente a 10: mejora >=0,01, ningún fold peor >0,02,
brecha mediana del checkpoint no mayor. Se reportan ambos candidatos y todos
los resultados; dos comparaciones y desarrollo adaptativo impiden afirmaciones
confirmatorias. No abre test ni cambia el modelo activo.

## Ejecución

En Windows con GPU: scripts/piloto_seleccion_cortes_gpu.ps1.
En Linux: `bash scripts/piloto_seleccion_cortes_gpu.sh` (comprueba CUDA,
prepara la selección, verifica los tres brazos y ejecuta la matriz y comparación).
Opcionalmente `WORKERS=2 bash scripts/piloto_seleccion_cortes_gpu.sh`.

Manualmente:

```bash
python -m cancer_mama.seleccion_cortes preparar
python -m cancer_mama.seleccion_cortes entrenar --dispositivo cuda --workers 4
python -m cancer_mama.seleccion_cortes comparar
```

Prueba técnica CPU, no válida para seleccionar modelos:

```bash
python -m cancer_mama.seleccion_cortes entrenar --prueba --dispositivo cpu --workers 0
```

Piloto pequeño real de CPU (distinto del smoke):

```bash
python -m cancer_mama.seleccion_cortes preparar
python -m cancer_mama.seleccion_cortes cpu
```

Usa una red de 30.393 parámetros, seis épocas, semilla 42 y fold 0. Muestra
estratificada fija de desarrollo: 30 pacientes fit, 18 selección interna y 44
exteriores. Conserva las mismas pacientes en los tres brazos y evalúa todos sus
cortes. Las salidas llevan sufijo _cpu y no se mezclan con la matriz GPU.
Esta prueba detecta señales iniciales, no confirma mejora ni sustituye la matriz.

Salida: resultados/18_piloto_seleccion_cortes_20261021. Preparación reproducible;
el entrenamiento comprueba firmas y puede reanudarse. No cambiar código/configuración
durante una ejecución; utilizar otra salida si cambia el protocolo.
