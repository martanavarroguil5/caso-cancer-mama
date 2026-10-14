# Protocolo: CNN principal solo imagen y consciente de las fases

Fecha de fijación: 14/10/2026. Estado: **preparado, no entrenado**.

## Objetivo docente

El modelo principal predice pCR exclusivamente a partir de PRE, EARLY y LATE.
Es una CNN 2D propia, inicializada desde cero, sin redes preentrenadas y sin edad,
volumen tumoral, HR o HER2. El multimodal v002 se conserva intacto como
comparador clínico, no como modelo principal de imagen.

No se fija como objetivo alcanzar AUC 0,80. Se prioriza una estimación honesta,
reproducible y comparable con la bibliografía. El test reservado permanece
cerrado durante todo el desarrollo.

## Arquitectura

Las fases son tiempos DCE ordenados, no canales RGB. Un encoder monocanal con
pesos compartidos procesa por separado PRE, EARLY y LATE. La fusión recibe:

1. características PRE;
2. características EARLY;
3. características LATE;
4. EARLY−PRE;
5. LATE−PRE;
6. LATE−EARLY.

Una convolución 1×1 fusiona esas vistas y dos bloques convolucionales producen
el logit. La red tiene 442.577 parámetros, frente a 551.921 del multimodal v002.
La reducción y el encoder compartido limitan capacidad sin borrar el orden
temporal. La escala sigue siendo `uint8/255` y después `2*x-1`; no se normaliza
cada fase por separado.

## Separación de pacientes

Se mantienen los cinco folds oficiales. Para cada fold exterior:

1. el fold exterior queda completamente aislado;
2. los otros cuatro folds se dividen, por paciente y estratificando cohorte ×
   pCR, en fit y selección interna del 15 %;
3. el checkpoint se selecciona por AUC de paciente en selección interna;
4. calibración y umbral se ajustan con esa misma selección interna;
5. solo entonces se predice una vez el fold exterior.

La partición interna depende del fold, pero es idéntica para ambas pérdidas y
semillas. De este modo el fold exterior no selecciona época ni umbral.

## Entrenamiento y evaluación

- BCE normal frente a BCE ponderada, sin mezclar ambas decisiones.
- Semillas 42 y 2026; folds exteriores 0–4: 20 ejecuciones.
- AdamW, LR 0,0008, weight decay 0,001, lote 32, máximo 40 épocas.
- Parada temprana: mínimo 8 épocas, paciencia 8.
- Flip horizontal y rotaciones de 90° compartidos entre las tres fases.
- Unidad de evaluación: paciente; agregación fijada en la **media** de cortes.
- Resultado principal: media ± desviación de AUC cruda de los cinco folds.

Cada run guarda juntas las curvas de loss y AUC de fit/selección y la brecha de
AUC en cada época. El comparador genera además una figura agregada con mediana e
IQR. Las últimas épocas no sustituyen el checkpoint seleccionado.

## Política de decisión

El falso positivo de pCR significa predecir respuesta completa cuando realmente
queda enfermedad residual; pCR no equivale a curación o supervivencia. Como
punto operativo principal se maximiza sensibilidad entre los umbrales que
alcanzan al menos 90 % de especificidad en selección interna. Se conserva
secundariamente el umbral de mayor macro-F1.

Se reportan AUC, average precision, Brier, log-loss, matriz, precision, recall y
F1 de ambas clases, macro-F1, balanced accuracy y el baseline que siempre dice
no pCR. La pérdida ganadora en desarrollo se congela antes de cualquier futuro
test independiente.

## Ejecución en el ordenador con GPU

Solo comprobar entorno y código, sin entrenar:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/entrenar_cnn_imagen_gpu.ps1 -SoloComprobar
```

Ejecutar humo, 20 entrenamientos reanudables y comparación:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/entrenar_cnn_imagen_gpu.ps1
```

La salida por defecto es
`resultados/04_entrenamiento/cnn_imagen_fases_20261014/`. Debe copiarse de
vuelta completa. Las ejecuciones de humo quedan en `pruebas/` y nunca son
elegibles. Repetir el comando con la misma configuración reanuda checkpoints.

## Límites

La comparación normal/ponderada sigue siendo selección de desarrollo. Las
predicciones exteriores proporcionan validación cruzada, no una nueva cohorte
independiente. El test histórico ya consultado no se reabre para escoger modelos;
la política final queda preparada para una prueba independiente futura.
