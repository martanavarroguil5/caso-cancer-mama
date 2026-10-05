# Resultados de los cuatro ensayos de generalización

Ensayos completados el 05/10/2026 en la RTX 3090. **Ninguna de las cuatro propuestas cumple todos los criterios fijados de adopción.** El entrenador activo conserva la configuración anterior; los candidatos no se incorporan. Una diferencia positiva que no supere estos criterios queda inconcluyente, sin presentarse como una mejora demostrada.

## Resultado principal

AUC cruda por paciente, media de cinco folds exteriores y ensemble de semillas 42/2026, con BCE ponderada. El control se reentrenó con el mismo protocolo que cada candidato.

| Cambio | AUC media | Diferencia | IC98,75% pareado | Folds mejores | Decisión |
|---|---:|---:|---|---:|---|
| Control | 0.5504 | — | — | — | Referencia |
| Dropout2d p=0,10 | 0.5729 | +0.0225 | [-0.0126, +0.0584] | 4/5 | No adoptado; inconcluyente |
| Weight decay 0,003 | 0.5542 | +0.0038 | [-0.0309, +0.0372] | 4/5 | No adoptado; inconcluyente |
| Traslación y escala | 0.5667 | +0.0163 | [-0.0152, +0.0506] | 4/5 | No adoptado; inconcluyente |
| EMA 0,99 | 0.5562 | +0.0058 | [-0.0262, +0.0362] | 4/5 | No adoptado; inconcluyente |

Los intervalos usan 4.000 bootstrap pareados de pacientes, estratificados por fold, cohorte y clase, y corrección Bonferroni para cuatro comparaciones. Son intervalos condicionados a los modelos entrenados: no recogen toda la variabilidad del aprendizaje ni eliminan el optimismo del desarrollo adaptativo.

## Consistencia y criterios de adopción

Se exigieron simultáneamente ΔAUC ≥ +0,01, mejora en al menos 4/5 folds, ΔAUC media positiva con cada semilla, AP media sin empeoramiento y límite inferior del IC98,75% mayor que cero. No se rebajaron esos requisitos después de observar los resultados.

| Cambio | ΔAUC semilla 42 | ΔAUC semilla 2026 | ΔAP media | Criterios cumplidos | IC95% de ΔAUC |
|---|---:|---:|---:|---:|---|
| Dropout2d p=0,10 | +0.0067 | +0.0221 | +0.0248 | 4/5 | [-0.0046, +0.0503] |
| Weight decay 0,003 | +0.0081 | +0.0007 | +0.0130 | 3/5 | [-0.0233, +0.0301] |
| Traslación y escala | +0.0180 | +0.0154 | +0.0176 | 4/5 | [-0.0094, +0.0423] |
| EMA 0,99 | +0.0046 | +0.0053 | +0.0130 | 3/5 | [-0.0199, +0.0298] |

### AUC por fold exterior

| Modelo | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---:|---:|---:|---:|---:|
| Control | 0.5673 | 0.5334 | 0.5802 | 0.5044 | 0.5667 |
| Dropout2d p=0,10 | 0.6129 | 0.5342 | 0.5610 | 0.5522 | 0.6041 |
| Weight decay 0,003 | 0.5733 | 0.5146 | 0.5816 | 0.5209 | 0.5803 |
| Traslación y escala | 0.5979 | 0.5085 | 0.5971 | 0.5389 | 0.5912 |
| EMA 0,99 | 0.5739 | 0.5274 | 0.5835 | 0.5106 | 0.5856 |

## Comparación requerida con BCE normal

La pérdida ponderada era el análisis principal declarado. BCE normal es una comparación secundaria de la práctica y no sustituye retrospectivamente ese criterio.

| Modelo | AUC media BCE normal | ΔAUC frente al control normal | AP media normal | AUC OOF normal |
|---|---:|---:|---:|---:|
| Control | 0.5634 | +0.0000 | 0.3619 | 0.5431 |
| Dropout2d p=0,10 | 0.5559 | -0.0075 | 0.3542 | 0.5360 |
| Weight decay 0,003 | 0.5569 | -0.0064 | 0.3582 | 0.5542 |
| Traslación y escala | 0.5412 | -0.0221 | 0.3498 | 0.5363 |
| EMA 0,99 | 0.5718 | +0.0084 | 0.3744 | 0.5604 |

## Métricas secundarias con BCE ponderada

Predicciones exteriores crudas agrupadas, sin calibración ni elección de umbral. F1, sensibilidad y especificidad usan el umbral fijo 0,5. AUC OOF agrupada y media de AUC por fold son métricas distintas; la segunda era el criterio principal.

| Modelo | AUC OOF | AP OOF | Brier | F1 | Sensibilidad | Especificidad |
|---|---:|---:|---:|---:|---:|---:|
| Control | 0.5412 | 0.3249 | 0.2557 | 0.4075 | 0.5373 | 0.5432 |
| Dropout2d p=0,10 | 0.5593 | 0.3502 | 0.2402 | 0.3898 | 0.4503 | 0.6426 |
| Weight decay 0,003 | 0.5475 | 0.3357 | 0.2399 | 0.3583 | 0.4006 | 0.6529 |
| Traslación y escala | 0.5512 | 0.3384 | 0.2432 | 0.3894 | 0.4317 | 0.6735 |
| EMA 0,99 | 0.5530 | 0.3342 | 0.2484 | 0.4000 | 0.4689 | 0.6361 |

## Diagnóstico de EMA y BatchNorm

El control con BatchNorm recalculada obtiene AUC media 0.5558: diferencia +0.0054 frente al control habitual. EMA frente a ese control obtiene diferencia +0.0004, IC95% [-0.0241, +0.0251].

Este diagnóstico separa el promedio de pesos de la recalculación de estadísticas. Es secundario: no convierte `control_bn` en un quinto candidato ni permite atribuir toda la diferencia EMA/control al promedio de parámetros.

## Análisis descriptivo por cohorte

AUC OOF de paciente por cohorte, exploratoria y sin criterios adicionales de selección. No equivale a dejar una cohorte completamente fuera del aprendizaje.

| Cohorte | Pacientes | pCR=1 | Control | Dropout2d | Weight decay | Afines | EMA |
|---|---:|---:|---:|---:|---:|---:|---:|
| duke | 209 | 44 | 0.4908 | 0.4787 | 0.5101 | 0.4983 | 0.4933 |
| spy1 | 104 | 26 | 0.5799 | 0.6149 | 0.5271 | 0.5473 | 0.5823 |
| spy2 | 784 | 252 | 0.5318 | 0.5540 | 0.5455 | 0.5503 | 0.5525 |

## Ejecución y reproducibilidad

Se completaron **80 jobs de entrenamiento**, **100 evaluaciones principales de modelos** y **20 diagnósticos de BatchNorm**, con 1431 épocas físicas en aproximadamente 82.1 minutos. EMA y `control_bn` comparten la trayectoria de optimización del control, con checkpoints y paradas propios. Las tres salidas no son tres entrenamientos independientes.

Se evaluaron 1097 pacientes y 10.945 cortes de desarrollo. Cinco folds originales, semillas 42/2026, dos pérdidas y máximo 30 épocas por job. Dentro de los cuatro folds disponibles se reservó 15% de pacientes para selección, estratificando por cohorte × pCR. Solo fit aprende pesos, calcula el peso BCE y recalcula BatchNorm. El fold exterior no selecciona épocas, calibración ni umbral.

Antes de la matriz pasaron seis pruebas CPU del ensayo: particiones, transformación conjunta, inicialización/arquitectura, EMA/BatchNorm, reanudación exacta de las cuatro rutas y reproducción numérica del control original. Las cuatro rutas pasaron humo CUDA y se repitieron durante ejecución concurrente: pesos e historias de selección idénticos. El cálculo de AUC del bootstrap se comprobó frente a scikit-learn con empates y predicciones constantes.

Durante la ejecución se corrigió la exportación JSON del análisis convirtiendo las medias NumPy a float nativo. La verificación comprobó que sus valores numéricos no cambian, y probó exportación y bootstrap con casos sintéticos. No cambió el código congelado de entrenamiento ni los criterios del protocolo.

Los seis casos de inspección de aumentos pertenecen a train, uno por cohorte y clase. Las tres fases se transforman juntas. La inspección visual no certifica que ninguna lesión de todo el conjunto pueda perder cobertura: no se dispone de máscaras de tumor en los PNG.

La auditoría completa verificó 20 parejas de inicialización, 5 particiones exteriores, el pareado de RNG de loader/aumentos, la cobertura OOF, checkpoints elegidos por selección interna y los hashes de sus artefactos. Los 3334 archivos protegidos previos mantienen sus SHA-256. Los pasos 01–03, `pipeline_datos.py`, entrenador activo, modelo histórico, calibración y umbral se conservan.

Se abrieron **cero imágenes de test** y **cero de la validación privada**. Las 1.097 pacientes de desarrollo ya se habían observado en investigaciones anteriores. Esta es evaluación interna adaptativa; no demuestra rendimiento en pacientes externos y no se compara directamente con las AUC de ensayos antiguos que usaron otras particiones o lotes.

## Evidencia y reconstrucción

Protocolo previo: [PROTOCOLO_CUATRO_MEJORAS.md](PROTOCOLO_CUATRO_MEJORAS.md). Evidencia local en `resultados/04_entrenamiento/cuatro_mejoras_20261005/`, excluida de Git para conservar artefactos y scripts aislados sin añadir experimentos al entrenador. El protocolo está fijado en el commit `3623227`.

- `protocolo_fijado.json`: hashes del protocolo y código congelado antes de la matriz.
- `preservacion_previa.json`: inventario previo y hashes de los archivos protegidos.
- `codigo/`: snapshot del entrenador/pipeline, ensayo, coordinador, pruebas y análisis.
- `runs/`: configuración, entorno, partición, inicialización, historia, `last.pt` y por salida `best.pt`, `inference.pt` y predicciones exteriores.
- `evaluacion/resultado.json`, `modelos.csv`, `oof_weighted.csv`, `oof_normal.csv`, `cohortes.csv` y `comparacion_auc.png`: resultados y análisis.
- `pruebas_cpu.log`, `pruebas_analisis.json`, `pruebas_analisis_exportacion.json`, `reproducibilidad_gpu.json` y `qc_afines.*`: verificaciones previas y de exportación.

Para auditar de nuevo las predicciones congeladas, sin volver a entrenar:

```bash
.venv/bin/python -B resultados/04_entrenamiento/cuatro_mejoras_20261005/codigo/evaluar.py
.venv/bin/python -B resultados/04_entrenamiento/cuatro_mejoras_20261005/codigo/documentar.py
```

Para reconstruir el entrenamiento en el mismo entorno, usar el código congelado y otra carpeta de evidencia. El coordinador reanuda y comprueba configuraciones/hashes; no mezcla código nuevo con checkpoints existentes. La identidad numérica entre hardware o versiones diferentes no está garantizada.
