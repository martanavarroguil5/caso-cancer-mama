# Preparación de datos — fold 0

## Decisiones

- La partición se realiza por paciente, nunca por corte.
- Fold 0 se reserva para validación; los otros cuatro folds forman train.
- Test permanece cerrado y no se crea un DataLoader de test.
- El baseline usa solo imágenes. Los vacíos clínicos no entran todavía en el modelo.
- Los PNG se convierten a `float32` en `[0,1]`; no se aplica normalización de ImageNet.
- Normalización activa: `none`.
- El único aumento inicial es volteo horizontal aleatorio, aplicado simultáneamente a PRE, EARLY y LATE.
- Validación no tiene aumentos y mantiene el orden (`shuffle=False`).

## Particiones

| Rol | Pacientes | Cortes | Pacientes pCR=0 | Pacientes pCR=1 | Tasa pCR |
|---|---:|---:|---:|---:|---:|
| train | 878 | 8,759 | 620 | 258 | 29.4% |
| validation | 219 | 2,186 | 155 | 64 | 29.2% |
| test_closed | 176 | 1,758 | 123 | 53 | 30.1% |

No existe ningún paciente compartido entre train, validación y test.

## Estadísticas calculadas solo con train

| Canal | Media | Desviación |
|---|---:|---:|
| PRE | 0.127491 | 0.158794 |
| EARLY | 0.195354 | 0.247697 |
| LATE | 0.205689 | 0.255529 |

- Media compartida: `0.176178`.
- Desviación compartida: `0.227650`.
- Rango observado: `[0.0, 1.0]`.
- Cortes utilizados: 8,759.

La configuración por defecto conserva `[0,1]`. La opción `shared` aplica una única media y desviación a los tres canales para no romper su relación temporal.

## Desbalance

- `pos_weight` por corte: `2.400233`.
- Razón negativa/positiva por paciente: `2.403101`.

Para `BCEWithLogitsLoss` se utilizará el peso por corte calculado únicamente con train. No se combinará inicialmente con sobremuestreo.

## Prueba de humo

- Train: `{"shape": [32, 3, 256, 256], "dtype": "torch.float32", "min": 0.0, "max": 1.0, "n_positivos": 9, "n_negativos": 23, "patient_ids_unicos_en_lote": 31}`
- Validación: `{"shape": [32, 3, 256, 256], "dtype": "torch.float32", "min": 0.0, "max": 1.0, "n_positivos": 0, "n_negativos": 32, "patient_ids_unicos_en_lote": 4}`

Las imágenes tienen forma `(batch, 3, 256, 256)`, tipo `float32`, etiquetas binarias y metadatos de paciente/corte alineados.

## Uso posterior

```python
import pipeline_datos as pdatos

samples = pdatos.cargar_samples()
particiones = pdatos.crear_particiones(samples, fold_validation=0)
# Cargar las estadísticas guardadas o calcularlas solo con particiones.train.
# construir_dataloaders(..., incluir_test=False)
```

El test solo debe abrirse una vez que arquitectura, pérdida, umbral y método de agregación por paciente estén cerrados.
