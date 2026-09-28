# Auditoría inicial de BreastDCEDL

## Unidad estadística y clases

- 1,273 pacientes y 12,703 cortes (38,109 PNG).
- Train: 775 pacientes pCR=0 y 322 pCR=1.
- Test: 123 pacientes pCR=0 y 53 pCR=1.
- Por corte, train contiene 7,729 pCR=0 y 3,216 pCR=1.
- La clase positiva representa 29.4% de train y 30.1% de test; en train hay 2.41 negativos por positivo.
- Un clasificador trivial que siempre prediga pCR=0 obtendría 70.6% de accuracy en train.
- Cada paciente aporta entre 5 y 10 cortes (mediana 10).
- Los folds no tienen fuga, pero su tasa pCR varía entre 26.8% y 34.1%; conviene promediar los cinco folds.

## Comprobaciones de integridad

- [x] Sin nulos en campos esenciales de samples.csv: 0 valores nulos
- [x] Sin identificadores de corte duplicados: 0 duplicados
- [x] Una fila por paciente en patients.csv: 0 duplicados
- [x] Etiquetas binarias: samples=[0, 1]; patients=[0, 1]
- [x] Etiqueta constante dentro de cada paciente: 0 pacientes inconsistentes
- [x] Cada paciente pertenece a un único split: 0 pacientes con fuga
- [x] Los pacientes coinciden entre ambos CSV: solo samples=0; solo patients=0
- [x] Split y etiqueta coinciden entre samples y patients: 0 cortes inconsistentes
- [x] Folds válidos y separados por paciente: train=[np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4)]; test=[np.int64(-1)]

- Ficheros ausentes: 0.
- Errores de lectura: 0.
- Formato/modo/tamaño/dtype: {'PNG': 38109} / {'L': 38109} / {'256x256': 38109} / {'uint8': 38109}.

## Canales y rangos

Orden de entrada: **PRE, EARLY, LATE**. Son tiempos de adquisición, no RGB.

| Fase | mín | p1 | mediana | p99 | máx |
|---|---:|---:|---:|---:|---:|
| PRE | 0 | 0 | 15 | 172 | 255 |
| EARLY | 0 | 0 | 22 | 255 | 255 |
| LATE | 0 | 0 | 23 | 255 | 255 |

Al dividir por 255, el tensor tiene forma `(3, 256, 256)`, tipo `float32` y rango `[0, 1]`.
En tejido (`PRE > 0,1`), EARLY tiene mayor media que PRE en
1269/1273 pacientes.
La media de LATE queda por debajo de EARLY (patrón de *washout*) en
253/1273 pacientes
(19.9%).
El realce `EARLY - PRE` es la vista que mejor hace visible la entrada de contraste.

## Ejemplos mostrados

[
  {
    "sample_id": "ACRIN-6698-103939_z011",
    "patient_id": "ACRIN-6698-103939",
    "pCR": 0,
    "slice_index": 11
  },
  {
    "sample_id": "ACRIN-6698-108969_z026",
    "patient_id": "ACRIN-6698-108969",
    "pCR": 1,
    "slice_index": 26
  }
]

## Figuras

- `resumen.png`: clases, folds, cortes por paciente y rangos de intensidad.
- `ejemplos_fases.png`: casos pCR=0 y pCR=1 con PRE, EARLY, LATE y realce.
