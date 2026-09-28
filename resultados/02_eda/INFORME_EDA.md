# EDA profesional de BreastDCEDL

## Resumen ejecutivo

- La unidad de análisis correcta es la paciente: **1,273 pacientes**, 1,097 en train y 176 en test.
- La clase positiva representa **29.4% de train**. Predecir siempre pCR=0 daría 70.6% de accuracy, por lo que accuracy aislada no será una métrica suficiente.
- Los datos clínicos están casi completos salvo `menopause` (13.5% ausente). Los vacíos son "no disponible", no ceros.
- Cohorte, adquisición y preprocesado están relacionados. La tasa pCR de train también cambia por cohorte (duke: 21.1% (n=209), spy1: 25.0% (n=104), spy2: 32.1% (n=784)). Esto crea riesgo de que el modelo aprenda protocolo/hospital en vez de respuesta tumoral.
- Las mayores diferencias numéricas con pCR incluyen coordenadas de recorte y tiempos de adquisición, no solo variables clínicas. Es una señal de posible confusión por cohorte o preprocesado.
- La PCA se usa como diagnóstico de estructura, no como prueba de separabilidad. No aparece una separación limpia de pCR en los dos primeros componentes; sí existe estructura por cohorte.
- Las asociaciones que se muestran son descriptivas y se calcularon solo en train. No implican causalidad ni rendimiento predictivo fuera de muestra.

## 1. Fuentes y granularidad

- `patients.csv`: una fila por paciente, objetivo, variables clínicas, cohorte y adquisición.
- `samples.csv`: una fila por corte; unas diez observaciones correlacionadas por paciente.
- PNG: PRE, EARLY y LATE son fases temporales alineadas, no canales RGB.
- `excluded_patients.csv`: dos pacientes excluidas. La causa se conserva como código de error, pero su significado clínico/técnico no está documentado con suficiente claridad.

Las tablas originales no se modifican. Las características derivadas se guardan aparte y siempre incluyen `pid` para trazabilidad.

## 2. Calidad y valores ausentes

| variable | ausentes | ausentes_pct |
|---|---|---|
| menopause | 172 | 13.5% |
| HER2pos | 5 | 0.4% |
| HER2 | 5 | 0.4% |
| HRposHER2neg | 5 | 0.4% |
| HR_HER2_STATUS | 5 | 0.4% |
| TripleNeg | 5 | 0.4% |
| age | 3 | 0.2% |
| race_black | 3 | 0.2% |
| race_white | 3 | 0.2% |
| HR | 2 | 0.2% |
| tum_vol | 1 | 0.1% |

Hallazgos de calidad:

- `pre` es constante (0) y no aporta información como predictor.
- `test` y `split` describen la misma partición y deben excluirse de cualquier modelo.
- `HR_HER2_STATUS`, `TripleNeg`, `HER2pos` y `HRposHER2neg` son representaciones derivadas de HR/HER2. Incluirlas todas duplica información.
- Las coordenadas de máscara y recorte proceden del preprocesado. Pueden actuar como atajos y no deberían entrar en un primer modelo.
- La ausencia de `menopause` depende claramente de la fuente: spy1 100.0%, spy2 3.7% y duke 0.0%. Debe conservarse como información de disponibilidad y tratarse dentro de cada fold.

## 3. Composición y desbalance

Train contiene 775 pacientes pCR=0 y 322 pCR=1. Test contiene 123 y 53, respectivamente. La tasa positiva de test (30.1%) se informa solo para describir el conjunto entregado; no se utiliza para seleccionar variables ni decisiones del modelo.

Los folds mantienen pacientes completas, pero la tasa positiva varía entre 26.8% y 34.1%. El resultado final debe promediar los cinco folds y reportar dispersión.

## 4. Variables clínicas y asociación descriptiva con pCR

Variables numéricas con mayor diferencia estandarizada en train:

| variable | mediana_pcr0 | mediana_pcr1 | smd_pcr1_menos_pcr0 |
|---|---|---|---|
| crop_width | 69.000 | 62.000 | -0.231 |
| tumor_z_span | 33.000 | 29.000 | -0.221 |
| post_early | 2.000 | 2.000 | 0.219 |
| mask_start | 20.000 | 25.500 | 0.197 |
| late_saturation_fraction | 0.019 | 0.018 | -0.193 |
| post_late | 5.000 | 5.000 | 0.180 |
| log1p_tum_vol | 2.669 | 2.446 | -0.179 |
| tum_vol | 13.426 | 10.537 | -0.174 |

Variables categóricas con mayor asociación descriptiva:

| variable | cramers_v_variable |
|---|---|
| HR_HER2_STATUS | 0.284 |
| HRposHER2neg | 0.281 |
| HR | 0.248 |
| HER2 | 0.177 |
| HER2pos | 0.177 |
| TripleNeg | 0.129 |

Estas magnitudes sirven para priorizar inspección. No se realizan afirmaciones clínicas ni selección definitiva basadas en este EDA.

## 5. Cohortes, adquisición y cambio de distribución

Variables con mayor diferencia entre test y train:

| variable | media_train | media_test | smd_test_menos_train |
|---|---|---|---|
| post_early | 1.735 | 1.585 | -0.318 |
| n_z | 88.710 | 76.733 | -0.273 |
| post_late | 4.786 | 4.409 | -0.272 |
| n_times | 6.291 | 5.835 | -0.259 |
| slice_thick | 1.803 | 1.936 | 0.236 |
| mask_end | 61.508 | 55.506 | -0.198 |
| enhancement_mean | 0.139 | 0.129 | -0.190 |
| mask_start | 26.145 | 22.199 | -0.186 |

En variables categóricas, los mayores cambios de distribución son:

| variable | total_variation_train_test |
|---|---|
| dataset | 0.152 |
| post_early | 0.149 |
| post_late | 0.147 |
| n_times | 0.146 |
| menopause | 0.094 |
| HR_HER2_STATUS | 0.061 |
| TripleNeg | 0.061 |
| n_xy | 0.055 |

Una |SMD| cercana o superior a 0,2 ya merece revisión; valores mayores no prueban fuga, pero sí cambio de dominio. Cohorte y parámetros de adquisición deben formar parte del análisis de errores y de las métricas estratificadas.

## 6. Redundancia y correlación

Pares con correlación de Spearman absoluta ≥ 0,90:

| variable_a | variable_b | spearman |
|---|---|---|
| tum_vol | log1p_tum_vol | 1.000 |
| HER2 | HER2pos | 1.000 |
| early_mean_all | late_mean_all | 0.955 |
| early_mean_tissue | early_p50_tissue | 0.953 |
| early_p50_tissue | late_p50_tissue | 0.949 |
| late_mean_tissue | late_p50_tissue | 0.947 |
| early_p10_tissue | late_p10_tissue | 0.942 |
| early_mean_tissue | late_p50_tissue | 0.912 |
| enhancement_std | enhancement_p90 | 0.909 |
| early_mean_tissue | late_mean_tissue | 0.908 |
| xy_spacing | fov_xy | 0.904 |

Para modelos tabulares no conviene introducir simultáneamente variables equivalentes o derivadas. Para la CNN, estas correlaciones sirven principalmente para detectar qué metadatos podrían funcionar como atajos.

## 7. PCA clínica y de imagen

PCA clínica/adquisición: PC1 explica 21.3% y PC2 15.6%. Las mayores contribuciones absolutas a PC1 son: post_early, n_z, tumor_z_span, n_times, post_late.

PCA de características de imagen: PC1 explica 29.4% y PC2 24.0%. Las mayores contribuciones absolutas a PC1 son: early_p10_tissue, pre_p50_tissue, early_p50_tissue, late_p50_tissue, early_mean_tissue.

La PCA trabaja con una fila por paciente, imputación mediana y escalado z-score. La imputación es únicamente para la visualización; cualquier pipeline predictivo deberá ajustar imputadores usando solo el fold de entrenamiento.

## 8. Revisión visual de extremos

Se revisaron automáticamente pacientes extremos por realce, saturación y fracción de fondo:

```json
[
  {
"criterio": "Mayor realce medio",
"variable": "enhancement_mean",
"pid": "Breast_MRI_142",
"valor": 0.3355407800053209,
"sample_id": "Breast_MRI_142_z007"
  },
  {
"criterio": "Menor realce medio",
"variable": "enhancement_mean",
"pid": "ISPY1_1069",
"valor": -0.0450672846685074,
"sample_id": "ISPY1_1069_z012"
  },
  {
"criterio": "Mayor saturación EARLY",
"variable": "early_saturation_fraction",
"pid": "ACRIN-6698-580632",
"valor": 0.0559066772460937,
"sample_id": "ACRIN-6698-580632_z024"
  },
  {
"criterio": "Mayor fondo",
"variable": "background_fraction",
"pid": "ISPY1_1193",
"valor": 0.9216354370117188,
"sample_id": "ISPY1_1193_z010"
  }
]
```

Son candidatos para inspección, no errores confirmados. No se elimina ningún paciente automáticamente.

## 9. Decisiones para el modelado

1. Dividir siempre por paciente y usar los folds ya definidos.
2. Mantener test cerrado durante selección de arquitectura, hiperparámetros y umbral.
3. Ajustar normalización e imputación exclusivamente en el train de cada fold.
4. Empezar con un modelo de imagen sin metadatos de adquisición para reducir atajos.
5. Comparar después con un modelo multimodal limitado a variables clínicas justificadas.
6. Reportar métricas por paciente, cohorte y subtipo, además del promedio global.
7. Evaluar calibración, sensibilidad, especificidad, AUC y matriz de confusión; no solo accuracy.
8. Tratar raza como variable sensible para auditoría de equidad, no como explicación causal.

## 10. Limitaciones

- Es un EDA observacional. Las asociaciones no son causales.
- Los cortes de una paciente están correlacionados; no se usan como observaciones independientes.
- Las cohortes se adquirieron y procesaron de manera distinta.
- Varias unidades clínicas no están explicitadas en el diccionario original; se conservan los nombres sin inventar unidades.
- La PCA es lineal y resume varianza global; no demuestra que una CNN vaya a clasificar pCR.
- No se corrigió por comparaciones múltiples porque no se presentan pruebas confirmatorias ni p-valores.

## Archivos generados

- `tablas/patient_image_features.csv`: características visuales por paciente.
- `tablas/patient_eda_features.csv`: tabla paciente integrada, sin alterar las fuentes.
- `tablas/feature_policy.csv`: política de uso de variables.
- `tablas/associations_*` y `split_shift_*`: resultados trazables del análisis.
- `figuras/`: calidad, clínica, cohortes, PCA, shift y casos extremos.
