# Interpretación de la comparación final de CNN de solo imagen

25/10/2026. Evaluación autorizada de los paquetes ya congelados, sobre las mismas
176 pacientes (53 pCR), 1758 cortes y 5274 PNG. No se ha entrenado ningún modelo
con datos del caso ni se ha ajustado calibración, umbral o receta con el test.

| Paquete congelado | AUC test final | IC95 % | AP |
|---|---:|---|---:|
| V001 histórica | 0,5462 | [0,4632; 0,6268] | 0,3417 |
| CNN híbrida | 0,5039 | [0,4152; 0,5900] | 0,3114 |
| V007 cinética | **0,6084** | **[0,5222; 0,6935]** | **0,3889** |

V007 tiene la mayor AUC puntual observada. Supera 0,60 en este test concreto,
pero el intervalo incluye valores inferiores a 0,60. La diferencia V007−V001
es +0,0621, IC95 % pareado [−0,0164; +0,1424]: no permite afirmar una mejora
concluyente sobre V001. La diferencia frente a la híbrida es +0,1045, IC95 %
[+0,0186; +0,1904], en una comparación exploratoria.

La unidad es la paciente, con todos sus cortes y una probabilidad final por
paquete. Los tres usan la misma población, etiquetas, PNG y remuestreos. Se
mantienen sus distintos preprocesamientos, calibraciones y ensembles archivados:
esto compara paquetes completos, no aísla el efecto de la arquitectura ni iguala
el presupuesto histórico de entrenamiento. No se escoge un subconjunto de test.

## Clasificación con las políticas guardadas

Una mayor AUC no garantiza sensibilidad suficiente con el umbral archivado.

| Modelo | Umbral final archivado | TN | FP | FN | TP | Sensibilidad | Especificidad |
|---|---:|---:|---:|---:|---:|---:|---:|
| V001 | 0,298241 | 73 | 50 | 29 | 24 | 45,28 % | 59,35 % |
| V007 | 0,308794 | 122 | 1 | 53 | 0 | **0 %** | 99,19 % |

Estos umbrales son diferentes; no representan el mismo punto operativo. Al
umbral común 0,50 sobre probabilidades finales, los tres clasifican todas las
pacientes como no pCR (TN=123, FP=0, FN=53, TP=0). La híbrida no tenía un umbral
operativo alternativo congelado. Las matrices a 0,50 crudo también están en el
informe. No se selecciona ahora un umbral mejor con estas pacientes.

## Qué demuestra y qué queda abierto

- V001 reutiliza sus predicciones originales: su AUC coincide exactamente con
  la evaluación de septiembre. No se repite su inferencia.
- La mayor AUC puntual de V007 no elimina el sobreajuste observado durante su
  entrenamiento ni resuelve la baja sensibilidad de la política archivada.
- El resultado mide inferencia por paciente usando todos sus cortes. No valida
  por separado la entrada de un solo corte que exige la demostración web.
- El test se ha consultado históricamente en el proyecto. Esta es una evaluación
  retrospectiva fuera del entrenamiento; no una confirmación externa ciega nueva.
- La validación privada del profesor permanece cerrada. No se activa otro
  modelo ni se inicia otra búsqueda tras los resultados.
- La comparación normal/ponderada de la receta híbrida continúa pendiente para
  completar esa parte de la entrega; esta evaluación no la sustituye.

Pasaron las 80 pruebas, incluidas tres comprobaciones del cálculo de métricas,
remuestreo pareado y reproducibilidad. La inferencia optimizada se contrastó con
las CLI congeladas usando datos sintéticos; las AUC se comprobaron además por
comparación directa de todos los pares positivo/negativo.

[Informe, curvas y protocolo](../resultados/23_evaluacion_test_imagen_20261025/INFORME_RESULTADOS.md).
[Predicciones pareadas](../resultados/23_evaluacion_test_imagen_20261025/comparacion_pacientes.csv).
[Métricas e intervalos](../resultados/23_evaluacion_test_imagen_20261025/metricas.json).

## Comprobación numérica de dispositivos

V007 coincide con CPU en dos pacientes de distinta longitud, con desviación
máxima 2,2e-8. En la híbrida se detectó una diferencia CPU/GPU ligeramente mayor
que la tolerancia estricta inicial de 1e-5. Se comprobó entonces el conjunto
completo: diferencia máxima de probabilidad 1,81e-5 y **AUC idéntica en CPU y GPU**.
Se conserva el incumplimiento de esa tolerancia en `auditoria_numerica.json`;
no se cambia la tolerancia, los pesos ni la evaluación principal prefijada en GPU.
