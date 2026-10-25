# Hipótesis de adquisición: CNN propia y media por paciente

25/10/2026. Protocolo ejecutable congelado antes del entrenamiento en
`resultados/25_robustez_adquisicion_20261025/protocolo.json`, con hash y fuentes.
Esta nota describe ese protocolo; no añade decisiones después de ver resultados.

## Motivo

Las aclaraciones del profesor fijan evaluación por paciente, media de cortes,
CNN propia 2D desde cero, BCE normal/ponderada y umbral en validación. La app de
un corte es una demostración de inferencia, no un motivo para cambiar la unidad
de evaluación. V008 permanece como experimento de un corte; V007 es referencia,
sin declararlo útil con su umbral histórico. Test ya conocido y validation privada
quedan fuera de este piloto y de cualquier ajuste.

La bibliografía de pCR no permite copiar directamente sus modelos o sus cifras:
Fridman emplea ViT/preentrenamiento; Braman/Cain son radiómica y Liu/Joo tienen
entradas adicionales o poblaciones distintas. Los mapas de realce y contexto ya
se probaron. Se han revisado los experimentos descartados antes de proponer otro.

## Hipótesis única

Un campo suave de intensidad compartido entre PRE/EARLY/LATE y los cortes de
cada paciente puede reducir dependencia de diferencias de adquisición. Inspirado
en [Chen et al., MICCAI2020](https://arxiv.org/abs/2006.13322), que investigan
segmentación cardíaca con campos adversariales. Nuestra adaptación aleatoria
no reproduce ese método ni cuenta con una mejora de pCR previamente demostrada.

Se conserva exactamente CNNMultiescalaCinetica:593.521 parámetros, canales,
recortes y dropout de V007. Se inicializa desde cero en cada entrenamiento.

Campo: rejilla uniforme4×4, interpolación bicúbica, limitada a[0,1];
`B=1−0,15·campo`, con probabilidad0,5 por paciente/época. B es idéntico para
sus fases y cortes. Primero intensidad, después geometría usual. No hay color
jitter, normalización por fase ni clipping de imágenes; no se aplica en inferencia.
El rango[0,85;1] evita nueva saturación y mantiene el signo de las diferencias.

## Comparación y roles

Control/campo × BCE normal/ponderada × semillas42/2026 × folds originales0/3:
dieciséis ejecuciones. Igual inicialización, pacientes y aumentos geométricos en
brazos emparejados. Se compara un solo cambio; no se busca una intensidad mejor
después de comprobar.

| Fold | Fit | Selección de época/pérdida | Calibración/umbral | Comprobación |
|---|---:|---:|---:|---:|
| 0 | 614 | 132 | 132 | 219 |
| 3 | 613 | 132 | 132 | 220 |

Los roles son disjuntos por paciente para cada modelo, estratificados por
cohorte/clase. Como en validación cruzada, los roles pueden cambiar entre folds.
Ninguna paciente entrena el modelo que produce su predicción de comprobación.

AdamW LR0,0003/WD0,01 (receta modal de selecciones internas históricas de V007),
lote32, AMP, coseno, máximo40 épocas, mínimo8, paciencia8, min_delta0,001 y clip5.
BCEWithLogitsLoss por corte, peso1/n_cortes y normalizador fijo para igual
contribución por paciente. Pos_weight=N0/N1 calculado en pacientes de fit.

La época se elige por AUC de probabilidades medias por paciente en selección;
la pérdida, por AUC del ensemble de ambas semillas en esas mismas pacientes.
Empates de pérdida a favor de normal. Se congelan todas las elecciones antes
de calibración/comprobación. La inferencia usa todos los cortes, sin selección
de los más favorables ni atención MIL.

## Calibración y umbral

Platt regularizado C1, pendiente no negativa, solo con pacientes de calibración,
usando exactamente el ensemble de dos modelos y la media de cortes que se
comprueba después. Cada brazo/fold/pérdida tiene su política, todas congeladas
antes de generar métricas de comprobación.

Regla primaria: máxima sensibilidad con especificidad≥95 % en calibración;
empates por precisión, especificidad y umbral mayor. Uso hipotético académico:
reducir falsos optimismos al predecir pCR. No valida uso clínico ni garantiza
95 % fuera de calibración. Si la red no consigue positivos fiables, se informa.
Máximo macro-F1 en calibración como regla secundaria, sin elegir luego entre
políticas según comprobación. También se informa umbral0,50 crudo.

## Métrica y cierre prefijado

Principal: media de AUC cruda de folds0/3, por paciente y ensemble de semillas.
Secundarias: agrupada cruda/calibrada, AP, logloss, Brier, FP/FN, precisión/recall
y F1 por clase, macro-F1, cohorte y brecha en época elegida frente a última.
IC95 % por bootstrap pareado estratificado por fold/cohorte/clase.

Solo habilita ampliación interna a cinco folds si se cumplen todas:

- AUC media del campo≥0,60.
- Δmedia frente a control≥0,02.
- Mejora en ambos folds.
- Brecha mediana seleccionada no aumenta más de0,01.
- Límite inferior del IC95 % pareado de Δmedia>0.

Si no pasa, cerrar esta hipótesis como insuficiente. No aumentar intensidad,
cambiar umbral, excluir cohortes o reiniciar búsquedas hasta obtener0,60.
Si pasa, no es una validación independiente: train tiene amplio historial
adaptativo. Test no se abre para decidir ampliación ni para ajustar nada.

## Integridad y entrega

Fuentes, roles, políticas, épocas y pesos identificados con hashes; checkpoints
para reanudación locales; inferencia segura exportada sin optimizador/RNG.
Verificación de AdamW/AMP, pérdidas y pesos finitos. Comparación de inicialización
en ambos brazos y hashes de versiones anteriores al principio/final.

V002/V007/V008 y demás paquetes archivados permanecen intactos; no se cambia
aplicación ni modelo activo. Solo main local, sin push ni ramas. Se conservan
informe, curvas, configuraciones y límites aunque el resultado sea negativo.
