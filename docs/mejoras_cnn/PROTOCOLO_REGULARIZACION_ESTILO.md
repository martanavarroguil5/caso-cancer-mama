# Regularización de estilo de la CNN cinética — 26/10/2026

Hipótesis única: mezclar estadísticas de características tempranas puede reducir
la dependencia de apariencia de adquisición. No se afirma que estas estadísticas
sean solo escáner: también pueden contener señal biológica y empeorar el modelo.

## Investigación y límites de traslado

- [Zhou et al., ICLR 2021, MixStyle](https://arxiv.org/html/2104.02008v1):
  interpola media y desviación por canal de características tempranas de ejemplos
  de fit, con gradiente detenido en las estadísticas. Se desactiva en inferencia.
  El artículo estudia clasificación natural, reidentificación y aprendizaje por
  refuerzo; no es evidencia de AUC pCR en estas imágenes.
- [Repositorio de los autores](https://github.com/KaiyangZhou/mixstyle-release):
  recomienda bloques tempranos y documenta p=0,5 y alpha=0,1. Se implementan las
  ecuaciones en nuestra CNN propia; no se descarga su arquitectura ni sus pesos.
- [CSU, 2024](https://pubmed.ncbi.nlm.nih.gov/38978834/): extensión de regularización
  de estilo en clasificación, incluida Camelyon17. Es otra técnica y otra tarea;
  no se implementa CSU ni se traslada su rendimiento a pCR.

Ya fallaron o resultaron inconcluyentes LR/WD, cambios de capacidad/dropout,
GroupNorm, SAM, FiLM, mixup, campos de brillo compartidos y la CNN espacial.
Este ensayo no modifica esos factores a la vez.

## Comparación congelada

Único candidato: arquitectura V007 de 593.521 parámetros desde cero, agregando
MixStyle tras bloques 0 y 1. Sin parámetros adicionales. p=0,5, Beta(0,1;0,1),
epsilon=1e-6, estadísticas FP32 y varianza muestral. Pareja aleatoria dentro del
lote de cortes de fit, sin etiquetas ni cohorte. Las tres escalas de un corte
comparten pareja y coeficiente, y las estadísticas se emparejan por escala.
La permutación puede mantener un corte o unir dos cortes de una misma paciente;
no se fuerza diversidad de cohorte. Esto adapta el algoritmo a nuestra red.

PRE/EARLY/LATE PNG/255, diferencias firmadas y geometría compartida; sin clínica,
pesos preentrenados, nuevas modalidades, máscaras ni modificación de PNG.
BCEWithLogitsLoss por corte con contribución 1/n_cortes por paciente y normalizador
fijo; probabilidad de paciente = media de probabilidades de todos sus cortes.
Esta regularización de características no cambia la normalización de entrada.

20 ejecuciones: cinco folds, semillas 42/2026, BCE normal/ponderada. Roles auditados
del ensayo 27: 745–746 pacientes fit, 132 selección y 219–220 exteriores. Mismos
LR=0,0003, WD=0,01, batch32, dropout0,35/0,1, coseno40, mínimo observado14,
paciencia8 y deltaBCE0,001. Se puede elegir un checkpoint anterior a época14.
Checkpoint principal por mínima BCE interna por paciente; pérdida elegida por
AUC de la pareja en selección, empate normal. Se congelan cinco parejas y
políticas antes de inferir exterior. La regla de checkpoint por AUC es secundaria.

Control cerrado: resultados/27_criterio_docente_20261025, sin reentrenar ni cambiar
su receta; mediaAUC0,556270. Sus hashes, roles y resultados se verifican. Comparador
adicional de prevalencias de cohorte aprendido solo en fit: mediaAUC0,549012.

Contraste principal: diferencia pareada de media de AUC cruda de cinco folds,
candidato menos control. Bootstrap2000 por paciente estratificado fold/clase.
Gate prefijado para motivar una réplica: media≥0,60, delta≥0,01, límite inferior
IC95% del delta>0 y mejora en al menos3/5 folds. No es una confirmación externa.
La cohorte y el resto de contrastes son secundarios, sin inferencia conjunta
corregida por toda la búsqueda histórica.

Políticas secundarias elegidas en selección de la misma pareja: especificidad
objetivo95% y macroF1; Platt no negativo C1. No ajustar umbrales con exterior.
No unir calibración de parejas para presentarla como validada en un ensemble10.
Informar media±DE por fold, AUC agrupada, cohortes, AP, Brier/logloss, FP/FN y
curvas fit/selección sin aumentos ni dropout, mismas pérdidas por paciente.

## Comprobación y entrega

Antes de pacientes: tests de identidad en eval, acoplamiento de escalas,
estadísticas constantes/AMP, inicialización idéntica, métricas por paciente,
separación de roles y reanudación exacta con RNG de MixStyle. Prueba sintética
CUDA con cuatro pasos AdamW/AMP y carga exacta de pesos en CNN original.
No se cuentan estas pruebas como entrenamientos del ensayo.

Fuentes, roles, artefactos de control y versiones anteriores se verifican por hash.
No se cambia código/configuración de una ejecución sana. Fallo real que cambie
el protocolo exige otra salida; no mezclar resultados. Preservar checkpoints.

Train tiene historial adaptativo; selección comparte checkpoint, pérdida,
calibración y umbral. No llamar a sus cifras validación independiente. No abrir
test ni privada, no activar modelo, no push ni nuevas ramas. Tras20 ejecuciones
se documenta el resultado y se detiene esta tanda, sin búsqueda automática de
otra configuración. Si falla, retirar código experimental conservándolo en
historia de main y registrar el descarte conforme a CONTRIBUTING.md.

Comandos:

```bash
.venv/bin/python -B -m cancer_mama.regularizacion_estilo comprobar
.venv/bin/python -B -m cancer_mama.regularizacion_estilo preparar
.venv/bin/python -u -B -m cancer_mama.regularizacion_estilo entrenar
```


## Cierre

Completado: mediaAUC0.557581, control0.556270,
delta+0.001311 con intervalo que incluye cero. No alcanza0,60.
Código experimental retirado; reproducible en commit local7f9b3f2f96c58e3261f344d25d52f77a6804c199.
[Informe completo](../../resultados/30_regularizacion_estilo_20261026/INFORME_RESULTADOS.md).
No se activa un candidato ni se abre otra tanda.
