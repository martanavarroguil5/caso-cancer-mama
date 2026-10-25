# Curvas de pérdida y comparación contra cohorte

25/10/2026. El usuario autoriza aplicar la indicación del profesor: buscar
aprendizaje conjunto de train/validación hasta unas12–14 épocas y AUC por paciente
superior a la referencia de cohorte, orientativamente0,54.

Se registra una tanda acotada de20 entrenamientos, sin cambiar arquitectura ni
reabrir test. CNNMultiescalaCinetica propia,593.521 parámetros, mismos recortes,
filtros, dropout, aumentos geométricos, LR0,0003 y WD0,01. Cinco folds originales,
dos semillas y BCE normal/ponderada. Usa los roles previamente auditados:
745–746 fit,132 selección y219–220 exterior por fold.

El cambio principal respecto al protocolo preparado es seleccionar checkpoint y
parada por pérdida interna, en vez de por máximo AUC. Para evaluación, ambas
pérdidas se calculan en modo eval sin aumentos/dropout, después de la media de
todos los cortes de cada paciente, con el mismo pos_weight del fit. Se conserva
separada la BCE de optimización por corte, con contribución1/n_cortes y
normalizador constante. La ponderada no se compara por magnitud con la normal:
la elección entre pérdidas usa AUC interna de la pareja de semillas.

Se observan como mínimo14 épocas, máximo40, paciencia8 y mejora mínimaBCE0,001.
Un checkpoint anterior sigue siendo elegible. No se fuerza una época12, no se
suavizan las curvas para ocultar puntos, y no se confunde alargar un entrenamiento
con eliminar sobreajuste. Diagnóstico prefijado1–14: pendiente negativa en ambas
pérdidas y promedio12–14 menor que1–3. Se informan también época óptima y brecha
AUC fit−selección. Una brecha pequeña con ambas AUC malas puede ser infraajuste.

Se conserva el mejor checkpoint por AUC de las mismas trayectorias como
diagnóstico secundario prefijado. Comparte la parada por pérdida, de modo que
no estima causalmente dos recetas completas. No se elegirá la regla mirando
exterior ni se seguirá buscando contra estos resultados automáticamente.

El comparador de cohorte usa solo prevalencias por centro estimadas en fit,
nunca etiquetas exteriores para aprender su orden. Su AUC media en estos roles
es0,549011615. Cohorte no entra en la CNN. Criterio puntual principal:
mediaAUC CNN>0,54 y diferencia positiva contra la referencia real. Se añade
IC95% pareado de la diferencia; una estimación apenas mayor no demuestra mejora.
Se conservan AUC agrupada y por cohorte como análisis secundarios, sin
sustituir la media de cinco folds cuando sean más favorables.

Las cinco elecciones de pérdidas y políticas se congelan antes de inferir los
folds exteriores con sus respectivas parejas. Calibración Platt no negativa y
umbral con especificidad mínima95% se ajustan en selection de la misma pareja;
su reutilización para checkpoint/pérdida/política hace optimistas sus métricas.
Cada exterior evalúa el procedimiento completo. El train tiene historia de
desarrollo adaptativo y no se presenta como confirmación externa nueva.

Se conservan todos los PNG originales; las incidencias ISPY1_1139 (EARLY=LATE)
e ISPY1_1193 (brillo LATE) siguen documentadas sin corrección ni exclusión
inferida. Ningún ajuste se justifica mirando sus etiquetas/resultados.

## Resultado y cierre

20/20 ejecuciones terminadas,289 épocas,67.337 pasos AdamW sin omisiones.
MediaAUC0,556270±0,035413 frente a cohorte0,549012; delta+0,007258,
IC95%[-0,032640;+0,047542]. OOF agrupada cruda0,512731. El suelo puntual0,54
se supera por la media, pero no se demuestra mejora frente a cohorte.
Cero de20 trayectorias cumple la tendencia conjunta1–14. Ensayo insuficiente;
no se activa, versiona ni publica un modelo ni se amplía el entrenamiento.

[Informe, curvas y verificación](../resultados/27_criterio_docente_20261025/INFORME_RESULTADOS.md).
Código retirado del árbol activo conforme a CONTRIBUTING.md, conservado en main
en db845626bfca69276217074256fc131c0bffbc01; sus comandos pertenecen a ese commit.
Los pesos/checkpoints de trabajo quedan locales e ignorados; los resúmenes,
historias, configuraciones y predicciones permanecen revisables.

[Cawley y Talbot2010](https://www.jmlr.org/papers/v11/cawley10a.html) describen el
sesgo por optimizar repetidamente criterios de selección; cinco folds sobre el
train ya observado no restablecen una evaluación ciega. Las curvas son un
diagnóstico necesario de entrenamiento, pero no una demostración suficiente
de discriminación exterior.
