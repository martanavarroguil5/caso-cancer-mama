# Protocolo de cierre para una tripleta PNG

Transcripción del protocolo JSON fijado antes de los cuatro entrenamientos del
25/10/2026. El registro autoritativo es
`resultados/24_cierre_cinetico_un_corte_20261025/protocolo.json` y su SHA-256;
esta explicación no se presenta como un nuevo preregistro posterior al resultado.

1. Mantener CNNMultiescalaCinetica de V007, 593.521 parámetros, mapas,
   filtros, recortes y dropout. Dos semillas 42/2026, BCE normal/ponderada;
   cuatro ejecuciones, sin búsqueda adicional. LR 0,0003 y WD 0,01 son la
   opción modal de las cinco elecciones internas originales, antes de test.
2. Solo train docente, 1.097 pacientes. Separación estratificada por cohorte/clase
   y por paciente: fit 657, selección 110, calibración 165, comprobación 165.
   Semilla 20261025 y offsets secuenciales para los tres grupos retenidos.
3. Fit usa todos sus cortes con contribución total uno por paciente. Pos_weight
   se obtiene solo de las pacientes de fit. Aumentos geométricos compartidos
   entre fases. Guardar época con mayor AUC de corte ponderada por paciente.
4. Seleccionar pérdida por AUC del ensemble de dos semillas en selection,
   empate a favor de BCE normal. Congelar sus dos pesos antes de calibration.
5. Ajustar Platt C=1, pendiente no negativa, con pacientes excluidas de fit y
   selection, sobre la misma media de dos semillas que usará la aplicación.
   Ajustar umbral por balanced accuracy ponderada por paciente en calibration.
6. Congelar política y hashes. Cargar entonces imágenes de check y evaluar
   primariamente una tripleta por paciente, escogida por hash de semilla
   734921 e identificador, sin mirar etiqueta ni predicción. Intervalos por
   bootstrap de pacientes, 2000 repeticiones estratificadas por cohorte/clase.
7. Análisis secundarios: distribución de cortes con peso por paciente y media
   de todos los cortes. No cambiar la métrica principal tras ver resultados.
8. Guardar candidato experimental aunque check sea débil; no volver a ajustar
   contra check. Test y validación privada cerrados durante todo el cierre.
   Train tiene antecedentes adaptativos: no es nueva confirmación externa.
9. Empaquetar los dos pesos elegidos, inferencia segura, fuentes y hashes como
   siguiente versión libre; aplicación de un corte, sin clínica ni controles de
   umbral. La aplicación histórica V002 y V007 no se sustituyen automáticamente.

## Verificación

El runner comprueba finitud, pasos AdamW, reanudación por época, invariancia de
fuentes y versiones previas, y separación de pacientes. Los checkpoints con
optimizer/RNG son locales. El código público y los pesos de inferencia no
incluyen datasets, imágenes privadas ni etiquetas de la validación docente.

[Informe de resultados](../../resultados/24_cierre_cinetico_un_corte_20261025/INFORME_RESULTADOS.md).
