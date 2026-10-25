# Auditoría de train y preparación de cinco folds

25/10/2026. Se han ejecutado únicamente los pasos de auditoría y preparación, sin entrenamiento ni corrección de datos.

- 32.835 PNG,10.945 cortes,1.097 pacientes de train; sin archivos ausentes/corruptos ni imágenes idénticas entre pacientes de train.
- 18 pacientes con avisos; confirmado EARLY=LATE en los diez cortes de ISPY1_1139, con índices de origen2/2. No se excluye ni se reconstruye una fase sin autorización del responsable.
- Datos y versiones históricas conservados. Test/validation no se abren para el análisis; no se afirma ausencia de duplicados por píxeles con esos conjuntos.
- Cinco folds originales con 745–746 fit,132 selección y219–220 exterior. Calibración/umbral comparten selección; solo el exterior evalúa el procedimiento completo.
- Futuro paquete de una pareja fija,932 fit y165 selección/política; no se entrena ni activa ahora. La CV no valida por sí sola esos futuros pesos.
- 99 pruebas; código de auditoría96a98cf. Todo permanece en main local, sin push.

[Informe completo, figuras y límites](../resultados/26_auditoria_train_protocolo_20261025/INFORME_AUDITORIA.md).
[Incidencias para consultar al profesor](../resultados/26_auditoria_train_protocolo_20261025/INCIDENCIAS_PARA_PROFESOR.md).
[Protocolo y roles preparados](../resultados/26_auditoria_train_protocolo_20261025/protocolo_cinco_folds/protocolo.json).
