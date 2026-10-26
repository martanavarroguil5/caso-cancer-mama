# Auditoría de calibración y decisión de V007

26/10/2026. Auditoría autorizada, sin entrenamiento ni publicación. V007 y las
versiones previas permanecen inmutables; no se cambia la aplicación.

## Problema y alcance

El paquete de V007 promedia diez redes. Su calibración global y umbral se
ajustaron sobre OOF donde cada paciente fue predicha por dos redes, con mezcla
de pérdidas normales/ponderadas y recetas distintas. Se comprueba si hay
incompatibilidad de escalas y se separa de la posible debilidad de discriminación.
No se atribuye automáticamente la sensibilidad nula histórica a esa diferencia.

La auditoría reconstruye roles y políticas originales, comprueba hashes de
pesos, calibración global, umbral equivalente crudo y contrato PNG/255+medias.
La inferencia nueva solo utiliza PNG sintéticos; no se cargan imágenes de
pacientes ni datos de test/privada. Las comparaciones reutilizan selección y
OOF históricas de train. Se describen como desarrollo adaptativo, no evaluación
independiente nueva. La marca test_evaluated_once=False dentro del paquete es
una instantánea archivada previa al test ya documentado; no permite repetirlo.

## Políticas prefijadas

- Principal nueva: umbral que maximiza macroF1 en selección interna de cada
  pareja de semillas, con su calibración Platt no negativa ajustada allí.
- Referencia: sensibilidad máxima con especificidad mínima90% en la misma
  selección. Debe reproducir las decisiones OOF históricas exactamente.
- Secundarias: especificidad mínima95%,0,5 crudo,0,5 calibrado y siempre no pCR.

Se congelan las cinco políticas antes de consultar las OOF para evaluar sus
decisiones. No se escoge otra receta después de observar exterior. Las mismas
pacientes ya sirvieron a selección de checkpoint/configuración, lo que hace
optimistas las métricas de selección; la evaluación exterior está separada de
esa selección por paciente pero tiene antecedentes globales adaptativos.

Principal: diferencia de macroF1 frente a la política90%, con IC95% pareado de
2000 bootstrap por fold/clase. Informar sensibilidad, especificidad, precisión,
F1 de ambas clases, FP/FN/TP/TN, exactitud y baseline. Intervalos de sensibilidad
y especificidad son secundarios, sin interpretarlos como prueba conjunta.

Se conserva el AUC de las mismas puntuaciones, que no cambia al variar umbral.
Platt creciente preserva AUC dentro de cada fold; su AUC agrupada es otra
cantidad. Brier/logloss no aíslan calibración de discriminación. Para diagnóstico
se muestran curvas suavizadas con kernel gaussiano fijo0,06, dominio percentiles
1–99 y bandas exploratorias200 bootstrap fold/clase; no se ajusta un modelo
desplegable con esas curvas ni se oculta el sesgo de suavizado.

## Límite de despliegue

Las políticas nuevas pertenecen a parejas, no al ensemble de diez, y se marcan
deployable=False. Se calcula la intersección de pacientes nunca usadas para fit
y para selección por ninguna red. No se ajusta política con inferencias de las
diez redes sobre su propio train ni se afirma que eso sea validación.

Si no hay reserva común suficiente, solo se prepara la decisión sobre un futuro
paquete fijo con su propia validación; no se ejecuta otro entrenamiento en esta
auditoría. El diseño932fit/165selección-política existente no garantiza mantener
los resultados de V007. La confirmación final corresponde a una evaluación
independiente permitida por el profesor.

## Reproducción

```bash
.venv/bin/python -B -m cancer_mama.auditoria_politica preparar
OPENBLAS_NUM_THREADS=1 .venv/bin/python -u -B -m cancer_mama.auditoria_politica analizar
```

Salida resultados/29_auditoria_politica_v007_20261026/. Fuente, archivos de
selección, OOF, roles, políticas y versiones protegidos mediante SHA256.
104 pruebas antes de ejecutar. Solo main local, sin push, sin activar modelo.

[Umbral y validación](https://scikit-learn.org/stable/modules/classification_threshold.html).
[Calibración y ensembles](https://scikit-learn.org/stable/modules/calibration.html).
