# Incidencias de train para consultar al responsable del dataset

Borrador preparado; no se ha enviado ningún mensaje ni modificado ningún dato.

Al revisar únicamente el train de BreastDCEDL MinCrop encontramos:

1. **ISPY1_1139:** los diez EARLY y LATE respectivos son idénticos píxel a píxel. En patients.csv constan n_times=3, pre=0, post_early=2 y post_late=2. ¿La repetición es intencionada? ¿Puede facilitarse la exportación correcta o indicarse una política común para este caso? No podemos recuperar otra adquisición a partir de los PNG suministrados.
2. **ISPY1_1193:** LATE tiene intensidad media de foreground próxima a0,955 en los diez cortes, frente a PRE≈0,046 y EARLY≈0,059. Solicitamos confirmar si corresponde a la adquisición/escala esperada; no hemos aplicado normalización independiente ni excluido el caso.
3. Existen casos con cambios de contorno entre fases (por ejemplo ISPY1_1069). La correlación de gradientes es solo una alarma, no confirma error de correspondencia ni justifica registrar imágenes automáticamente. ¿Hay información sobre el alineamiento aplicado al generar MinCrop?

No hemos utilizado test ni validación privada para detectar o decidir estas incidencias. La mayoría de train supera los controles de formato y correspondencia. Son incidencias concretas, no una explicación demostrada del bajo rendimiento global.

Evidencia: fases_repetidas_confirmadas.json y las figuras del informe de auditoría.
