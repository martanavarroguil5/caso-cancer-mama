# Demostración BreastDCEDL de un corte

Usuario y tarea: estudiante y profesor durante una defensa; cargar tres PNG,
comprobar alineación y registrar una predicción reproducible. Sensación: lectura
clara de una lámina, sin mensajes de éxito clínico ni ajustes sobre muestras.

Dominio: corte axial, PRE, EARLY, LATE, realce firmado, pCR, umbral congelado,
incertidumbre, versión de pesos. Colores: papel gris #f5f7f8, tinta azul #183244,
placa oscura #101820, acción cian #187d86, gris de fase #748194 y aviso ámbar
#6a4c1d. El color comunica acción o limitación, no diagnóstico.

Firma: una misma línea de corte validado encima de PRE → EARLY → LATE → realce.
El resultado queda debajo y se invalida si cambia cualquier fase.

Sustituciones de patrones genéricos: tarjetas KPI por lámina de cuatro paneles;
selector de umbral por valor congelado de solo lectura; resultado verde/rojo por
probabilidad y clase con limitación educativa explícita.

Profundidad por bordes suaves, superficies en el mismo papel, fotografía sobre
placa oscura. Tipografía system-ui para lectura y números tabulares en métricas.
Unidad de espaciado 8 px, radios 4–5 px. Estado vacío claro, botón deshabilitado
hasta tripleta válida, errores de formato visibles y carga de pesos con estado.
Metadatos de modelo en expander; no deben competir con las tres fases.
