# Protocolo: CNN cinética multiescala con realce DCE explícito

Fecha de fijación: 21/10/2026. Estado: **60 entrenamientos terminados; resultado
insuficiente conservado como v007**.

## Resultado y cierre (21/10/2026)

Se completaron 60 ejecuciones y 948 épocas en 3 h 17 min, con 220.884 pasos
de AdamW y ninguno omitido por AMP. No cambió la arquitectura de 593.521
parámetros del commit `39bd951` ni se cargó test. Las cinco elecciones de
parámetros y pérdida se congelaron antes de evaluar imágenes exteriores.

El candidato obtuvo AUC OOF global cruda **0,526404**, IC95 %
**[0,498565; 0,552856]**, media por fold **0,576593 ± 0,030063** y AUC global
con calibración interna **0,561803**. La referencia obtuvo 0,535981 global.
La diferencia pareada es −0,009577, IC95 % [−0,028690; +0,008952]. No hay
evidencia de mejora ni se alcanza el mínimo 0,60 o el objetivo 0,70.

Se archivan los diez pesos seleccionados como
`v007_cnn_cinetica_multiescala_20261021`, sin promoción. V002 continúa activo
y no se inicia otra búsqueda antes de revisar el resultado con el usuario.
El código de entrenamiento se conserva para reproducir esta preparación.

[Informe, métricas y curvas](../../resultados/15_informe_cinetico_20261021/INFORME_RESULTADOS.md).
La ampliación siguiente y el protocolo original se conservan como trazabilidad;
el lanzador PowerShell mantiene la matriz original de 20, no la búsqueda de 60.

## Ajuste autorizado de entrenamiento, con arquitectura fija (21/10/2026)

Tras revisar el candidato, se autoriza entrenar y ajustar su optimización para
alcanzar un AUC mínimo de 0,60 y un objetivo de 0,70. Esto amplía el ensayo
original de 20 ejecuciones antes de consultar sus resultados. No se cambian
la clase `CNNMultiescalaCinetica`, sus 593.521 parámetros, filtros, recortes,
mapas cinéticos, BatchNorm, pooling, clasificador ni tasas de dropout.

Se prefijan tres combinaciones de AdamW:

| Nombre | Tasa de aprendizaje | Weight decay |
|---|---:|---:|
| referencia | 0,0008 | 0,001 |
| lr_menor | 0,0003 | 0,001 |
| lr_menor_wd_mayor | 0,0003 | 0,01 |

Cada combinación usa BCE normal/ponderada, semillas 42/2026 y cinco folds:
60 entrenamientos. Se mantienen lote 32, aumentos, máximo 40 épocas, parada
temprana de ocho épocas y selección del checkpoint por AUC interno de paciente.
Cada fold elige entre seis alternativas por el AUC de las probabilidades
medias de sus dos semillas sobre los mismos pacientes de selección interna.
En empate exacto se prefiere referencia, menor weight decay y BCE normal.

Durante la búsqueda solo se cargan imágenes de fit y selección interna. Primero
se guardan las cinco elecciones, sus pesos y hashes; después se evalúan los
folds exteriores. El control usa la configuración referencia y escoge su pérdida
también solo dentro de cada fold. Se informan AUC OOF agrupado, media de folds,
calibración, sensibilidad, especificidad y matriz de confusión; el umbral y la
calibración se ajustan internamente, con especificidad objetivo del 90 %.
El historial previo hace que estas OOF sigan siendo evidencia de desarrollo,
pendiente de confirmación independiente. El test reservado permanece cerrado.

La monitorización revisa pesos y pérdidas finitos, pasos de AdamW omitidos por
AMP, brecha fit/selección y parada temprana. Una menor brecha por sí sola no
demuestra mayor discriminación. Los cambios posteriores al protocolo se
registrarán como otro ensayo; no se cambiará una ejecución en marcha.

Ejecución en Linux con GPU:

```bash
.venv/bin/python -B -m cancer_mama.ajuste_cinetico verificar
.venv/bin/python -B -m cancer_mama.ajuste_cinetico prueba
.venv/bin/python -u -B -m cancer_mama.ajuste_cinetico entrenar
```

Salida de trabajo ignorada por Git:
`resultados/04_entrenamiento/ajuste_cnn_cinetica_20261021/`. La acción `prueba`
realiza un lote por pérdida y no participa en la selección. El ajuste guarda
estado, configuración, código, predicciones y checkpoints reanudables. Un
bloqueo de proceso impide iniciar dos lanzadores simultáneos.

El protocolo original que se conserva debajo documenta la hipótesis y el plan
inicial. La ampliación anterior gobierna esta ejecución autorizada.

## Motivo e hipótesis

La CNN solo imagen por fases obtuvo AUC OOF agrupado 0,5087 con BCE normal y
0,5246 con BCE ponderada. Su fusión resta características después del encoder;
BatchNorm y las no linealidades pueden haber eliminado parte de la magnitud o
del signo del realce DCE.

Un probe con seis estadísticas cinéticas simples obtuvo AUC OOF 0,5337 e IC95 %
[0,4972, 0,5701]: el realce global aislado ofrece una señal débil. Además,
aproximadamente el 73 % de los píxeles es fondo. Por ello el candidato inicial
de una sola escala se sustituyó antes de entrenarlo por un paquete más fuerte.
El modelo histórico con fusión temprana de los tres tiempos alcanzó AUC OOF
0,5945 (aunque solo 0,5462 en test), mientras que el control posterior con
encoder por fase y fusión tardía bajó a 0,5246. La nueva arquitectura conserva
la pista favorable de la fusión temprana:

1. mapas firmados `EARLY-PRE`, `LATE-PRE` y `EARLY-LATE` antes de convolucionar;
2. vistas centradas de 256, 128 y 64 píxeles para combinar contexto, entorno y núcleo;
3. las seis entradas temporales se fusionan desde la primera convolución;
4. Dropout2d 0,10 después del backbone para contener el sobreajuste observado.

No se añaden variables clínicas, preentrenamiento, focal loss, máscaras obtenidas
de las etiquetas ni ajustes del umbral para inflar el AUC. Es un candidato
compuesto: si mejora, este experimento no podrá atribuir el efecto a uno solo de
sus tres elementos.

Como comprobación exploratoria previa, un probe fijado con 12 descriptores
seleccionados dentro de cada fold, calculados a las tres escalas sobre el corte
central de cada paciente, obtuvo AUC OOF 0,5552, media de folds 0,5615 ± 0,0338
e IC95 % bootstrap [0,5180, 0,5905]. No se usó test. Este diagnóstico respalda
probar las escalas, pero no estima el rendimiento de la CNN ni constituye una
confirmación independiente.

## Arquitectura

La red sigue siendo una CNN 2D propia entrenada desde cero. Contiene:

1. tres recortes centrales de 256, 128 y 64, remuestreados a 128×128;
2. seis mapas ordenados: PRE, EARLY, LATE, EARLY-PRE, LATE-PRE y EARLY-LATE;
3. fusión temprana en un backbone de cuatro bloques `[24, 48, 96, 160]`, el
   patrón del modelo histórico cercano a 0,60;
4. los mismos filtros y BatchNorm compartidos entre las tres escalas;
5. Dropout2d, pooling global medio/máximo por escala y clasificador binario.

Procesar las tres escalas como un único lote evita actualizar BatchNorm tres
veces con distribuciones distintas. Las fases representan tiempos ordenados y
nunca se tratan como RGB. Las intensidades mantienen la escala `[0,1]` del
modelo histórico; las diferencias quedan en `[-1,1]` y conservan el signo.

## Comparación prefijada

El protocolo de evaluación permanece igual que en la CNN por fases:

- cinco folds exteriores por paciente y selección interna del 15 %;
- semillas 42 y 2026;
- BCE normal y ponderada: 20 ejecuciones en total;
- media de cortes como predicción de paciente;
- mismo optimizador, aumentos, parada temprana, épocas y regla de checkpoint;
- umbral elegido solo en validación con especificidad mínima del 90 %;
- test reservado sin cargar durante desarrollo.

La pérdida se selecciona por la media del AUC exterior de los cinco folds. A
continuación se realiza un bootstrap estratificado y pareado entre sus
predicciones OOF y las de la CNN por fases congelada. Solo se considera apoyada
la adopción si se cumplen dos condiciones: AUC OOF agrupado al menos 0,60 y
límite inferior del IC95 % de `AUC_cinética − AUC_fases` mayor que cero. Esta
es una comparación adaptativa de desarrollo, no un test independiente, y llegar
a 0,60 OOF no garantiza repetir esa cifra en test. El informe calcula además la
diferencia pareada frente al modelo histórico de 0,5945 para saber si el cambio
realmente supera la mejor referencia previa, aunque no se usa para alterar el
protocolo una vez vistos los resultados.

## Ejecución en el ordenador con GPU

Comprobar entorno, tests y configuración sin entrenar:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/entrenar_cnn_cinetica_gpu.ps1 -SoloComprobar
```

Ejecutar la prueba corta, los 20 entrenamientos reanudables y la comparación:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/entrenar_cnn_cinetica_gpu.ps1
```

La salida queda en
`resultados/04_entrenamiento/cnn_imagen_cinetica_multiescala_20261021/`. Hay que copiarla
completa de vuelta. Las ejecuciones cortas se guardan en `pruebas/` y no son
elegibles. Repetir el comando con la misma configuración reanuda checkpoints.

## Interpretación prevista

El AUC no cambia al desplazar el umbral: el umbral solo cambia sensibilidad,
especificidad y la matriz de confusión. Por eso la decisión arquitectónica se
hará con AUC OOF pareado; el punto operativo se informará después. Si no hay
mejora compatible con el criterio prefijado, se documentará el resultado y no
se seguirá optimizando contra los mismos folds.
