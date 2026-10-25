# Opciones de CNN 2D con entrada solo imagen — 25/10/2026

El modelo del caso debe construirse en PyTorch desde pesos aleatorios, recibir
PRE/EARLY/LATE PNG alineadas, producir un logit y entrenarse con
BCEWithLogitsLoss. El caso exige comparar pérdida normal y ponderada y separar
pacientes. Las transformaciones geométricas se comparten entre fases.

## Evaluación final compartida

La comparación posterior de los paquetes congelados, sobre las mismas 176
pacientes de test, obtiene V001 **0,5462**, híbrida **0,5039** y V007 **0,6084**.
Son resultados diferentes de las OOF de desarrollo de la tabla siguiente.
V007 tiene IC95 % [0,5222; 0,6935] y su umbral archivado detecta 0 de 53 pCR;
no se interpreta la AUC puntual como garantía de superar 0,60 ni como una
política de clasificación satisfactoria. El test tiene antecedentes de consulta.

[Evaluación final y límites](EVALUACION_FINAL_CNN_IMAGEN.md).

## Resultados conservados

| Opción | Pacientes de desarrollo | AUC OOF agrupada | Media AUC por fold | Alcance |
|---|---:|---:|---:|---|
| V001, CNN histórica propia | 1097 | 0,5945 | Véase evaluación original | Selección histórica; test original ya evaluado |
| CNN híbrida por paciente del 25/10 | 1097 | 0,5638 calibrada; 0,5245 cruda | 0,5673 ± 0,0352 | Selección interna de receta y tres semillas; desarrollo adaptativo |
| V007, CNN cinética 2D | 1097 | 0,5264 cruda; 0,5618 calibrada internamente | 0,5766 ± 0,0301 | Sesenta ajustes y elección interna por fold |
| V006, CNN por fases ponderada | 1097 | 0,5246 cruda | 0,5486 ± 0,0504 | Comparación equiparada con BCE normal |
| V005, CNN por fases normal | 1097 | 0,5087 cruda | 0,5330 ± 0,0253 | Control normal de V006 |
| Piloto anti-sobreajuste ponderado, p001 | 439 | 0,5386 cruda | 0,5571 | Solo folds 0 y 3; exploratorio |

La CNN histórica tiene la mayor AUC agrupada conservada entre estas opciones,
pero esa cifra no demuestra AUC 0,60 en pacientes nuevas. Su test original,
evaluado el 30/09/2026, obtuvo **0,5462** sobre 176 pacientes, con IC95 %
[0,4588; 0,6352]. No se vuelve a evaluar ni se utiliza para ajustar.

No se puede escoger una opción como ganadora cambiando entre AUC agrupada,
media de folds y AUC calibrada: son cantidades diferentes y los protocolos
históricos tampoco son equivalentes. Ninguna opción demuestra actualmente
AUC independiente de al menos 0,60.

## Sobreajuste y limitaciones

- V007: brecha mediana fit–selección de 0,0583 en el checkpoint y 0,2689 al
  final. La parada temprana conserva pesos anteriores, pero no elimina el problema.
- V006: brecha mediana final de 0,1328; en el checkpoint elegido, 0,0067.
- CNN híbrida 25/10: brecha mediana fit–selección de 0,0433; fit–outer de
  0,0762. Las curvas son menos divergentes, pero la discriminación sigue débil.
- Una brecha pequeña también puede reflejar poco aprendizaje. El historial de
  búsquedas hace que las OOF sigan siendo desarrollo, no validación independiente.

## Uso y preparación de entrega

Para una base simple y defendible, V001 ya dispone de diez pesos y un contrato
PNG compatible. Para continuar investigando con una separación interna más
explícita, la CNN híbrida 25/10 conserva quince pesos y recetas reproducibles.
Su comparación completa normal/ponderada de la misma receta queda pendiente;
los resultados existentes no se presentan como si esa comparación estuviera hecha.
Los pilotos espaciales solo tienen selección interna y no se ordenan junto a
modelos con evaluación exterior completa.

El multimodal histórico se conserva separado de esta comparación: sus cifras
con entradas clínicas no se atribuyen a un modelo de solo imágenes. No se
activa automáticamente otra versión ni se inicia un nuevo entrenamiento.

## Evidencia

- [CNN histórica y sus pesos](../modelos/versiones/v001_cnn_historica_20260930/).
- [Informe de la CNN híbrida](../resultados/20_busqueda_imagen_20261025/INFORME_RESULTADOS.md).
- [Paquete de la CNN híbrida](../modelos/imagen_sola_20261025/README.md).
- [Informe cinético](../resultados/15_informe_cinetico_20261021/INFORME_RESULTADOS.md).
- [Comparación de pérdidas por fases](../resultados/14_informe_cnn_imagen_20261014/INFORME_RESULTADOS.md).
- [Piloto anti-sobreajuste](../resultados/16_informe_piloto_antioverfit_20261021/INFORME_RESULTADOS.md).
- [Selección de cortes](../resultados/19_informe_seleccion_cortes_20261022/INFORME_RESULTADOS.md).

## Cierre de la entrada de un corte: V008 experimental

La misma arquitectura cinética se entrenó con cuatro roles separados y dos
pérdidas/dos semillas. V008 conserva BCE normal, elegida solo en selección.
En check interno de 165 pacientes con una tripleta cada una: AUC 0,5598,
IC95 % [0,4596; 0,6519], sensibilidad 60,42 % y especificidad 46,15 % al umbral
0,292333096. Calibración y umbral se fijaron con otras 165 pacientes.

Es un candidato funcional para `app_imagen.py`, con resultado insuficiente;
no una mejora validada sobre V007 ni una nueva evaluación del test. No se
mezcla esa comprobación de un corte con las cifras anteriores de todos los cortes.

[Informe completo](../resultados/24_cierre_cinetico_un_corte_20261025/INFORME_RESULTADOS.md).
