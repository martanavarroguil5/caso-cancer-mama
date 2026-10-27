# Protocolo de la fusión tardía imagen + clínica

Fijado el 27/10/2026, antes de entrenar las redes internas y de ver ningún
resultado anidado. Solo usa train. El test público y la validación privada no
intervienen en ninguna decisión.

## Contexto y permiso

El profesor permite desde el 27/10/2026 usar datos clínicos además de las
imágenes. El enunciado exige una CNN 2D desde cero con un único logit: esa CNN
es `party_final_v2` y no cambia. La clínica se añade **después**, en una
regresión logística que recibe la puntuación de la CNN como una variable más.

Análisis previo exploratorio (OOF de `cv_presupuesto`, sin anidar, 27/10):
clínica 0,702; clínica + imagen 0,707 (Δ +0,005 [−0,005; +0,015]); subtipo
0,675; subtipo + imagen 0,694 (Δ +0,019 [+0,005; +0,033]). La puntuación de
imagen no se correlaciona con volumen ni edad (ρ ≈ 0,02) ni predice HR/HER2
(AUC 0,49/0,50). Ese análisis motivó el diseño y no sustituye a este protocolo.

## Por qué fusión tardía

- Los tres multimodales de `main` (V002, FiLM, SAM) entrenaban la red junto con
  la clínica y acabaron ignorando la imagen: quitar el término visual dejaba la
  AUC igual o mejor. No se repite.
- Con la CNN congelada, la imagen entra como un coeficiente visible y su aporte
  se mide con una comparación pareada.
- Si en la evaluación no hay clínica, la CNN sola sigue siendo un modelo
  completo (respaldo natural).

## Variables

| Grupo | Variables | Nota |
|---|---|---|
| `subtipo` | HR, HER2 (0/1) | Se usan juntas: el grupo existe si se conocen las dos |
| `edad` | edad en años | |
| `volumen` | log(1 + `tum_vol`) | Unidades de `patients.csv`; en Duke se midió de otra forma |
| `imagen` | logit de la probabilidad de paciente de la CNN | Media de cortes y de semillas |

Excluidas: cohorte y parámetros de adquisición (identifican el centro),
menopausia (falta en todo I-SPY1), raza.

## Submodelos por patrón de datos

Con datos clínicos incompletos no se imputa: se usa un **submodelo por patrón**
(Fletcher Mercaldo y Blume, *Biostatistics* 2020), uno por cada combinación de
grupos disponibles. Con tres grupos hay 8 combinaciones; cada una existe con
imagen (fusión) y sin imagen (referencia clínica). La combinación vacía con
imagen es la CNN sola recalibrada.

Cada submodelo es una regresión logística L2 (C = 1) sobre variables
estandarizadas. La estandarización se ajusta con las filas de ajuste, que son
las pacientes que tienen observadas todas las variables del submodelo. Una
paciente a la que le falte una variable se predice con el submodelo de las
variables que sí tiene.

## Puntuación de imagen sin fugas (CV anidada)

La CV exterior es la de siempre (`fold` de `samples.csv`, 5 folds).

- **Pacientes evaluadas del fold exterior k:** su puntuación es la OOF de
  `cv_presupuesto` (`ponderada_aux0`, 3 semillas), de redes que no vieron el
  fold k.
- **Pacientes de ajuste (folds ≠ k):** su puntuación sale de una CV interna que
  **excluye por completo el fold k**. Para cada fold interno j ≠ k se entrena con
  los folds ∉ {k, j} y se predice el fold j. La receta es la misma (`presupuesto`,
  pérdida ponderada, 12 épocas, última época), con un 15 % de selección
  estratificada (semilla 3000 + 10k + j) que solo vigila curvas. Son
  5 × 4 × 3 = 60 redes.

Así, ningún coeficiente de la logística se ajusta con información del fold que
evalúa. Como diagnóstico se informa también del stacking ingenuo (ajustar con las
OOF exteriores), para medir cuánto optimismo evita el anidado.

## Brazos y comparaciones

Todos los brazos se evalúan en las mismas 1.097 pacientes, con la AUC media de
los 5 folds y el bootstrap pareado estratificado por fold × clase (2.000
remuestreos) de siempre.

**Comparación principal (fijada):** `completa+imagen` − `completa` (pérdida
normal). Responde a si la imagen aporta algo sobre toda la clínica.

**Secundarias (fijadas):**

1. `subtipo+imagen` − `subtipo`: aporte de la imagen cuando solo se conoce el
   subtipo.
2. `completa+imagen` − `imagen`: aporte de la clínica sobre la imagen.
3. Pérdida ponderada (`class_weight` balanceado) frente a normal, por el
   requisito del enunciado.
4. Anidado frente a ingenuo en `completa+imagen`.

**Exploratoria (una sola, fijada):** `completa+imagen` con interacción
imagen × subtipo (la imagen discrimina más en HER2+). Solo se adoptaría si su IC
frente al brazo principal excluye el 0.

Se informan además AP, Brier, AUC dentro de cohorte y de subtipo, y los
coeficientes estandarizados de la imagen en cada fold.

## Paquete final

`modelos/party_fusion_v1/`, sin pesos nuevos de red:

- Referencia a `party_final_v2` verificada por SHA-256 de su `paquete.json`.
- Los 8 submodelos de fusión (y los 7 clínicos de referencia), ajustados con las
  1.097 pacientes de train. La variable de imagen de cada paciente es su OOF de
  `cv_presupuesto` (red que no la vio), no la predicción del modelo final sobre
  su propio train.
- Umbral de cada submodelo: Youden sobre sus OOF anidadas, con el de máximo
  macro-F1 guardado como alternativa (misma regla que el paquete de imagen).
- Se adopta el paquete de fusión aunque la comparación principal no sea
  significativa: con la clínica completa no empeora y con clínica parcial cubre
  los casos sin datos. Los resultados se informan tal cual.

Limitación conocida: en inferencia la puntuación de imagen sale de las 5 redes
entrenadas con todo train, algo más fuertes que las de la CV con las que se
ajustó la logística. Es el mismo supuesto que la calibración Platt del paquete
de imagen.

## Disponibilidad de la clínica

`patients.csv` trae la clínica del test público, así que el test podrá
evaluarse una vez con el paquete de fusión. La validación privada no la trae:
en la demo se introduce a mano y, si falta, se usa el submodelo que
corresponda; sin ninguna variable, la CNN sola.
