# Protocolo del piloto: CNN entrenada sobre la clínica (offset)

Fijado el 27/10/2026, antes de escribir el código del piloto y de ver ningún
resultado. Solo usa train. El test público y la validación privada no
intervienen en ninguna decisión.

## Motivación (análisis previo, solo OOF de train)

Análisis del 27/10 sobre las OOF de `cv_presupuesto` (`runs/analisis_aporte_imagen/`):

- La puntuación de la CNN es casi independiente de la clínica: la clínica
  explica el 0,6 % de su varianza y la imagen no predice HR ni HER2 (AUC 0,49 y
  0,50). Quitarle lo explicado por la clínica deja su AUC igual (0,575 → 0,572).
- Por eso aporta poco sobre la clínica (+0,005): es independiente pero débil.
- La señal no es homogénea. AUC de la imagen por subtipo: HER2+ 0,61, triple
  negativo 0,58, HR+/HER2− 0,54. Aporte sobre edad + volumen dentro del
  subtipo: +0,039, +0,011 y −0,009. Por tercil de riesgo clínico: bajo 0,52,
  medio 0,59, alto 0,60.
- Descartados con datos: otra agregación de cortes, localizar el tumor (los PNG
  ya están centrados en él) y rasgos cinéticos clásicos (AUC ≤ 0,53, no aportan).

La CNN actual reparte su gradiente por igual entre todas las pacientes, también
las que la clínica ya resuelve. La hipótesis es que, si aprende **solo lo que la
clínica no explica**, su aporte combinado crece.

## Qué cambia

La red sigue siendo PartyNet (mismo tronco, misma receta) y **no recibe la
clínica como entrada**. Lo que cambia es la pérdida:

    logit_paciente_corte = o_clínico (fijo, no se entrena) + f_imagen(x)

`o_clínico` es el logit de la logística clínica completa (edad, log volumen, HR,
HER2; submodelos por patrón de `fusion.py`, pérdida normal). La red solo puede
mejorar la predicción a través de la imagen: a diferencia de V002/FiLM/SAM de
`main`, no hay ningún camino entrenable desde la clínica, así que no puede
aprender la clínica e ignorar la imagen.

### Brazos (solo dos, fijados)

| Brazo | Cabeza | Qué prueba |
|---|---|---|
| `offset` | 1 salida, como ahora | La red aprende el residuo de la clínica |
| `offset_subtipo` | 3 pesos de salida (HR+/HER2−, HER2+, TN) sobre la misma representación de 64, con un único sesgo compartido | Además, que la imagen pese distinto según el subtipo |

En `offset_subtipo` el subtipo solo elige qué vector de pesos se aplica; el
sesgo es común para que la cabeza no pueda reajustar el efecto principal del
subtipo, que ya está en el offset. Si falta HR o HER2, se usa la media de las
tres salidas.

## Offset sin fugas

Para cada fold exterior k (los `fold` de `samples.csv`):

- **Pacientes de desarrollo (folds ≠ k, fit y selección):** offset por ajuste
  cruzado dentro del desarrollo, 5 particiones estratificadas por pCR
  (semilla 4000 + k). Ninguna paciente recibe un offset ajustado con su propia
  etiqueta.
- **Pacientes del exterior k:** offset de la logística ajustada con todo el
  desarrollo.
- Sin ningún grupo clínico disponible: logit de la prevalencia del ajuste.

## Receta

Idéntica a `cv_presupuesto` (`plan.json`): presupuesto de 12 épocas, LR lineal
hasta 0, última época, 3 cortes por paciente y época, EMA 0,98, sin auxiliares,
TTA. Mismos roles (15 % de selección, solo para curvas) y mismas semillas (42,
2026, 7), así que la comparación con la CNN actual es pareada.

Cambios obligados por el offset: **pérdida normal** (con el offset la base ya
está calibrada; `pos_weight` la desplazaría) y sesgo inicial 0.

Puntuación de paciente: media de cortes de σ(o + f), luego media de semillas.
Son 2 brazos × 5 folds × 3 semillas = 30 redes.

## Comparaciones

Mismas 1.097 pacientes, AUC media de los 5 folds y bootstrap pareado
estratificado por fold × clase (2.000 remuestreos), como en la fusión.

**Principal (fijada), para cada brazo:** brazo − `party_fusion_v1`
(`completa+imagen`, OOF anidada, 0,714).

**Secundarias:**

1. Brazo − su propio offset clínico en el exterior: aporte de la imagen, frente
   al +0,006 de la fusión tardía.
2. `offset_subtipo` − `offset`.
3. AUC por subtipo y por cohorte.

**Exploratoria:** AUC de f(x) sola (sin offset). Se espera menor que la CNN
actual porque no se entrena para predecir pCR sola.

## Criterio de adopción (fijado)

Un brazo se adopta si cumple **las dos** condiciones:

1. Δ frente a `party_fusion_v1` ≥ +0,010 en la media de folds.
2. El IC95 % de Δ frente a su propio offset clínico excluye el 0.

Si los dos brazos cumplen, se elige el de mayor Δ frente a la fusión. Si ninguno
cumple, `party_fusion_v1` sigue vigente y el resultado se documenta como
negativo. Elegir el mejor de dos brazos introduce un pequeño optimismo; se
declara.

Si se adopta, el paquete final (todas las pacientes de train, offset por ajuste
cruzado, 5 semillas, umbrales por OOF) es un paso aparte que se hará solo con la
aprobación del usuario. Sin clínica, el respaldo sigue siendo `party_final_v2`.
