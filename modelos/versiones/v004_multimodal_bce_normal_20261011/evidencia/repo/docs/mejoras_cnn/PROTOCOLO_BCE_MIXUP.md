# CNN desde cero: BCE normal y mixup multimodal

Protocolo fijado antes de entrenar esta tanda. Se conserva la CNN original de ocho
convoluciones, 551.921 parámetros, entrada PRE/EARLY/LATE PNG 256×256, cuatro
variables clínicas e indicadores de ausencia. No se importa ni usa ningún modelo
preentrenado. Es compatible con la sección 7 del enunciado docente.

## Restricciones del caso

`breastdcedl/documentation/caso_breastdcedl.pdf`, páginas 4–6: fases alineadas,
sin normalización ImageNet ni aumentos de color; CNN 2D desde cero, un logit,
BCEWithLogitsLoss y comparación normal/ponderada. Página 6: las entradas de la
aplicación son tres PNG de un corte. No se usa una red 3D ni se descargan NIfTI.

Ninguna de ellas modificó la versión activa v002.

## Hipótesis y referencias

La BCE ponderada cambia el objetivo probabilístico; interesa comprobar si BCE
normal reduce la log-loss sin perder discriminación. Mixup interpola datos y
etiquetas para regularizar la red. Se ensaya como hipótesis, no como mejora
garantizada ni como reproducción de un resultado clínico.

- [Zhang et al., ICLR 2018](https://arxiv.org/abs/1710.09412): mixup como regularización.
- [Zhang et al., ICML 2022](https://proceedings.mlr.press/v162/zhang22f.html): condiciones bajo las que mejora calibración.
- [Wang et al., CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_On_the_Pitfall_of_Mixup_for_Uncertainty_Calibration_CVPR_2023_paper.html): limitaciones cuando se añade calibración posterior.

## Comparación fijada

1. `bce_normal`: BCEWithLogitsLoss sin ponderación.
2. `bce_mixup`: lo mismo, con lambda ~ Beta(0,2; 0,2), mezclando imágenes,
   diseño clínico y etiqueta con la misma permutación y coeficiente.
3. Control ponderado: se reutilizan los entrenamientos completos del ensayo LR
   0,0008, con pos_weight=N0/N1, selección interna AUC y log-loss. No se reentrenan
   innecesariamente ni se utiliza el rendimiento histórico optimista de v002.

La inicialización clínica logística de los dos brazos nuevos se ajusta sin pesos
solo en fit, coherente con BCE normal; la rama de imagen empieza a contribuir con
peso cero y aprende desde inicialización aleatoria Kaiming. Todos sus pesos siguen
siendo entrenables. Se conserva la normalización 2*x−1 y el aumentado geométrico
de v002, sincronizado entre fases. Se imputa y estandariza clínica antes de mixup:
no se propagan NaN ni se mezclan imágenes con clínica de otras parejas.

AdamW LR 0,0008, weight_decay 0,001, dropout 0,35, batch 64, grad clip 5,
coseno hasta LR 0,00001; máximo 46 épocas, mínimo 12, paciencia 10, progreso
mínimo 0,001. Checkpoint por log-loss de paciente en selección interna, incluyendo
época 0. Semillas 42 y 2026, cinco folds, mismas particiones que el control
(15% interno estratificado por cohorte y pCR, seed de partición 20261010+fold).

Se conservan todos los cortes, una BCE por corte y el peso empírico que esto da a
cada paciente, igual que el control. Las métricas se calculan por paciente. Las
curvas fit/selección se miden sin aumentos, sin dropout y FP32. La pérdida con
mixup de optimización no se presenta como directamente comparable a esas curvas.

Inferencia principal de los nuevos brazos: probabilidades crudas; promedio de
semillas por corte y mediana por paciente. Umbral fijado con selección interna.
Platt ajustado en selección interna se reporta solo como diagnóstico secundario;
no se escoge entre calibrada/cruda por rendimiento exterior.

## Criterio de evidencia y límites

Control principal: CNN ponderada elegida por AUC con Platt interno; el control
ponderado elegido por pérdida se muestra también. Misma población y folds.
Se exige límite superior del intervalo de delta log-loss < 0, límite inferior de
delta AUC media ≥ −0,01 y Brier no mayor. Bootstrap pareado por paciente, 2.000
réplicas, estratificado por fold/cohorte/clase, alpha=0,05/12 por las diez familias
exploradas en esta búsqueda y los dos nuevos brazos. La corrección no cubre toda
la historia ni vuelve independiente una búsqueda adaptativa. Un resultado favorable
será una señal interna que requiere comprobar estabilidad, no validación externa.

Los folds exteriores no deciden épocas, calibración, mezcla, umbral ni LR. No se
cargan imágenes del test ni de validación privada. No se promueve automáticamente.

Se guardan hash de fuentes/datos/PNG/caché y controles, configuración, particiones,
entorno, tiempo por época, memoria máxima, checkpoints, estado de optimizador,
scheduler, AMP, RNG e historiales para reanudación verificable. Caché uint8 en GPU
solo para acelerar lectura; las operaciones y escalas son las de la CNN original.

```bash
.venv/bin/python -B -m cancer_mama.experimento_bce_mixup --prueba
.venv/bin/python -u -B -m cancer_mama.experimento_bce_mixup
```
