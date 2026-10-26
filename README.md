# Rama `party`: PartyNet, pCR solo con imágenes

Rama limpia para construir el mejor modelo de solo imagen posible. Todo el
historial anterior (V001–V008, aplicaciones, informes 01–31) sigue intacto en
`main`, y aquí se puede consultar con `git show main:<ruta>`.

**PartyNet** es una CNN 2D propia entrenada desde cero, sin pesos
preentrenados. Recibe PRE/EARLY/LATE, deriva internamente los realces firmados
y devuelve **un único logit de pCR por corte**. La probabilidad de una paciente
es la media de las de sus cortes. El código permite además tareas auxiliares
de entrenamiento (HR, HER2 y volumen), que nunca son entrada del modelo; se
probaron y se descartaron.

**Estado (26/10/2026):** modelo final `modelos/party_final/` con la receta
`media` y pérdida normal: AUC media de CV 0,584 ± 0,036, frente a 0,549 de la
referencia de solo cohorte. Test aún sin evaluar. Resultados, curvas y
decisiones en [docs/LECCIONES.md](docs/LECCIONES.md) §6–7.

## Estructura

```text
pcr_party/
  datos.py       índice, validación de PNG, caché uint8 y roles por paciente
  modelo.py      PartyNet (593.716 parámetros)
  entrenar.py    receta, aumentos en GPU, EMA, bucle y predicción con TTA
  cv.py          validación cruzada de 5 folds y resumen con bootstrap
  final.py       paquete final con todo train, Platt y umbral desde las OOF
  inferencia.py  carga del paquete, predicción y evaluación única de test
tests/           pruebas unitarias (CPU, datos sintéticos)
docs/LECCIONES.md
```

`breastdcedl` y `.venv` son enlaces simbólicos a la copia de `main`. `cache/` y
`runs/` se ignoran en Git; de `modelos/` solo se versiona el paquete final.

## Cómo entrenar

Todo se ejecuta desde la raíz de esta carpeta
(`/home/jorgesrab/Escritorio/caso-cancer-mama-party`), en el equipo con la GPU.

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m pcr_party cache
.venv/bin/python -m pcr_party humo
```

### 1. Validación cruzada

Hay tres recetas en `pcr_party/entrenar.py`: `control` (todos los cortes en
cada época), `lenta` y `media` (3 cortes al azar por paciente y época). La
elegida es `media`.

```bash
nohup .venv/bin/python -m pcr_party cv --nombre cv_media --receta media --lambdas 0 \
    --control cv_party > runs/cv_media.out 2>&1 &
tail -f runs/cv_media.out
```

Con 5 folds × 3 semillas × 2 pérdidas son 30 redes, ≈35 min. `--control`
añade la comparación pareada con otra CV hecha con los mismos roles y semillas.
Antes de una CV se puede calibrar una receta con una sola red, mirando solo
fit y selección:

```bash
.venv/bin/python -m pcr_party piloto --receta media --fold 0 --ajuste lr=2e-4
```

La CV original del control con tareas auxiliares (60 redes, ≈2 h) se lanzó con
`cv` sin opciones.

El bucle va semilla a semilla, así que al terminar la primera semilla ya hay
un `RESUMEN.md` con los cinco folds, y se actualiza tras cada semilla. Si se corta, relanzar la misma orden continúa
donde se quedó: las redes con `hecho.json` no se repiten.

Salidas en `runs/<nombre>/`: `RESUMEN.md`, `resumen.json`, `curvas.png`
(fit frente a selección), `oof_<brazo>.csv`, y por cada red su historia,
predicciones y pesos.

Cómo leer el resumen:

- **Media AUC ± DE** de los 5 folds: la métrica principal. Comparar con la
  referencia de cohorte (≈0,549), V007 (0,577) y el control 27 (0,556).
- **`ponderada_aux1 − ponderada_aux0`** (y la normal): prueba de la hipótesis
  multitarea. Comparación pareada: mismos roles, semillas, inicialización y
  aumentos.
- **AUC dentro de cohorte**: si es mucho menor que la agrupada, el modelo está
  aprendiendo sobre todo la cohorte.
- **AUC HR / HER2 aux**: si en los brazos aux1 salen ≈0,5, las cabezas
  auxiliares no aprendieron.
- **`<brazo> − control <cv>`**: comparación pareada con la CV de control.
- **Pico sel., brecha elegida / final y tendencia 1–14**: diagnóstico de
  sobreajuste. Tendencia = las pérdidas de fit y selección bajan juntas entre
  las épocas 1 y 14 (indicación del profesor).

### 2. Modelo final (≈3 min)

Elige el brazo con mejor media de AUC en la CV, sin mirar test, y entrena con
el 100 % de train. Es el comando con el que se creó `modelos/party_final/`:

```bash
.venv/bin/python -m pcr_party final --cv cv_media --perdida normal --lambda-aux 0
```

Las épocas son la mediana de las elegidas en la CV de ese brazo; se pueden
fijar con `--epocas N`. Usa 5 semillas. Escribe `modelos/party_final/` con los
pesos, `paquete.json` (receta, Platt, umbral Youden, métricas OOF, entorno y
SHA-256) e historias.

### 3. Predicción y test

```bash
.venv/bin/python -m pcr_party predecir --pre a_PRE.png --early a_EARLY.png --late a_LATE.png
.venv/bin/python -m pcr_party evaluar-test --confirmo-modelo-congelado
```

`evaluar-test` se ejecuta **una sola vez**, con el paquete ya congelado:
escribe `evaluacion_test.json` y se niega a repetirse. Nada del test vuelve a
la elección de brazo, épocas, calibración ni umbral.

## Reglas de la rama

- Entrada solo de píxeles; las variables clínicas solo son objetivos
  auxiliares en train.
- Separación estricta por paciente. El fold exterior no decide nada.
- Ni test ni validación privada para ajustar.
- Sin push salvo autorización expresa.
