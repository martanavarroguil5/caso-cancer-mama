# Modelo exclusivamente de imágenes — 25/10/2026

Quince pesos entrenados, cinco grupos de tres semillas. Cada grupo utiliza una
CNN 2D propia desde cero y descriptores derivados únicamente de los PNG. Ninguna
entrada clínica, cohorte, identificador o dato de adquisición entra en el modelo.
Los identificadores se usan solo para separar pacientes y auditar resultados.

AUC OOF calibrada de desarrollo: **0.5638**.
Media de AUC por fold: **0.5673 ± 0.0352**.
La búsqueda es adaptativa; estas cifras no son una evaluación externa. Test cerrado.
No se ha demostrado utilidad clínica. El objetivo ideal 0,70 se informa según
el resultado real, sin cambiar etiquetas, signos ni seleccionar con test.

## Inferencia sin dataset ni clínica

Desde la raíz del proyecto, usando el código congelado:

```bash
PYTHONPATH=modelos/imagen_sola_20261025/codigo/src .venv/bin/python -m cancer_mama.inferencia_mil \
  --package modelos/imagen_sola_20261025 \
  --pre paciente_z000_PRE.png paciente_z001_PRE.png \
  --early paciente_z000_EARLY.png paciente_z001_EARLY.png \
  --late paciente_z000_LATE.png paciente_z001_LATE.png
```

Los PNG deben ser monocromos de 8 bits y 256 × 256, entre uno y diez cortes
alineados de la misma paciente, en orden axial. Las tres listas deben tener
igual longitud. La evaluación usa los cortes disponibles; un único corte tiene
menos información y su rendimiento no se ha evaluado por separado.

Se reduce a 128 px con BOX; cada modelo replica sus descriptores de intensidad,
realce y/o textura. El escalado guardado procede solo del fit del fold. Las
probabilidades de tres semillas se promedian antes de aplicar su calibración
Platt interna; se promedian después los cinco grupos. El manifest incluye los
SHA-256 de todos los pesos y fuentes. La CLI comprueba los hashes de los pesos.

La ROC publicada usa únicamente el grupo que excluye a cada paciente. La media
de los cinco grupos sirve para pacientes nuevas y no debe reevaluarse sobre
train como si fuera independiente.

Informe: `resultados/20_busqueda_imagen_20261025/INFORME_RESULTADOS.md`.
