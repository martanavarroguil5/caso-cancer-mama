# Caso BreastDCEDL: CNN propia con PyTorch

Predicción educativa de pCR a partir de las fases PRE/EARLY/LATE. El caso
requiere una CNN 2D inicializada desde cero, un logit y BCE normal/ponderada.
Las entradas del ejercicio son PNG monocromos de 256×256 y las particiones se
separan por paciente. Este proyecto no tiene validez clínica.

## Opciones de solo imágenes

La [comparación de modelos permitidos](docs/OPCIONES_CNN_IMAGEN.md) separa
el desarrollo de la [evaluación final compartida](docs/EVALUACION_FINAL_CNN_IMAGEN.md).
Sobre las mismas 176 pacientes de test, los paquetes congelados obtienen AUC
**0,5462 (V001), 0,5039 (híbrida) y 0,6084 (V007)**. V007 tiene IC95 %
[0,5222; 0,6935]; su umbral archivado no detecta ninguna de las 53 pCR. Son
resultados retrospectivos y no una validación externa ciega ni garantía de AUC
poblacional ≥0,60. La validación privada del profesor permanece cerrada.

- [Modelos conservados](modelos/README.md).
- [CNN híbrida: informe y curvas](resultados/20_busqueda_imagen_20261025/INFORME_RESULTADOS.md).
- [CNN híbrida: pesos e inferencia](modelos/imagen_sola_20261025/README.md).
- [CNN por fases: comparación normal/ponderada](resultados/14_informe_cnn_imagen_20261014/INFORME_RESULTADOS.md).
- [CNN cinética: ajustes e informe](resultados/15_informe_cinetico_20261021/INFORME_RESULTADOS.md).
- [Piloto de regularización](resultados/16_informe_piloto_antioverfit_20261021/INFORME_RESULTADOS.md).
- [Piloto de selección de cortes](resultados/19_informe_seleccion_cortes_20261022/INFORME_RESULTADOS.md).

## Aplicación docente de un solo corte

`app_imagen.py` usa el candidato experimental V008: dos semillas de la misma
CNN cinética, calibración separada y umbral congelado para una tripleta.
Comprobación interna de 165 pacientes: AUC **0,5598**, sensibilidad **60,42 %** y
especificidad **46,15 %**; resultado insuficiente para acreditar 0,60. No se le
atribuye el 0,6084 de V007 con todos los cortes de las pacientes de test.

```bash
.venv/bin/python -m streamlit run app_imagen.py
```

[Protocolo de cierre](docs/cierre_cinetico/PROTOCOLO_UN_CORTE.md) ·
[Informe y curvas](resultados/24_cierre_cinetico_un_corte_20261025/INFORME_RESULTADOS.md).
La aplicación histórica `app.py` conserva V002; no se sustituye ese archivo.

## Entorno y pruebas

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -B -m unittest discover -s tests -v
.venv/bin/python -B modelos/versionar.py verificar
```

En Windows, usar `.venv\Scripts\python.exe`. El dataset docente
`breastdcedl/` y los checkpoints de trabajo son locales. Solo se publican
pesos de inferencia, código, configuraciones e informes.

## Inferencia de solo imágenes

CNN histórica con código congelado:

```bash
.venv/bin/python -B modelos/versionar.py predecir v001_cnn_historica_20260930 -- \
  --pre paciente_z000_PRE.png --early paciente_z000_EARLY.png --late paciente_z000_LATE.png
```

CNN híbrida por paciente:

```bash
PYTHONPATH=modelos/imagen_sola_20261025/codigo/src .venv/bin/python -B -m cancer_mama.inferencia_mil \
  --package modelos/imagen_sola_20261025 \
  --pre paciente_z000_PRE.png --early paciente_z000_EARLY.png --late paciente_z000_LATE.png
```

Los conjuntos de evaluación no se usan para ajustar umbrales ni configuraciones.
Las OOF aplican únicamente los modelos del fold que excluyó a cada paciente;
no equivalen a pasar los datos de desarrollo por todo el ensemble.

## Aplicación y archivo multimodal

La aplicación existente se inicia con `.venv/bin/python -m streamlit run app.py`.
Mantiene la versión multimodal histórica V002, separada de los candidatos de
solo imagen. Sus métricas con clínica no describen el rendimiento con solo PNG.
El código de entrenamiento y las versiones anteriores se conservan en `main`.

## Datos y atribución

Dataset BreastDCEDL, Fridman y colaboradores, Scientific Data (2026),
DOI 10.1038/s41597-026-06589-6 y Zenodo 10.5281/zenodo.18114231.
Se conserva la licencia CC BY-NC 4.0 y el formato docente original.

## Revisar train y preparar la evaluación por paciente

La auditoría exhaustiva abre únicamente imágenes de train y no modifica datos ni
modelos. Los avisos de desplazamiento son candidatos a revisión, no correcciones.
El segundo comando genera los cinco folds internos y la partición del futuro
modelo final; no ejecuta entrenamientos. Una salida ya terminada no se sobrescribe.

```bash
.venv/bin/python -B -m cancer_mama.revision_train auditar --output resultados/auditoria_train_nueva
.venv/bin/python -B -m cancer_mama.revision_train preparar-protocolo --output resultados/auditoria_train_nueva
```

[Auditoría completa y protocolo](docs/AUDITORIA_TRAIN_Y_PROTOCOLO.md).

[Criterio del profesor: curvas y comparación frente a cohorte](docs/CRITERIO_DOCENTE_CURVAS.md).

[CNN espacial compacta por corte y comparación prefijada](docs/CNN_ESPACIAL_POR_CORTE.md).
