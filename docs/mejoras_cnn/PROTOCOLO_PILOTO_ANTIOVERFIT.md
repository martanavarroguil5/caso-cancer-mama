# Protocolo: piloto anti-sobreajuste de la CNN solo imagen

Fecha de fijación: 21/10/2026. Estado: **preparado, no entrenado**.

## Pregunta

¿Un paquete prefijado de simplificación y regularización reduce la brecha entre
fit y selección y mejora la AUC exterior frente a la misma CNN temporal simple?
El piloto sirve para descartar una mala dirección, no para demostrar una mejora
ni para elegir retrospectivamente otro modelo.

## Brazos

Ambos usan una CNN 2D propia desde cero, una sola vista 256×256, entrada ordenada
PRE/EARLY/LATE, fusión desde la primera convolución, cuatro bloques
`[24, 48, 96, 160]`, pooling global medio/máximo y cabeza de 64 unidades.

- **Control:** cortes mezclados, espejo y rotaciones de 90°, sin Dropout2d.
- **Anti-overfit:** lotes de seis pacientes, igual peso total por paciente,
  Dropout2d 0,10 y aumentos conjuntos: traslación ±3 %, escala 0,95–1,05,
  ganancia y gamma 0,90–1,10 y ruido común de desviación 0,01.

Los aumentos espaciales y de intensidad se aplican de forma idéntica a las tres
fases. No se normaliza cada fase por separado ni se generan mapas de diferencia.
Ambos brazos mantienen AdamW, LR 0,0008, weight decay 0,001, dropout de cabeza
0,35, máximo 15 épocas, mínimo 4 y paciencia 4.

## Evaluación fijada

- folds exteriores 0 y 3, elegidos antes de entrenar;
- semillas 42 y 2026;
- BCE normal y ponderada;
- 16 entrenamientos más pruebas cortas no elegibles;
- checkpoint por AUC de paciente en selección interna;
- media de cortes y después media de las dos semillas;
- test independiente y validación privada cerrados.

El paquete se considera prometedor dentro de cada BCE solo si cumple a la vez:

1. mejora de AUC media de los dos folds al menos +0,01;
2. ningún fold empeora más de 0,02;
3. la mediana de la brecha AUC fit−selección del checkpoint no aumenta.

Se calcula además un IC95 % pareado con 2.000 remuestreos estratificados por
fold y clase. Aunque el gate se cumpla, dos folds no autorizan promocionar el
modelo ni abrir test: justificaría completar los cinco folds con protocolo nuevo.

## Ejecución

Comprobación sin entrenar:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/piloto_antioverfit_gpu.ps1 -SoloComprobar
```

Piloto completo:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/piloto_antioverfit_gpu.ps1
```

La salida reanudable queda en
`resultados/04_entrenamiento/piloto_antioverfit_20261021/`.
