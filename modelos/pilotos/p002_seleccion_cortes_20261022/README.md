# Piloto experimental de selección de cortes

Doce pesos: k=10/3/1 × semillas 42/2026 × folds 0/3. BCE ponderada por paciente. Ningún brazo alcanza AUC media 0,60. No es una versión productiva y no sustituye V002.

El parámetro k identifica los cortes usados en fit. En inferencia se usan todos los cortes aportados. Cuatro modelos por k, media de probabilidades crudas entre modelos y cortes, umbral fijo 0,5 y sin calibración. Las OOF del informe usan solo dos semillas del fold que excluye cada paciente, por lo que no equivalen a aplicar los cuatro modelos del paquete sobre train. Positivo significa pCR.

```bash
PYTHONPATH=codigo/src python -B inferencia.py verificar
PYTHONPATH=codigo/src python -B inferencia.py predecir --k 3 --pre paciente_z000_PRE.png --early paciente_z000_EARLY.png --late paciente_z000_LATE.png
```

Solo pesos de inferencia, fuentes congeladas, configuraciones, métricas y hashes. No se incluyen datasets ni estado de optimizador/RNG.
