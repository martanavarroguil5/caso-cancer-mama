# Paquete experimental del piloto anti-sobreajuste

Dieciséis pesos: control/antioverfit × normal/weighted × semillas 42/2026 × folds 0/3. No es una versión final de cinco folds, no cambia V002 y no se integra en la aplicación.

Desde esta carpeta, con las dependencias instaladas:

```bash
PYTHONPATH=codigo/src python -B inferencia.py verificar
PYTHONPATH=codigo/src python -B inferencia.py predecir --arm antioverfit --loss weighted --pre paciente_z000_PRE.png --early paciente_z000_EARLY.png --late paciente_z000_LATE.png
```

La inferencia promedia cuatro modelos y cortes; usa probabilidad cruda y umbral diagnóstico 0,5, sin calibración. Las OOF del informe usan únicamente las dos semillas del fold que excluye a cada paciente: no equivalen a la inferencia conjunta del paquete sobre train.
