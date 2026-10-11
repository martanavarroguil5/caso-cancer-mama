# Contribuir

## Preparación

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m unittest discover -s tests -v
```

En Windows, sustituye `.venv/bin/python` por `.venv\Scripts\python.exe`.

## Flujo de trabajo

1. Trabaja directamente en `main`, la única rama del proyecto. No crees ramas adicionales salvo petición expresa del usuario.
2. Mantén separados los cambios de código, documentación y resultados cuando puedan revisarse de forma independiente.
3. Ejecuta las pruebas antes de confirmar cambios.
4. Usa mensajes de commit imperativos y concretos, por ejemplo: `Corrige la agregación por paciente`.
5. Documenta los experimentos descartados en `docs/EXPERIMENTOS_DESCARTADOS.md`, con resultados y commits de referencia. Retira su código de la versión actual: se conserva en el historial de `main`, sin ramas ni carpetas duplicadas de archivo. Versiona los modelos que deban conservarse en `modelos/versiones/`.
6. No confirmes datasets, entornos virtuales, checkpoints de trabajo ni ejecuciones pesadas. Los pesos seleccionados se publican como versiones completas en `modelos/versiones/`, siguiendo el [historial de modelos](modelos/README.md).

## Reproducibilidad

- Ajusta transformaciones, imputación y escalas exclusivamente con el train de cada fold.
- No abras el test reservado durante selección o ajuste.
- Conserva semillas, configuraciones, hashes y predicciones OOF de cada experimento aceptado.
- Si cambia el contrato de datos o inferencia, actualiza las pruebas y el README en el mismo cambio.
