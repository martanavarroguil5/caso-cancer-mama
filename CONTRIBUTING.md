# Contribuir

## Preparación

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m unittest discover -s tests -v
```

En Windows, sustituye `.venv/bin/python` por `.venv\Scripts\python.exe`.

## Flujo de trabajo

1. Crea una rama corta desde `main` para cada cambio.
2. Mantén separados los cambios de código, documentación y resultados cuando puedan revisarse de forma independiente.
3. Ejecuta las pruebas antes de confirmar cambios.
4. Usa mensajes de commit imperativos y concretos, por ejemplo: `Corrige la agregación por paciente`.
5. No confirmes datasets, entornos virtuales, checkpoints de trabajo ni ejecuciones pesadas. Los pesos seleccionados se publican como versiones completas en `modelos/versiones/`, siguiendo el [historial de modelos](modelos/README.md).

## Reproducibilidad

- Ajusta transformaciones, imputación y escalas exclusivamente con el train de cada fold.
- No abras el test reservado durante selección o ajuste.
- Conserva semillas, configuraciones, hashes y predicciones OOF de cada experimento aceptado.
- Si cambia el contrato de datos o inferencia, actualiza las pruebas y el README en el mismo cambio.
