# Caso cancer de mama

El codigo de este proyecto se comparte por GitHub. El dataset BreastDCEDL se
descarga directamente desde el bucket publico del curso en cada ordenador y
queda fuera de Git. Son 38.109 imagenes (aproximadamente 1,37 GB).

## Preparar un ordenador nuevo

Instala Git y Python, clona el repositorio y abre la carpeta:

```powershell
git clone https://github.com/martanavarroguil5/caso-cancer-mama.git
cd caso-cancer-mama
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe descargar_datos.py
```

En macOS o Linux, sustituye las dos ultimas llamadas a Python por
`.venv/bin/python`. El script guarda los datos en `breastdcedl/`; si se corta,
vuelve a ejecutarlo para descargar solo lo que falte. Para probar con menos
datos, ejecuta `descargar_datos.py --pacientes 5`. La guia del caso se descarga
en `breastdcedl/GUIA.md`.

## Trabajar en dos ordenadores

Antes de empezar en cualquiera de ellos, ejecuta `git pull`. Al terminar,
revisa `git status`, incorpora solo el codigo, notebooks y documentacion que
quieras compartir, haz `git commit` y `git push`. En el otro ordenador, vuelve
a ejecutar `git pull` antes de continuar. Conviene terminar y subir los cambios
en un equipo antes de pasar al otro.

`breastdcedl/`, los entornos virtuales y los pesos de modelos entrenados se
ignoran en Git. Conserva los modelos grandes en un almacenamiento separado y
anota en el proyecto como obtener el modelo final si lo necesitas para la
entrega.
