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

## Analisis reproducible

Desde la raiz del proyecto:

```powershell
.\.venv\Scripts\python.exe 01_auditoria_datos.py
.\.venv\Scripts\python.exe 02_eda_profesional.py
.\.venv\Scripts\python.exe 03_preparar_datos.py --fold 0
```

La auditoria valida pacientes, clases, particiones, PNG, canales y rangos. El
EDA trabaja a nivel de paciente, integra variables clinicas y caracteristicas de
imagen, estudia cohortes y cambio de distribucion, genera PCA y documenta que
variables deben excluirse o vigilarse. Los resultados se guardan en
`resultados/01_auditoria/` y `resultados/02_eda/` sin modificar los CSV fuente.
El tercer script crea y verifica las particiones y los DataLoaders del baseline
de imagen, calcula estadisticas solo con train y mantiene test cerrado.

Para regenerar tambien las caracteristicas de imagen, en lugar de reutilizar la
cache derivada:

```powershell
.\.venv\Scripts\python.exe 02_eda_profesional.py --recalcular-imagenes
```

Las comprobaciones del pipeline se ejecutan con:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

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
