param(
    [string]$Datos = "breastdcedl",
    [string]$Salida = "resultados/04_entrenamiento/cnn_imagen_cinetica_multiescala_20261021",
    [int]$Workers = 4,
    [int]$Lote = 32,
    [int]$Epocas = 40,
    [switch]$OmitirTests,
    [switch]$SoloComprobar
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Python = Join-Path $Repo ".venv/Scripts/python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "No existe .venv. Crea el entorno e instala PyTorch con CUDA."
}
if (-not (Test-Path -LiteralPath (Join-Path $Datos "metadata/samples.csv"))) {
    throw "No se encuentra metadata/samples.csv dentro de '$Datos'."
}

& $Python -c "import torch,sys; print('PyTorch',torch.__version__); print('CUDA',torch.version.cuda); print('GPU',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO DISPONIBLE'); sys.exit(0 if torch.cuda.is_available() else 2)"
if ($LASTEXITCODE -ne 0) { throw "CUDA no está disponible." }

if (-not $OmitirTests) {
    & $Python -B -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw "Los tests han fallado; no se entrena." }
}

& $Python -B -m cancer_mama.entrenamiento_cinetico verificar
if ($LASTEXITCODE -ne 0) { throw "La configuración cinética multiescala no es válida." }

if ($SoloComprobar) {
    Write-Host "Entorno y CNN cinética multiescala verificados. No se ha entrenado ningún modelo."
    exit 0
}

& $Python -B -m cancer_mama.entrenamiento_cinetico entrenar `
    --prueba --perdidas normal weighted --semillas 42 --folds 0 `
    --epocas 1 --lote 4 --workers 0 --dispositivo cuda `
    --datos $Datos --salida $Salida
if ($LASTEXITCODE -ne 0) { throw "La prueba corta en GPU ha fallado." }

& $Python -B -m cancer_mama.entrenamiento_cinetico entrenar `
    --perdidas normal weighted --semillas 42 2026 --folds 0 1 2 3 4 `
    --epocas $Epocas --lote $Lote --workers $Workers --dispositivo cuda `
    --datos $Datos --salida $Salida
if ($LASTEXITCODE -ne 0) { throw "La matriz de entrenamiento no ha terminado." }

& $Python -B -m cancer_mama.entrenamiento_cinetico comparar `
    --datos $Datos --salida $Salida
if ($LASTEXITCODE -ne 0) { throw "La comparación final ha fallado." }

Write-Host "Experimento cinético multiescala completo. Copia de vuelta: $Salida"
