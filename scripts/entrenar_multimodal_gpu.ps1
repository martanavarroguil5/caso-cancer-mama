param(
    [string]$Datos = "breastdcedl",
    [string]$Salida = "resultados/04_entrenamiento/multimodal_actual",
    [int]$Workers = 4,
    [int]$Lote = 64,
    [int]$Epocas = 46,
    [switch]$OmitirTests,
    [switch]$SoloHumo
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Python = Join-Path $Repo ".venv/Scripts/python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "No existe .venv. Crea el entorno e instala primero PyTorch con CUDA."
}
if (-not (Test-Path -LiteralPath (Join-Path $Datos "metadata/samples.csv"))) {
    throw "No se encuentra el dataset en '$Datos'. Debe contener metadata/samples.csv."
}

& $Python -c "import torch, sys; print('PyTorch', torch.__version__); print('CUDA', torch.version.cuda); print('GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO DISPONIBLE'); sys.exit(0 if torch.cuda.is_available() else 2)"
if ($LASTEXITCODE -ne 0) { throw "CUDA no está disponible en este entorno." }

if (-not $OmitirTests) {
    & $Python -B -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw "Los tests han fallado; no se inicia el entrenamiento." }
}

& $Python -B -m cancer_mama.entrenamiento entrenar `
    --prueba --semillas 42 --folds 0 `
    --epocas 1 --lote 8 --workers 0 --dispositivo cuda `
    --datos $Datos --salida $Salida
if ($LASTEXITCODE -ne 0) { throw "La prueba corta en GPU ha fallado." }

if ($SoloHumo) {
    Write-Host "Prueba corta completada. No se ha iniciado la matriz completa."
    exit 0
}

& $Python -B -m cancer_mama.entrenamiento entrenar `
    --semillas 42 2026 --folds 0 1 2 3 4 `
    --epocas $Epocas --lote $Lote --workers $Workers --dispositivo cuda `
    --datos $Datos --salida $Salida
if ($LASTEXITCODE -ne 0) { throw "La matriz de entrenamiento no ha terminado correctamente." }

& $Python -B -m cancer_mama.entrenamiento comparar --datos $Datos --salida $Salida
if ($LASTEXITCODE -ne 0) { throw "El comparador no ha terminado correctamente." }

Write-Host "Multimodal completo. Copia de vuelta la carpeta: $Salida"
