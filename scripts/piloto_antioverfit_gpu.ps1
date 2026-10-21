param(
    [string]$Datos = "breastdcedl",
    [string]$Salida = "resultados/04_entrenamiento/piloto_antioverfit_20261021",
    [int]$Workers = 4,
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

& $Python -c "import torch,sys; print('PyTorch',torch.__version__); print('GPU',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO DISPONIBLE'); sys.exit(0 if torch.cuda.is_available() else 2)"
if ($LASTEXITCODE -ne 0) { throw "CUDA no está disponible." }

if (-not $OmitirTests) {
    & $Python -B -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw "Los tests han fallado; no se entrena." }
}

& $Python -B -m cancer_mama.entrenamiento_antioverfit verificar
if ($LASTEXITCODE -ne 0) { throw "El protocolo del piloto no es válido." }

if ($SoloComprobar) {
    Write-Host "Entorno y piloto anti-sobreajuste verificados. No se ha entrenado."
    exit 0
}

& $Python -B -m cancer_mama.entrenamiento_antioverfit entrenar `
    --prueba --brazos control antioverfit --perdidas normal weighted `
    --semillas 42 --folds 0 --workers 0 --dispositivo cuda `
    --datos $Datos --salida $Salida
if ($LASTEXITCODE -ne 0) { throw "La prueba corta en GPU ha fallado." }

& $Python -B -m cancer_mama.entrenamiento_antioverfit entrenar `
    --brazos control antioverfit --perdidas normal weighted `
    --semillas 42 2026 --folds 0 3 --workers $Workers --dispositivo cuda `
    --datos $Datos --salida $Salida
if ($LASTEXITCODE -ne 0) { throw "Los 16 entrenamientos no han terminado." }

& $Python -B -m cancer_mama.entrenamiento_antioverfit comparar `
    --salida $Salida
if ($LASTEXITCODE -ne 0) { throw "La comparación del piloto ha fallado." }

Write-Host "Piloto terminado. Revisa: $Salida/INFORME_PILOTO.md"
