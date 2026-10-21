param([int]$Workers = 4)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
$Python = Join-Path $ProjectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $Python)) { $Python = 'python' }
& $Python -c 'import torch; assert torch.cuda.is_available(), "CUDA no disponible"; print(torch.cuda.get_device_name(0))'
if ($LASTEXITCODE -ne 0) { throw 'CUDA no está disponible' }
& $Python -m cancer_mama.seleccion_cortes preparar
if ($LASTEXITCODE -ne 0) { throw 'Falló la preparación' }
& $Python -m cancer_mama.seleccion_cortes entrenar --prueba --dispositivo cuda --workers 0
if ($LASTEXITCODE -ne 0) { throw 'Falló la prueba técnica de los tres brazos' }
& $Python -m cancer_mama.seleccion_cortes entrenar --dispositivo cuda --workers $Workers
if ($LASTEXITCODE -ne 0) { throw 'Entrenamiento interrumpido; ejecutar de nuevo para reanudar' }
& $Python -m cancer_mama.seleccion_cortes comparar
if ($LASTEXITCODE -ne 0) { throw 'Falló la comparación' }
