# Sets up an isolated venv for Parakeet-TDT benchmarking.
# Keeps nemo_toolkit + torch (~6GB) out of the main app's faster-whisper venv
# so we cannot accidentally break the production transcriber while experimenting.
#
# Usage (from repo root):
#   pwsh -File scripts/setup_parakeet_venv.ps1
#
# After it finishes:
#   .\.venv-parakeet\Scripts\Activate.ps1
#   python scripts/bench_parakeet.py --help

$ErrorActionPreference = "Stop"
$VenvPath = ".venv-parakeet"

if (Test-Path $VenvPath) {
    Write-Host "[setup] $VenvPath already exists. Delete it manually to recreate." -ForegroundColor Yellow
    exit 0
}

Write-Host "[setup] Creating venv at $VenvPath ..." -ForegroundColor Cyan
python -m venv $VenvPath
if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }

$Pip = ".\$VenvPath\Scripts\pip.exe"
$Py  = ".\$VenvPath\Scripts\python.exe"

Write-Host "[setup] Upgrading pip ..." -ForegroundColor Cyan
& $Py -m pip install --upgrade pip wheel setuptools

Write-Host "[setup] Installing torch with CUDA 12.4 (~2.5GB) ..." -ForegroundColor Cyan
& $Pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) { throw "torch install failed" }

Write-Host "[setup] Installing nemo_toolkit[asr] (~2GB) ..." -ForegroundColor Cyan
& $Pip install "nemo_toolkit[asr]>=2.0,<3.0"
if ($LASTEXITCODE -ne 0) { throw "nemo install failed" }

Write-Host "[setup] Installing bench utilities ..." -ForegroundColor Cyan
& $Pip install soundfile jiwer datasets faster-whisper

Write-Host "[setup] Verifying ..." -ForegroundColor Cyan
& $Py -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
& $Py -c "import nemo.collections.asr as a; print('nemo asr OK')"
& $Py -c "from faster_whisper import WhisperModel; print('faster-whisper OK')"

Write-Host "[setup] Done. Activate with: .\$VenvPath\Scripts\Activate.ps1" -ForegroundColor Green
