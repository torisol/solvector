@echo off
setlocal ENABLEDELAYEDEXPANSION

REM ============================================================
REM  install.bat  —  Windows one-shot setup for this repo
REM  - Creates .venv
REM  - Installs PyTorch (CUDA 12.4 wheelhouse)
REM  - Installs this package in editable mode (pip install -e .)
REM  - Prints a quick CUDA sanity line
REM ============================================================

REM Jump to repo root (directory of this .bat)
cd /d "%~dp0"

echo.
echo [1/5] Creating virtual environment at .venv (if missing)...
if not exist ".venv" (
  py -3 -m venv .venv
  if errorlevel 1 (
    echo [!] Failed to create venv with "py -3". Trying "python -m venv"...
    python -m venv .venv
  )
)

if not exist ".venv" (
  echo [X] Could not create virtual environment .venv
  exit /b 1
)

echo.
echo [2/5] Activating virtual environment...
call ".venv\Scripts\activate.bat"

echo.
echo [3/5] Upgrading base tooling...
python -m pip install --upgrade pip setuptools wheel

echo.
echo [4/5] Installing PyTorch (CUDA 12.4 wheelhouse)...
echo     -> If this step fails, see the CPU fallback message below.
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
if errorlevel 1 (
  echo.
  echo [!] PyTorch cu124 install reported an error.
  echo     You can try a CPU-only fallback with:
  echo         python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
  echo     Then re-run this script, or proceed to step 5 to install project deps.
)

echo.
echo [5/5] Installing project in editable mode...
if not exist pyproject.toml (
  echo [!] pyproject.toml is missing; creating a minimal one for editable install...
  > pyproject.toml echo [build-system]
  >> pyproject.toml echo requires = ["setuptools>=61.0"]
  >> pyproject.toml echo build-backend = "setuptools.build_meta"
  >> pyproject.toml echo.
  >> pyproject.toml echo [project]
  >> pyproject.toml echo name = "solvector"
  >> pyproject.toml echo version = "0.9.3"
  >> pyproject.toml echo description = "Y-vector training and evaluation utilities"
  >> pyproject.toml echo requires-python = ">=3.9"
  >> pyproject.toml echo dependencies = ["numpy>=1.23","tqdm>=4.66","regex>=2023.0.0"]
  >> pyproject.toml echo.
  >> pyproject.toml echo [tool.setuptools]
  >> pyproject.toml echo packages = ["solvector"]
)

python -m pip install -e .

echo.
echo [✓] Setup complete. CUDA sanity check:
python -c "import torch; print('torch.__version__ =', torch.__version__); print('cuda.is_available =', torch.cuda.is_available()); print('device =', 'cuda' if torch.cuda.is_available() else 'cpu')"

echo.
echo To start working, activate your environment with:
echo     call .venv\Scripts\activate.bat
echo.
echo Then try:
echo     python -m solvector.gpu_check
echo     .\run_train_32.bat data\train.jsonl models\yvec_tf_mean_32.pt
echo.
endlocal
