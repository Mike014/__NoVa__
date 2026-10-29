# Setup di NoVa: ambiente Python, Ollama e modelli locali.
# Doppio clic su setup.bat, oppure: powershell -ExecutionPolicy Bypass -File setup.ps1
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Write-Step($message) {
    Write-Host ""
    Write-Host "==> $message" -ForegroundColor Cyan
}

function Refresh-Path {
    $machine = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [System.Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machine;$user"
}

function Find-Ollama {
    Refresh-Path
    $cmd = Get-Command ollama -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe",
        "$env:ProgramFiles\Ollama\ollama.exe"
    )
    foreach ($path in $candidates) {
        if (Test-Path $path) { return $path }
    }
    return $null
}

Write-Host "NoVa — setup locale (Ollama)" -ForegroundColor Green

Write-Step "Controllo Python"
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    throw "Python non trovato. Installalo da https://www.python.org/downloads/ e rilancia questo script."
}
& python --version

Write-Step "Creo l'ambiente virtuale .venv"
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Creazione del virtualenv fallita." }
}
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

Write-Step "Installo le dipendenze Python"
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Aggiornamento di pip fallito." }
& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Installazione delle dipendenze fallita." }

Write-Step "Scarico il modello spaCy en_core_web_sm"
& $venvPython -m spacy download en_core_web_sm
if ($LASTEXITCODE -ne 0) { throw "Download del modello spaCy fallito." }

Write-Step "Controllo Ollama"
$ollama = Find-Ollama
if (-not $ollama) {
    Write-Host "Ollama non è installato. Scarico l'installer ufficiale..."
    $installer = Join-Path $env:TEMP "OllamaSetup.exe"
    Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" -OutFile $installer -UseBasicParsing
    Write-Host "Installo Ollama (potrebbe comparire una richiesta di Windows)..."
    $install = Start-Process -FilePath $installer -ArgumentList "/VERYSILENT", "/NORESTART" -PassThru
    $deadline = (Get-Date).AddMinutes(15)
    while (-not $install.HasExited -and (Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 3
        # L'installer avvia l'app e a volte non termina: appena l'eseguibile c'è, si va avanti.
        if (Find-Ollama) { break }
    }
    if (-not $install.HasExited -and -not (Find-Ollama)) {
        throw "Installazione di Ollama non conclusa. Avvia OllamaSetup.exe a mano da $installer"
    }
    $ollama = Find-Ollama
}
if (-not $ollama) {
    throw "Ollama risulta installato ma ollama.exe non è stato trovato. Riavvia il terminale e rilancia lo script."
}
Write-Host "Ollama: $ollama"
& $ollama --version

Write-Step "Scarico i modelli (circa 12 GB, la prima volta ci vuole un po')"
$models = @("gemma2:2b", "llama3.2:3b", "mistral:7b", "qwen2.5-coder:7b")
foreach ($model in $models) {
    Write-Host "ollama pull $model"
    & $ollama pull $model
    if ($LASTEXITCODE -ne 0) { throw "Download del modello $model fallito." }
}

Write-Step "Controllo il file .env"
if (-not (Test-Path ".env")) {
    @"
DISCORD_API_TOKEN=
OLLAMA_HOST=http://localhost:11434
OLLAMA_AGENT_MODEL=qwen2.5-coder:7b
"@ | Set-Content -Path ".env" -Encoding utf8
    Write-Host "Creato .env vuoto: apri il file e incolla il token del bot Discord." -ForegroundColor Yellow
} else {
    $tokenSet = Select-String -Path ".env" -Pattern '^\s*DISCORD_API_TOKEN\s*=\s*\S+' -Quiet
    if ($tokenSet) {
        Write-Host "DISCORD_API_TOKEN è presente. La chiave Hugging Face non serve più: puoi togliere HUGGINGFACE_API_KEY."
    } else {
        Write-Host "Nel .env manca DISCORD_API_TOKEN. Aggiungilo prima di avviare il bot." -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Setup completato." -ForegroundColor Green
Write-Host "Avvia il bot con un doppio clic su avvia.bat, oppure:"
Write-Host "  .\.venv\Scripts\python src\bot.py"
