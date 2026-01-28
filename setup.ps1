# Claude Memory System Setup Script
# Run this script as Administrator in PowerShell

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Claude Memory System - Setup Script" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "WARNING: Not running as Administrator. Some operations may fail." -ForegroundColor Yellow
    Write-Host "Please run PowerShell as Administrator for full setup." -ForegroundColor Yellow
    Write-Host ""
}

# Step 1: Check prerequisites
Write-Host "Step 1: Checking prerequisites..." -ForegroundColor Green

# Check Ollama
$ollamaExists = Get-Command ollama -ErrorAction SilentlyContinue
if ($ollamaExists) {
    Write-Host "  [OK] Ollama is installed" -ForegroundColor Green
    # Check for nomic-embed-text
    $models = ollama list 2>$null
    if ($models -match "nomic-embed-text") {
        Write-Host "  [OK] nomic-embed-text model is available" -ForegroundColor Green
    } else {
        Write-Host "  [..] Pulling nomic-embed-text model..." -ForegroundColor Yellow
        ollama pull nomic-embed-text
    }
} else {
    Write-Host "  [X] Ollama not found. Please install from https://ollama.ai" -ForegroundColor Red
}

# Check Python
$pythonExists = Get-Command python -ErrorAction SilentlyContinue
if ($pythonExists) {
    $pythonVersion = python --version 2>&1
    Write-Host "  [OK] Python: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "  [X] Python not found. Please install Python 3.10+" -ForegroundColor Red
}

# Check Node.js
$nodeExists = Get-Command node -ErrorAction SilentlyContinue
if ($nodeExists) {
    $nodeVersion = node --version 2>&1
    Write-Host "  [OK] Node.js: $nodeVersion" -ForegroundColor Green
} else {
    Write-Host "  [X] Node.js not found. Please install Node.js 18+" -ForegroundColor Red
}

Write-Host ""

# Step 2: Install PostgreSQL if needed
Write-Host "Step 2: Setting up PostgreSQL..." -ForegroundColor Green

$pgExists = Get-Command psql -ErrorAction SilentlyContinue
if (-not $pgExists) {
    # Check common install locations
    $pgPaths = @(
        "C:\Program Files\PostgreSQL\17\bin",
        "C:\Program Files\PostgreSQL\16\bin",
        "C:\Program Files\PostgreSQL\15\bin"
    )
    foreach ($path in $pgPaths) {
        if (Test-Path "$path\psql.exe") {
            $env:Path += ";$path"
            $pgExists = $true
            Write-Host "  [OK] Found PostgreSQL at $path" -ForegroundColor Green
            break
        }
    }
}

if (-not $pgExists) {
    Write-Host "  [..] PostgreSQL not found. Installing via winget..." -ForegroundColor Yellow
    Write-Host "  Note: You'll be prompted for a password during installation." -ForegroundColor Yellow
    Write-Host "  Remember this password - default is 'postgres'" -ForegroundColor Yellow
    Write-Host ""

    # Install PostgreSQL 17
    winget install PostgreSQL.PostgreSQL.17 --accept-package-agreements --accept-source-agreements

    # Add to PATH
    $env:Path += ";C:\Program Files\PostgreSQL\17\bin"

    Write-Host "  [OK] PostgreSQL installed. Please restart this script after installation completes." -ForegroundColor Green
    Write-Host ""
    Write-Host "After PostgreSQL is installed, run these commands manually:" -ForegroundColor Yellow
    Write-Host '  $env:Path += ";C:\Program Files\PostgreSQL\17\bin"' -ForegroundColor White
    Write-Host "  .\setup.ps1" -ForegroundColor White
    exit
} else {
    Write-Host "  [OK] PostgreSQL is available" -ForegroundColor Green
}

Write-Host ""

# Step 3: Create database and enable pgvector
Write-Host "Step 3: Setting up database..." -ForegroundColor Green

$password = Read-Host "Enter PostgreSQL password (default: postgres)" -AsSecureString
if ($password.Length -eq 0) {
    $password = ConvertTo-SecureString "postgres" -AsPlainText -Force
}
$BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($password)
$pgPassword = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)

$env:PGPASSWORD = $pgPassword

# Create database
Write-Host "  Creating database claude_memory..." -ForegroundColor Yellow
psql -U postgres -c "CREATE DATABASE claude_memory;" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] Database created" -ForegroundColor Green
} else {
    Write-Host "  [OK] Database may already exist" -ForegroundColor Yellow
}

# Enable pgvector extension
Write-Host "  Enabling pgvector extension..." -ForegroundColor Yellow
psql -U postgres -d claude_memory -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] pgvector extension enabled" -ForegroundColor Green
} else {
    Write-Host "  [X] Failed to enable pgvector. You may need to install it manually." -ForegroundColor Red
    Write-Host "      Download from: https://github.com/pgvector/pgvector/releases" -ForegroundColor Yellow
}

# Create tables
Write-Host "  Creating tables..." -ForegroundColor Yellow
$createTableSQL = @"
CREATE TABLE IF NOT EXISTS memories (
    id SERIAL PRIMARY KEY,
    type VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    embedding vector(768),
    metadata JSONB DEFAULT '{}',
    session_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC);
"@
psql -U postgres -d claude_memory -c $createTableSQL 2>$null
Write-Host "  [OK] Tables created" -ForegroundColor Green

$env:PGPASSWORD = ""

Write-Host ""

# Step 4: Setup Python environment
Write-Host "Step 4: Setting up Memory Agent (Python)..." -ForegroundColor Green

$memoryAgentPath = "C:\Users\moham\Desktop\Claude Memory\memory-agent"
Push-Location $memoryAgentPath

# Create virtual environment
if (-not (Test-Path "venv")) {
    Write-Host "  Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate and install dependencies
Write-Host "  Installing Python dependencies..." -ForegroundColor Yellow
& ".\venv\Scripts\pip.exe" install -r requirements.txt --quiet

Pop-Location
Write-Host "  [OK] Memory Agent setup complete" -ForegroundColor Green

Write-Host ""

# Step 5: Setup MCP Bridge (Node.js)
Write-Host "Step 5: Setting up MCP Bridge (Node.js)..." -ForegroundColor Green

$mcpBridgePath = "C:\Users\moham\mcp-servers\memory-bridge"
Push-Location $mcpBridgePath

Write-Host "  Installing Node.js dependencies..." -ForegroundColor Yellow
npm install --silent

Pop-Location
Write-Host "  [OK] MCP Bridge setup complete" -ForegroundColor Green

Write-Host ""

# Step 6: Display startup instructions
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To start the Memory System:" -ForegroundColor Green
Write-Host ""
Write-Host "1. Start Ollama (if not running):" -ForegroundColor White
Write-Host "   ollama serve" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Start Memory Agent (in a new terminal):" -ForegroundColor White
Write-Host "   cd 'C:\Users\moham\Desktop\Claude Memory\memory-agent'" -ForegroundColor Gray
Write-Host "   .\venv\Scripts\activate" -ForegroundColor Gray
Write-Host "   python main.py" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Claude Code will automatically start the MCP Bridge" -ForegroundColor White
Write-Host ""
Write-Host "4. Test with Claude Code using:" -ForegroundColor White
Write-Host "   memory_store, memory_search, memory_retrieve, memory_summarize" -ForegroundColor Gray
Write-Host ""
