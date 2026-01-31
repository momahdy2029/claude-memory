#!/usr/bin/env node

/**
 * Post-installation script for Claude Memory Agent
 *
 * This runs automatically after npm install to:
 * 1. Check Python availability
 * 2. Install Python dependencies
 * 3. Run the configuration wizard (if interactive)
 */

const { execSync, spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const readline = require('readline');

const AGENT_DIR = path.dirname(__dirname);

// Colors for terminal output
const colors = {
    reset: '\x1b[0m',
    green: '\x1b[32m',
    yellow: '\x1b[33m',
    red: '\x1b[31m',
    cyan: '\x1b[36m',
    bold: '\x1b[1m'
};

function log(msg) {
    console.log(msg);
}

function success(msg) {
    console.log(`${colors.green}✓${colors.reset} ${msg}`);
}

function warn(msg) {
    console.log(`${colors.yellow}!${colors.reset} ${msg}`);
}

function error(msg) {
    console.log(`${colors.red}✗${colors.reset} ${msg}`);
}

function header(msg) {
    console.log(`\n${colors.bold}${colors.cyan}${msg}${colors.reset}\n`);
}

// Check Python availability
function getPython() {
    const pythonCommands = ['python3', 'python', 'py'];

    for (const cmd of pythonCommands) {
        try {
            const result = execSync(`${cmd} --version`, {
                encoding: 'utf8',
                stdio: ['pipe', 'pipe', 'pipe']
            });
            const match = result.match(/Python (\d+)\.(\d+)/);
            if (match) {
                const major = parseInt(match[1]);
                const minor = parseInt(match[2]);
                if (major >= 3 && minor >= 9) {
                    return { cmd, version: result.trim() };
                }
            }
        } catch (e) {
            continue;
        }
    }
    return null;
}

// Check if Ollama is running
function checkOllama() {
    try {
        const http = require('http');
        return new Promise((resolve) => {
            const req = http.get('http://localhost:11434/api/tags', { timeout: 2000 }, (res) => {
                resolve(res.statusCode === 200);
            });
            req.on('error', () => resolve(false));
            req.on('timeout', () => {
                req.destroy();
                resolve(false);
            });
        });
    } catch (e) {
        return Promise.resolve(false);
    }
}

// Install Python dependencies
function installPythonDeps(python) {
    const requirementsPath = path.join(AGENT_DIR, 'requirements.txt');

    if (!fs.existsSync(requirementsPath)) {
        warn('requirements.txt not found, skipping Python dependencies');
        return true;
    }

    log('Installing Python dependencies...');
    try {
        execSync(`${python} -m pip install -r requirements.txt -q --disable-pip-version-check`, {
            cwd: AGENT_DIR,
            stdio: ['pipe', 'pipe', 'pipe']
        });
        success('Python dependencies installed');
        return true;
    } catch (e) {
        error('Failed to install Python dependencies');
        console.log('  Run manually: pip install -r requirements.txt');
        return false;
    }
}

// Create .env file if it doesn't exist
function createEnvFile() {
    const envPath = path.join(AGENT_DIR, '.env');
    const examplePath = path.join(AGENT_DIR, '.env.example');

    if (fs.existsSync(envPath)) {
        success('.env file already exists');
        return true;
    }

    // Create basic .env
    const envContent = `# Claude Memory Agent Configuration
# Generated during npm install

# Server
PORT=8102
HOST=0.0.0.0
MEMORY_AGENT_URL=http://localhost:8102

# Ollama
OLLAMA_HOST=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text

# Database (relative to agent directory)
DATABASE_PATH=${path.join(AGENT_DIR, 'memories.db').replace(/\\/g, '/')}

# Logging
LOG_LEVEL=INFO
`;

    try {
        fs.writeFileSync(envPath, envContent);
        success('Created .env configuration file');
        return true;
    } catch (e) {
        warn('Could not create .env file');
        return false;
    }
}

// Main installation
async function main() {
    header('Claude Memory Agent - Post-Installation Setup');

    // Check Python
    log('Checking Python...');
    const python = getPython();
    if (!python) {
        error('Python 3.9+ is required but not found');
        console.log('\n  Please install Python from https://python.org/');
        console.log('  Then run: claude-memory-agent install\n');
        process.exit(1);
    }
    success(`Found ${python.version}`);

    // Install Python dependencies
    installPythonDeps(python.cmd);

    // Create .env file
    createEnvFile();

    // Check Ollama (non-blocking)
    log('Checking Ollama...');
    const ollamaRunning = await checkOllama();
    if (ollamaRunning) {
        success('Ollama is running');
    } else {
        warn('Ollama not detected - embeddings require Ollama');
        console.log('  Install from: https://ollama.ai/');
        console.log('  Then run: ollama pull nomic-embed-text');
    }

    // Print next steps
    header('Installation Complete!');
    console.log('Next steps:');
    console.log('');
    console.log('  1. Run the setup wizard:');
    console.log('     claude-memory-agent install');
    console.log('');
    console.log('  2. Start the agent:');
    console.log('     claude-memory-agent start');
    console.log('');
    console.log('  3. Open the dashboard:');
    console.log('     claude-memory-agent dashboard');
    console.log('');

    if (!ollamaRunning) {
        console.log('  For embeddings, install and start Ollama:');
        console.log('     ollama pull nomic-embed-text');
        console.log('     ollama serve');
        console.log('');
    }
}

// Run if not in CI/CD environment
if (!process.env.CI && !process.env.npm_config_ignore_scripts) {
    main().catch(console.error);
} else {
    console.log('Skipping post-install in CI environment');
}
