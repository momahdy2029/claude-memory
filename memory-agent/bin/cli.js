#!/usr/bin/env node

/**
 * Claude Memory Agent CLI
 *
 * This is the npm entry point that wraps the Python CLI.
 * Usage:
 *   claude-memory-agent install   - Run installation wizard
 *   claude-memory-agent start     - Start the agent
 *   claude-memory-agent stop      - Stop the agent
 *   claude-memory-agent status    - Check status
 *   claude-memory-agent dashboard - Open dashboard
 */

const { spawn, execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const AGENT_DIR = path.dirname(__dirname);
const args = process.argv.slice(2);
const command = args[0] || 'help';

// Check Python availability
function getPython() {
    const pythonCommands = ['python3', 'python', 'py'];

    for (const cmd of pythonCommands) {
        try {
            const result = execSync(`${cmd} --version`, {
                encoding: 'utf8',
                stdio: ['pipe', 'pipe', 'pipe']
            });
            if (result.includes('Python 3')) {
                return cmd;
            }
        } catch (e) {
            continue;
        }
    }
    return null;
}

// Check if Python dependencies are installed
function checkDependencies(python) {
    try {
        execSync(`${python} -c "import fastapi, uvicorn, dotenv"`, {
            cwd: AGENT_DIR,
            stdio: ['pipe', 'pipe', 'pipe']
        });
        return true;
    } catch (e) {
        return false;
    }
}

// Install Python dependencies
function installDependencies(python) {
    console.log('Installing Python dependencies...');
    try {
        execSync(`${python} -m pip install -r requirements.txt -q`, {
            cwd: AGENT_DIR,
            stdio: 'inherit'
        });
        return true;
    } catch (e) {
        console.error('Failed to install dependencies:', e.message);
        return false;
    }
}

// Run Python script
function runPython(script, scriptArgs = []) {
    const python = getPython();

    if (!python) {
        console.error('Error: Python 3 is required but not found.');
        console.error('Please install Python 3.9+ from https://python.org/');
        process.exit(1);
    }

    const proc = spawn(python, [script, ...scriptArgs], {
        cwd: AGENT_DIR,
        stdio: 'inherit',
        shell: process.platform === 'win32'
    });

    proc.on('close', (code) => {
        process.exit(code || 0);
    });

    proc.on('error', (err) => {
        console.error('Failed to start:', err.message);
        process.exit(1);
    });
}

// Doctor - diagnose issues (async)
async function runDoctor() {
    console.log('\n🩺 Claude Memory Agent - System Check\n');
    console.log('='.repeat(50));

    let issues = 0;
    let warnings = 0;

    // Check Python
    process.stdout.write('Python 3.9+........... ');
    const python = getPython();
    if (python) {
        console.log('✓ OK');
    } else {
        console.log('✗ NOT FOUND');
        console.log('   Install from: https://python.org/');
        issues++;
    }

    // Check Ollama
    process.stdout.write('Ollama................ ');
    const ollamaOk = await checkUrl('http://localhost:11434/api/tags');
    if (ollamaOk) {
        console.log('✓ Running');
    } else {
        console.log('✗ NOT RUNNING');
        console.log('   Install from: https://ollama.ai/download');
        console.log('   Then run: ollama serve');
        issues++;
    }

    // Check embedding model
    if (ollamaOk) {
        process.stdout.write('Embedding model....... ');
        const modelOk = await checkOllamaModel();
        if (modelOk) {
            console.log('✓ nomic-embed-text');
        } else {
            console.log('✗ NOT INSTALLED');
            console.log('   Run: ollama pull nomic-embed-text');
            issues++;
        }
    }

    // Check Memory Agent
    process.stdout.write('Memory Agent.......... ');
    const agentOk = await checkUrl('http://localhost:8102/health');
    if (agentOk) {
        console.log('✓ Running');
    } else {
        console.log('✗ NOT RUNNING');
        console.log('   Run: claude-memory-agent start');
        warnings++;
    }

    // Check .env file
    process.stdout.write('.env file............. ');
    if (fs.existsSync(path.join(AGENT_DIR, '.env'))) {
        console.log('✓ Exists');
    } else {
        console.log('✗ MISSING');
        console.log('   Run: claude-memory-agent install');
        issues++;
    }

    // Summary
    console.log('\n' + '='.repeat(50));
    if (issues === 0 && warnings === 0) {
        console.log('✅ All systems operational!\n');
    } else if (issues === 0) {
        console.log(`⚠️  ${warnings} warning(s) - agent may not be running\n`);
    } else {
        console.log(`❌ ${issues} issue(s) found - fix above problems\n`);
    }
}

// Helper: Check if URL responds
function checkUrl(url) {
    return new Promise((resolve) => {
        const http = require('http');
        const req = http.get(url, { timeout: 2000 }, (res) => {
            resolve(res.statusCode === 200);
        });
        req.on('error', () => resolve(false));
        req.on('timeout', () => { req.destroy(); resolve(false); });
    });
}

// Helper: Check if Ollama has embedding model
function checkOllamaModel() {
    return new Promise((resolve) => {
        const http = require('http');
        http.get('http://localhost:11434/api/tags', (res) => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => {
                try {
                    const json = JSON.parse(data);
                    const models = json.models || [];
                    const hasModel = models.some(m => m.name && m.name.includes('nomic-embed-text'));
                    resolve(hasModel);
                } catch (e) {
                    resolve(false);
                }
            });
        }).on('error', () => resolve(false));
    });
}

// Print help
function printHelp() {
    console.log(`
Claude Memory Agent v2.0.1
Persistent semantic memory for Claude Code sessions

USAGE:
  claude-memory-agent <command> [options]

COMMANDS:
  install     Run the installation wizard
  start       Start the memory agent in background
  stop        Stop the running agent
  restart     Restart the memory agent
  status      Check if agent is running
  dashboard   Open the web dashboard
  logs        Show recent log output
  doctor      Diagnose issues and check requirements
  uninstall   Remove Claude Code integration
  help        Show this help message

QUICK START:
  1. Install Ollama:        https://ollama.ai/download
  2. Pull embedding model:  ollama pull nomic-embed-text
  3. Start Ollama:          ollama serve
  4. Configure:             claude-memory-agent install
  5. Start agent:           claude-memory-agent start
  6. Open dashboard:        claude-memory-agent dashboard

REQUIREMENTS:
  - Python 3.9+          (https://python.org)
  - Ollama               (https://ollama.ai)
  - nomic-embed-text     (ollama pull nomic-embed-text)
  - Claude Code          (npm install -g @anthropic-ai/claude-code)

For more info: https://www.npmjs.com/package/claude-memory-agent
`);
}

// Main
function main() {
    switch (command) {
        case 'install':
        case 'setup':
            runPython('install.py', args.slice(1));
            break;

        case 'start':
            runPython('memory-agent', ['start', ...args.slice(1)]);
            break;

        case 'stop':
            runPython('memory-agent', ['stop', ...args.slice(1)]);
            break;

        case 'status':
            runPython('memory-agent', ['status', ...args.slice(1)]);
            break;

        case 'dashboard':
            runPython('memory-agent', ['dashboard', ...args.slice(1)]);
            break;

        case 'logs':
            runPython('memory-agent', ['logs', ...args.slice(1)]);
            break;

        case 'restart':
            // Stop then start
            console.log('Restarting Memory Agent...');
            try {
                execSync(`${getPython()} memory-agent stop`, { cwd: AGENT_DIR, stdio: 'inherit', shell: process.platform === 'win32' });
            } catch (e) { /* ignore stop errors */ }
            setTimeout(() => {
                runPython('memory-agent', ['start']);
            }, 1000);
            break;

        case 'doctor':
        case 'diagnose':
            runDoctor();
            break;

        case 'uninstall':
            runPython('install.py', ['--uninstall']);
            break;

        case 'run':
        case 'server':
            // Run directly (not in background)
            runPython('main.py', args.slice(1));
            break;

        case 'help':
        case '--help':
        case '-h':
            printHelp();
            break;

        case '--version':
        case '-v':
            console.log('claude-memory-agent v2.0.0');
            break;

        default:
            console.error(`Unknown command: ${command}`);
            console.error('Run "claude-memory-agent help" for usage.');
            process.exit(1);
    }
}

main();
