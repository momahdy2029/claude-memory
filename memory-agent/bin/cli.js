#!/usr/bin/env node

/**
 * Claude Memory Agent CLI
 *
 * Cross-platform entry point that manages the Python memory agent server.
 * Usage:
 *   claude-memory-agent install   - Run installation wizard
 *   claude-memory-agent start     - Start the agent in background
 *   claude-memory-agent stop      - Stop the running agent
 *   claude-memory-agent status    - Check if agent is running
 *   claude-memory-agent dashboard - Open the web dashboard
 */

const { spawn, execSync } = require('child_process');
const path = require('path');
const fs = require('fs');
const http = require('http');

const AGENT_DIR = path.dirname(__dirname);
const PID_FILE = path.join(AGENT_DIR, 'memory-agent.pid');
const LOG_FILE = path.join(AGENT_DIR, 'memory-agent.log');
const ENV_FILE = path.join(AGENT_DIR, '.env');
const args = process.argv.slice(2);
const command = args[0] || 'help';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Read PORT from .env file, default 8102 */
function getPort() {
    try {
        const env = fs.readFileSync(ENV_FILE, 'utf8');
        const match = env.match(/^PORT=(\d+)/m);
        if (match) return parseInt(match[1], 10);
    } catch (_) {}
    return parseInt(process.env.PORT || '8102', 10);
}

/** Read HOST from .env file, default 127.0.0.1 */
function getHost() {
    try {
        const env = fs.readFileSync(ENV_FILE, 'utf8');
        const match = env.match(/^HOST=(.+)/m);
        if (match) {
            const h = match[1].trim();
            return (h === '0.0.0.0') ? 'localhost' : h;
        }
    } catch (_) {}
    return 'localhost';
}

/** Detect Python 3 command */
function getPython() {
    const commands = ['python3', 'python', 'py'];
    for (const cmd of commands) {
        try {
            const out = execSync(`${cmd} --version`, {
                encoding: 'utf8',
                stdio: ['pipe', 'pipe', 'pipe'],
                timeout: 5000,
            });
            if (out && out.includes('Python 3')) return cmd;
        } catch (_) {}
    }
    return null;
}

/** Require Python or exit */
function requirePython() {
    const py = getPython();
    if (!py) {
        console.error('Error: Python 3 is required but not found.');
        console.error('Install from: https://python.org/');
        process.exit(1);
    }
    return py;
}

/** Run a Python script with inherited stdio */
function runPython(script, scriptArgs = []) {
    const python = requirePython();
    const proc = spawn(python, [script, ...scriptArgs], {
        cwd: AGENT_DIR,
        stdio: 'inherit',
        shell: process.platform === 'win32',
    });
    proc.on('close', (code) => process.exit(code || 0));
    proc.on('error', (err) => {
        console.error('Failed to start:', err.message);
        process.exit(1);
    });
}

/** HTTP GET with timeout, resolves { ok, status, body } */
function httpGet(url, timeout = 3000) {
    return new Promise((resolve) => {
        const req = http.get(url, { timeout }, (res) => {
            let body = '';
            res.on('data', (c) => (body += c));
            res.on('end', () => resolve({ ok: res.statusCode === 200, status: res.statusCode, body }));
        });
        req.on('error', () => resolve({ ok: false, status: 0, body: '' }));
        req.on('timeout', () => { req.destroy(); resolve({ ok: false, status: 0, body: '' }); });
    });
}

/** Check if the agent is responding on its health endpoint */
async function isAgentRunning() {
    const port = getPort();
    const host = getHost();
    const { ok } = await httpGet(`http://${host}:${port}/health`);
    return ok;
}

/** Read PID from file */
function readPid() {
    try {
        if (fs.existsSync(PID_FILE)) {
            const pid = parseInt(fs.readFileSync(PID_FILE, 'utf8').trim(), 10);
            if (!isNaN(pid)) return pid;
        }
    } catch (_) {}
    return null;
}

/** Check if a process with given PID is alive */
function isProcessAlive(pid) {
    try {
        process.kill(pid, 0); // signal 0 = check existence
        return true;
    } catch (_) {
        return false;
    }
}

/** Kill a process by PID (cross-platform) */
function killProcess(pid) {
    try {
        if (process.platform === 'win32') {
            execSync(`taskkill /PID ${pid} /T /F`, { stdio: 'pipe' });
        } else {
            process.kill(pid, 'SIGTERM');
        }
        return true;
    } catch (_) {
        return false;
    }
}

/** Open a URL in the default browser */
function openBrowser(url) {
    const cmd = process.platform === 'darwin' ? 'open'
        : process.platform === 'win32' ? 'start'
        : 'xdg-open';
    try {
        execSync(`${cmd} ${url}`, { stdio: 'ignore', shell: true });
    } catch (_) {
        console.log(`Open in browser: ${url}`);
    }
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

async function cmdStart() {
    // Already running?
    if (await isAgentRunning()) {
        console.log('Memory agent is already running.');
        return;
    }

    const python = requirePython();
    const mainPy = path.join(AGENT_DIR, 'main.py');

    if (!fs.existsSync(mainPy)) {
        console.error('Error: main.py not found in ' + AGENT_DIR);
        process.exit(1);
    }

    console.log('Starting Memory Agent...');

    const logFd = fs.openSync(LOG_FILE, 'w');
    const spawnOpts = {
        cwd: AGENT_DIR,
        stdio: ['ignore', logFd, logFd],
        detached: true,
        shell: process.platform === 'win32',
    };

    // Windows needs special flags
    if (process.platform === 'win32') {
        spawnOpts.windowsHide = true;
    }

    const child = spawn(python, [mainPy], spawnOpts);
    child.unref();

    // Write PID
    fs.writeFileSync(PID_FILE, String(child.pid), 'utf8');

    // Wait up to 8 seconds for health check
    for (let i = 0; i < 16; i++) {
        await new Promise((r) => setTimeout(r, 500));
        if (await isAgentRunning()) {
            const port = getPort();
            console.log(`Memory agent started (PID: ${child.pid}, port: ${port})`);
            fs.closeSync(logFd);
            return;
        }
    }

    fs.closeSync(logFd);
    console.log(`Agent started (PID: ${child.pid}) but health check not responding yet.`);
    console.log(`Check logs: claude-memory-agent logs`);
}

async function cmdStop() {
    const pid = readPid();

    if (!pid) {
        // Try health check anyway
        if (await isAgentRunning()) {
            console.log('Agent is running but no PID file found. Cannot stop automatically.');
            console.log('Find the process manually and kill it.');
        } else {
            console.log('Memory agent is not running.');
        }
        return;
    }

    if (!isProcessAlive(pid)) {
        console.log('Memory agent is not running (stale PID file).');
        try { fs.unlinkSync(PID_FILE); } catch (_) {}
        return;
    }

    console.log(`Stopping Memory Agent (PID: ${pid})...`);
    if (killProcess(pid)) {
        // Wait for it to die
        for (let i = 0; i < 10; i++) {
            await new Promise((r) => setTimeout(r, 300));
            if (!isProcessAlive(pid)) {
                console.log('Memory agent stopped.');
                try { fs.unlinkSync(PID_FILE); } catch (_) {}
                return;
            }
        }
        console.log('Sent kill signal but process may still be shutting down.');
    } else {
        console.log('Failed to stop agent. Try killing PID ' + pid + ' manually.');
    }
}

async function cmdStatus() {
    const port = getPort();
    const host = getHost();
    const pid = readPid();
    const running = await isAgentRunning();

    if (running) {
        console.log(`Memory agent is RUNNING`);
        console.log(`  URL:  http://${host}:${port}`);
        if (pid) console.log(`  PID:  ${pid}`);
    } else {
        console.log('Memory agent is NOT running.');
        if (pid && isProcessAlive(pid)) {
            console.log(`  PID ${pid} exists but health check failed.`);
            console.log('  Check logs: claude-memory-agent logs');
        }
    }
}

async function cmdDashboard() {
    const port = getPort();
    const host = getHost();
    const url = `http://${host}:${port}/dashboard`;

    if (!(await isAgentRunning())) {
        console.log('Memory agent is not running. Start it first:');
        console.log('  claude-memory-agent start');
        return;
    }

    console.log(`Opening dashboard: ${url}`);
    openBrowser(url);
}

function cmdLogs() {
    if (!fs.existsSync(LOG_FILE)) {
        console.log('No log file found. Start the agent first.');
        return;
    }
    const content = fs.readFileSync(LOG_FILE, 'utf8');
    const lines = content.split('\n');
    // Show last 50 lines
    const tail = lines.slice(-50).join('\n');
    console.log(tail || '(empty log)');
}

async function cmdRestart() {
    console.log('Restarting Memory Agent...');
    await cmdStop();
    await new Promise((r) => setTimeout(r, 1000));
    await cmdStart();
}

async function runDoctor() {
    console.log('\nClaude Memory Agent - System Check\n');
    console.log('='.repeat(50));

    let issues = 0;
    let warnings = 0;

    // Python
    process.stdout.write('Python 3.9+........... ');
    const python = getPython();
    if (python) {
        console.log('OK');
    } else {
        console.log('NOT FOUND');
        console.log('   Install from: https://python.org/');
        issues++;
    }

    // Ollama
    process.stdout.write('Ollama................ ');
    const { ok: ollamaOk } = await httpGet('http://localhost:11434/api/tags');
    if (ollamaOk) {
        console.log('Running');
    } else {
        console.log('NOT RUNNING');
        console.log('   Install from: https://ollama.ai/download');
        console.log('   Then run: ollama serve');
        issues++;
    }

    // Embedding model
    if (ollamaOk) {
        process.stdout.write('Embedding model....... ');
        const { ok: modelOk, body } = await httpGet('http://localhost:11434/api/tags');
        let hasModel = false;
        try {
            const models = JSON.parse(body).models || [];
            hasModel = models.some((m) => m.name && m.name.includes('nomic-embed-text'));
        } catch (_) {}
        if (hasModel) {
            console.log('nomic-embed-text');
        } else {
            console.log('NOT INSTALLED');
            console.log('   Run: ollama pull nomic-embed-text');
            issues++;
        }
    }

    // Memory Agent
    process.stdout.write('Memory Agent.......... ');
    if (await isAgentRunning()) {
        console.log('Running');
    } else {
        console.log('NOT RUNNING');
        console.log('   Run: claude-memory-agent start');
        warnings++;
    }

    // .env
    process.stdout.write('.env file............. ');
    if (fs.existsSync(ENV_FILE)) {
        console.log('Exists');
    } else {
        console.log('MISSING');
        console.log('   Run: claude-memory-agent install');
        issues++;
    }

    // Summary
    console.log('\n' + '='.repeat(50));
    if (issues === 0 && warnings === 0) {
        console.log('All systems operational!\n');
    } else if (issues === 0) {
        console.log(`${warnings} warning(s) - agent may not be running\n`);
    } else {
        console.log(`${issues} issue(s) found - fix above problems\n`);
    }
}

function printHelp() {
    console.log(`
Claude Memory Agent v2.2.4
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

// ---------------------------------------------------------------------------
// Main dispatch
// ---------------------------------------------------------------------------

async function main() {
    switch (command) {
        case 'install':
        case 'setup':
            try {
                const { main: runOnboarding } = require('./onboarding');
                runOnboarding().catch((err) => {
                    console.error('Setup error:', err.message);
                    process.exit(1);
                });
            } catch (_) {
                runPython('install.py', args.slice(1));
            }
            break;

        case 'start':
            await cmdStart();
            break;

        case 'stop':
            await cmdStop();
            break;

        case 'restart':
            await cmdRestart();
            break;

        case 'status':
            await cmdStatus();
            break;

        case 'dashboard':
            await cmdDashboard();
            break;

        case 'logs':
            cmdLogs();
            break;

        case 'doctor':
        case 'diagnose':
            await runDoctor();
            break;

        case 'uninstall':
            runPython('install.py', ['--uninstall']);
            break;

        case 'run':
        case 'server':
            // Run main.py directly (foreground, not detached)
            runPython('main.py', args.slice(1));
            break;

        case 'help':
        case '--help':
        case '-h':
            printHelp();
            break;

        case '--version':
        case '-v':
            console.log('claude-memory-agent v2.2.4');
            break;

        default:
            console.error(`Unknown command: ${command}`);
            console.error('Run "claude-memory-agent help" for usage.');
            process.exit(1);
    }
}

main();
