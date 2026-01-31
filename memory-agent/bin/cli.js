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

// Print help
function printHelp() {
    console.log(`
Claude Memory Agent v2.0.0
Persistent semantic memory for Claude Code sessions

USAGE:
  claude-memory-agent <command> [options]

COMMANDS:
  install     Run the installation wizard
  start       Start the memory agent in background
  stop        Stop the running agent
  status      Check if agent is running
  dashboard   Open the web dashboard
  logs        Show recent log output
  help        Show this help message

QUICK START:
  1. claude-memory-agent install   # Configure everything
  2. claude-memory-agent start     # Start the agent
  3. claude-memory-agent dashboard # Open web UI

REQUIREMENTS:
  - Python 3.9+
  - Ollama with nomic-embed-text model (for embeddings)
  - Claude Code (for full integration)

For more info: https://github.com/yourusername/claude-memory-agent
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
