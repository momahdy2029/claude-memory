'use strict';

const os = require('os');
const http = require('http');
const { execSync } = require('child_process');
const chalk = require('chalk');

/**
 * Try running a command and return its stdout, or null on failure.
 * @param {string} cmd - Command to execute
 * @returns {string|null}
 */
function tryExec(cmd) {
    try {
        return execSync(cmd, {
            encoding: 'utf8',
            stdio: ['pipe', 'pipe', 'pipe'],
            timeout: 5000,
        }).trim();
    } catch (_) {
        return null;
    }
}

/**
 * Detect the Python interpreter available on this system.
 * Tries python3, python, py in order. Returns { found, version, cmd }.
 * @returns {{ found: boolean, version: string|null, cmd: string|null }}
 */
function detectPython() {
    const commands = ['python3', 'python', 'py'];
    for (const cmd of commands) {
        const output = tryExec(`${cmd} --version`);
        if (output && output.includes('Python 3')) {
            const match = output.match(/Python\s+([\d.]+)/);
            return { found: true, version: match ? match[1] : output, cmd };
        }
    }
    return { found: false, version: null, cmd: null };
}

/**
 * Detect whether Claude Code CLI is installed.
 * @returns {{ found: boolean, version: string|null }}
 */
function detectClaude() {
    const commands = process.platform === 'win32'
        ? ['claude', 'claude.cmd', 'claude-code', 'claude-code.cmd']
        : ['claude', 'claude-code'];

    for (const cmd of commands) {
        const output = tryExec(`${cmd} --version`);
        if (output) {
            const firstLine = output.split('\n')[0].trim();
            return { found: true, version: firstLine };
        }
    }
    return { found: false, version: null };
}

/**
 * Check whether Ollama is running and list available models.
 * Makes an HTTP GET to localhost:11434/api/tags with a 2-second timeout.
 * @returns {Promise<{ running: boolean, models: string[] }>}
 */
function detectOllama() {
    return new Promise((resolve) => {
        const req = http.get('http://localhost:11434/api/tags', { timeout: 2000 }, (res) => {
            let data = '';
            res.on('data', (chunk) => { data += chunk; });
            res.on('end', () => {
                try {
                    const json = JSON.parse(data);
                    const models = (json.models || []).map((m) => {
                        const name = m.name || '';
                        return name.includes(':') ? name.split(':')[0] : name;
                    });
                    resolve({ running: true, models });
                } catch (_) {
                    resolve({ running: false, models: [] });
                }
            });
        });

        req.on('error', () => resolve({ running: false, models: [] }));
        req.on('timeout', () => {
            req.destroy();
            resolve({ running: false, models: [] });
        });
    });
}

/**
 * Detect the full environment: OS, Node, Python, Claude Code, and Ollama.
 * @returns {Promise<Object>} Environment information object
 */
async function detectEnvironment() {
    const python = detectPython();
    const claude = detectClaude();
    const ollama = await detectOllama();

    return {
        os: {
            platform: os.platform(),
            arch: os.arch(),
            release: os.release(),
        },
        node: {
            version: process.version,
            path: process.execPath,
        },
        python,
        claude,
        ollama,
    };
}

/**
 * Print the environment detection results in a formatted, colored layout.
 * @param {Object} env - Environment object from detectEnvironment()
 */
function printEnvironment(env) {
    const ok = chalk.green('\u2713');
    const warn = chalk.yellow('!');

    console.log('');
    console.log(chalk.bold('  Environment Detection'));
    console.log(chalk.dim('  ' + '\u2500'.repeat(40)));

    // OS
    console.log(`  ${ok} ${chalk.bold('OS')}         ${env.os.platform} ${env.os.arch} (${env.os.release})`);

    // Node
    console.log(`  ${ok} ${chalk.bold('Node.js')}    ${env.node.version}`);

    // Python
    if (env.python.found) {
        console.log(`  ${ok} ${chalk.bold('Python')}     ${env.python.version} ${chalk.dim('(' + env.python.cmd + ')')}`);
    } else {
        console.log(`  ${warn} ${chalk.bold('Python')}     ${chalk.yellow('not found')} ${chalk.dim('(install Python 3.9+)')}`);
    }

    // Claude Code
    if (env.claude.found) {
        console.log(`  ${ok} ${chalk.bold('Claude')}     ${env.claude.version}`);
    } else {
        console.log(`  ${warn} ${chalk.bold('Claude')}     ${chalk.yellow('not found')} ${chalk.dim('(npm i -g @anthropic-ai/claude-code)')}`);
    }

    // Ollama
    if (env.ollama.running) {
        console.log(`  ${ok} ${chalk.bold('Ollama')}     running`);
        if (env.ollama.models.length > 0) {
            const modelList = env.ollama.models.join(', ');
            console.log(`             ${chalk.dim('models: ' + modelList)}`);
        } else {
            console.log(`             ${chalk.dim('no models pulled')}`);
        }
    } else {
        console.log(`  ${warn} ${chalk.bold('Ollama')}     ${chalk.yellow('not running')} ${chalk.dim('(optional - for ollama provider)')}`);
    }

    console.log('');
}

module.exports = { detectEnvironment, printEnvironment };
