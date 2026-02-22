#!/usr/bin/env node
'use strict';

const path = require('path');
const { spawn } = require('child_process');

const AGENT_DIR = path.resolve(path.dirname(__dirname));

async function runWizard() {
  // Import lib modules
  const { printBanner } = require('./lib/banner');
  const { detectEnvironment, printEnvironment } = require('./lib/environment');
  const { stepHeader, TOTAL_STEPS } = require('./lib/ui');
  const { promptScope } = require('./lib/steps/scope');
  const { promptProvider } = require('./lib/steps/provider');
  const { promptModel } = require('./lib/steps/model');
  const { promptServer } = require('./lib/steps/server');
  const { promptAdvanced } = require('./lib/steps/advanced');
  const { promptConfirm } = require('./lib/steps/confirm');
  const { runInstaller } = require('./lib/installer');

  // Screen 1: Banner + Environment Detection
  printBanner();
  stepHeader(1, TOTAL_STEPS, 'Environment Detection');
  const env = await detectEnvironment();
  printEnvironment(env);

  console.log(''); // spacing

  // Screen 2: Installation Scope
  const { scope, projectPath } = await promptScope();

  // Screen 3: Embedding Provider
  const { provider } = await promptProvider(env);

  // Screen 4: Model Selection
  const { model, pullModel } = await promptModel(provider, env);

  // Pull Ollama model if needed
  if (pullModel && provider === 'ollama') {
    const ora = require('ora');
    const spinner = ora(`Downloading ${model}...`).start();
    try {
      const { execSync } = require('child_process');
      execSync(`ollama pull ${model}`, { stdio: ['pipe', 'pipe', 'pipe'], timeout: 300000 });
      spinner.succeed(`Downloaded ${model}`);
    } catch (e) {
      spinner.fail(`Failed to download ${model}. You can run "ollama pull ${model}" manually later.`);
    }
  }

  // Screen 5: Server Settings
  const { port, host, autoStart } = await promptServer();

  // Screen 6: Advanced Settings
  const advancedResult = await promptAdvanced();

  // Build final config
  const config = {
    scope,
    projectPath,
    provider,
    model,
    port,
    host,
    autoStart,
    logLevel: advancedResult ? advancedResult.logLevel : 'INFO',
    dbPath: advancedResult ? advancedResult.dbPath : null,
    authEnabled: advancedResult ? advancedResult.authEnabled : false,
    hotTierDays: advancedResult ? advancedResult.hotTierDays : 14,
    warmTierDays: advancedResult ? advancedResult.warmTierDays : 90,
    hooks: advancedResult ? advancedResult.hooks : ['session_start', 'grounding', 'session_end'],
  };

  // Screen 7: Review & Confirm
  const confirmed = await promptConfirm(config);
  if (!confirmed) {
    console.log('\n  Installation cancelled.\n');
    process.exit(0);
  }

  // Run Installation
  console.log('');
  const result = await runInstaller(config, AGENT_DIR);

  if (result.success) {
    const chalk = require('chalk');
    const boxen = require('boxen');

    const completionMsg = [
      '',
      chalk.green.bold('  Installation Complete!'),
      '',
      `  Dashboard:  ${chalk.cyan(`http://localhost:${port}/dashboard`)}`,
      `  Health:     ${chalk.cyan(`http://localhost:${port}/health`)}`,
      '',
      chalk.dim('  Quick commands:'),
      `    ${chalk.bold('claude-memory-agent status')}    - Check agent status`,
      `    ${chalk.bold('claude-memory-agent dashboard')} - Open dashboard`,
      `    ${chalk.bold('claude-memory-agent doctor')}    - Diagnose issues`,
      `    ${chalk.bold('claude-memory-agent logs')}      - View logs`,
      '',
    ].join('\n');

    console.log(boxen(completionMsg, {
      padding: 1,
      margin: 1,
      borderStyle: 'round',
      borderColor: 'green',
    }));
  } else {
    console.log('\n  Installation completed with errors:');
    result.errors.forEach(e => console.log(`    - ${e}`));
    console.log('');
  }
}

// Non-interactive fallback for CI/piped
function runDefaults() {
  const { runInstaller } = require('./lib/installer');
  const config = {
    scope: 'global',
    projectPath: null,
    provider: 'sentence-transformers',
    model: 'BAAI/bge-base-en-v1.5',
    port: 8102,
    host: '127.0.0.1',
    autoStart: false,
    logLevel: 'INFO',
    dbPath: null,
    authEnabled: false,
    hotTierDays: 14,
    warmTierDays: 90,
    hooks: ['session_start', 'grounding', 'session_end'],
  };

  return runInstaller(config, AGENT_DIR);
}

// Main entry
async function main() {
  try {
    if (!process.stdin.isTTY) {
      // Non-interactive: use defaults silently
      await runDefaults();
    } else {
      await runWizard();
    }
  } catch (err) {
    if (err.name === 'ExitPromptError') {
      // User pressed Ctrl+C
      console.log('\n  Setup cancelled.\n');
      process.exit(0);
    }
    console.error('Setup error:', err.message);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = { main, runWizard, runDefaults };
