#!/usr/bin/env node
'use strict';

/**
 * Post-installation script for Claude Memory Agent
 * Dispatches to the onboarding wizard (interactive) or installs with defaults (CI/piped).
 */

const path = require('path');
const fs = require('fs');

// Fix executable permissions on Unix (Windows npm pack doesn't preserve them)
function fixPermissions() {
  if (process.platform === 'win32') return;
  try {
    const cliPath = path.join(__dirname, 'cli.js');
    fs.chmodSync(cliPath, 0o755);
  } catch (_) {
    // Non-fatal — user can chmod manually
  }
}

async function main() {
  fixPermissions();

  // Skip in CI
  if (process.env.CI) {
    console.log('Skipping post-install in CI environment');
    return;
  }

  try {
    const { main: runOnboarding } = require('./onboarding');
    await runOnboarding();
  } catch (err) {
    // Fallback: if onboarding deps missing, do basic setup
    console.log('Running basic setup (onboarding dependencies not available)...');
    try {
      const { basicSetup } = require('./lib/installer');
      await basicSetup(path.dirname(__dirname));
    } catch (fallbackErr) {
      console.log('Post-install: run "claude-memory-agent install" to complete setup.');
    }
  }
}

main().catch(() => {
  console.log('Post-install: run "claude-memory-agent install" to complete setup.');
});
