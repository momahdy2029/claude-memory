'use strict';

const { confirm } = require('@inquirer/prompts');
const chalk = require('chalk');
const { stepHeader, printBox, TOTAL_STEPS } = require('../ui.js');

/**
 * Step 7: Review & Install
 * Displays a summary of all collected configuration and asks for confirmation.
 *
 * @param {{ scope: string, projectPath: string|null, provider: string, model: string,
 *           port: number, host: string, autoStart: boolean, logLevel?: string,
 *           dbPath?: string, authEnabled?: boolean, hotTierDays?: number,
 *           warmTierDays?: number, hooks?: string[] }} config
 * @returns {boolean} true if user confirms, false to abort
 */
async function promptConfirm(config) {
    stepHeader(7, TOTAL_STEPS, 'Review & Install');

    // Format scope display
    const scopeLabels = { global: 'Global', project: 'Project-specific', both: 'Both' };
    const scopeDisplay = scopeLabels[config.scope] || config.scope;

    // Format binding display
    const bindingDisplay = config.host === '127.0.0.1'
        ? 'localhost only'
        : 'all interfaces (0.0.0.0)';

    // Format auto-start display
    const autoStartDisplay = config.autoStart ? 'Yes' : 'No';

    // Format log level (use default if not customized)
    const logLevel = config.logLevel || 'INFO';

    // Format auth display
    const authDisplay = config.authEnabled ? 'Enabled' : 'Disabled';

    // Format hooks display
    const defaultHooks = ['session_start', 'grounding', 'session_end'];
    const hooks = config.hooks || defaultHooks;
    const hooksDisplay = hooks.join(', ');

    const label = (text) => chalk.dim(text);
    const value = (text) => chalk.white.bold(text);

    const lines = [
        `${label('Scope:')}      ${value(scopeDisplay)}`,
    ];

    if (config.projectPath) {
        lines.push(`${label('Project:')}    ${value(config.projectPath)}`);
    }

    lines.push(
        `${label('Provider:')}   ${value(config.provider)}`,
        `${label('Model:')}      ${value(config.model)}`,
        `${label('Port:')}       ${value(String(config.port))}`,
        `${label('Binding:')}    ${value(bindingDisplay)}`,
        `${label('Auto-start:')} ${value(autoStartDisplay)}`,
        `${label('Log Level:')}  ${value(logLevel)}`,
        `${label('Auth:')}       ${value(authDisplay)}`,
        `${label('Hooks:')}      ${value(hooksDisplay)}`,
    );

    if (config.dbPath) {
        lines.push(`${label('DB Path:')}    ${value(config.dbPath)}`);
    }

    if (config.hotTierDays) {
        lines.push(`${label('Hot Tier:')}   ${value(config.hotTierDays + ' days')}`);
    }

    if (config.warmTierDays) {
        lines.push(`${label('Warm Tier:')}  ${value(config.warmTierDays + ' days')}`);
    }

    printBox('Installation Summary', lines);

    const proceed = await confirm({
        message: 'Proceed with installation?',
        default: true,
    });

    return proceed;
}

module.exports = { promptConfirm };
