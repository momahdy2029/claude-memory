'use strict';

const { select, input, confirm, checkbox } = require('@inquirer/prompts');
const chalk = require('chalk');
const { stepHeader, TOTAL_STEPS } = require('../ui.js');

/**
 * Step 6: Advanced Settings
 * Prompts the user for optional advanced configuration, or returns null for defaults.
 *
 * @returns {{ logLevel: string, dbPath: string, authEnabled: boolean,
 *             hotTierDays: number, warmTierDays: number, hooks: string[] } | null}
 */
async function promptAdvanced() {
    stepHeader(6, TOTAL_STEPS, 'Advanced Settings');

    const mode = await select({
        message: 'Configuration mode:',
        choices: [
            {
                name: 'Use defaults (recommended)',
                value: 'defaults',
            },
            {
                name: 'Customize',
                value: 'customize',
            },
        ],
    });

    if (mode === 'defaults') {
        return null;
    }

    const logLevel = await select({
        message: 'Log level:',
        choices: [
            { name: 'DEBUG', value: 'DEBUG' },
            { name: 'INFO', value: 'INFO' },
            { name: 'WARNING', value: 'WARNING' },
            { name: 'ERROR', value: 'ERROR' },
        ],
        default: 'INFO',
    });

    const dbPath = await input({
        message: 'Database path (leave empty for ~/.claude-memory/):',
        default: '',
    });

    const authEnabled = await confirm({
        message: 'Enable API authentication?',
        default: false,
    });

    const hotTierDaysStr = await input({
        message: 'Hot memory tier (days):',
        default: '14',
        validate: (value) => {
            const num = Number(value);
            if (!Number.isInteger(num) || num < 1) {
                return 'Must be a positive integer';
            }
            return true;
        },
    });

    const warmTierDaysStr = await input({
        message: 'Warm memory tier (days):',
        default: '90',
        validate: (value) => {
            const num = Number(value);
            if (!Number.isInteger(num) || num < 1) {
                return 'Must be a positive integer';
            }
            return true;
        },
    });

    const hooks = await checkbox({
        message: 'Enable hooks:',
        choices: [
            { name: 'session_start', value: 'session_start', checked: true },
            { name: 'grounding', value: 'grounding', checked: true },
            { name: 'session_end', value: 'session_end', checked: true },
            { name: 'problem_detector', value: 'problem_detector', checked: false },
            { name: 'auto_capture', value: 'auto_capture', checked: false },
        ],
    });

    return {
        logLevel,
        dbPath,
        authEnabled,
        hotTierDays: Number(hotTierDaysStr),
        warmTierDays: Number(warmTierDaysStr),
        hooks,
    };
}

module.exports = { promptAdvanced };
