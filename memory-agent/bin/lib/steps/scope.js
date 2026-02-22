'use strict';

const { select, input } = require('@inquirer/prompts');
const fs = require('fs');
const path = require('path');
const chalk = require('chalk');
const { stepHeader, TOTAL_STEPS } = require('../ui.js');

/**
 * Step 2: Installation Scope
 * Prompts the user to choose where the memory agent configuration is installed.
 *
 * @returns {{ scope: 'global'|'project'|'both', projectPath: string|null }}
 */
async function promptScope() {
    stepHeader(2, TOTAL_STEPS, 'Installation Scope');

    const scope = await select({
        message: 'Where should the memory agent be configured?',
        choices: [
            {
                name: 'Global (recommended)',
                value: 'global',
                description: 'Configures ~/.claude/settings.json',
            },
            {
                name: 'Project-specific',
                value: 'project',
                description: 'Configures .claude/settings.local.json in your project',
            },
            {
                name: 'Both',
                value: 'both',
                description: 'Global config + project-specific override',
            },
        ],
    });

    let projectPath = null;

    if (scope === 'project' || scope === 'both') {
        projectPath = await input({
            message: 'Project directory path:',
            default: process.cwd(),
            validate: (value) => {
                const resolved = path.resolve(value);
                if (!fs.existsSync(resolved)) {
                    return `Directory does not exist: ${resolved}`;
                }
                return true;
            },
        });
        projectPath = path.resolve(projectPath);
    }

    return { scope, projectPath };
}

module.exports = { promptScope };
