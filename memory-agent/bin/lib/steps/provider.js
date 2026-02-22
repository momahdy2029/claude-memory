'use strict';

const { select, confirm } = require('@inquirer/prompts');
const chalk = require('chalk');
const { stepHeader, printBox, TOTAL_STEPS } = require('../ui.js');

/**
 * Step 3: Embedding Provider
 * Prompts the user to choose between Ollama and sentence-transformers.
 *
 * @param {{ ollama: { running: boolean, models: string[] } }} env
 * @returns {{ provider: 'ollama'|'sentence-transformers' }}
 */
async function promptProvider(env) {
    stepHeader(3, TOTAL_STEPS, 'Embedding Provider');

    const ollamaRunning = env.ollama && env.ollama.running;
    const ollamaStatus = ollamaRunning
        ? chalk.green(' (running)')
        : chalk.yellow(' (not detected)');

    let provider = null;

    while (provider === null) {
        const choice = await select({
            message: 'Which embedding provider would you like to use?',
            choices: [
                {
                    name: `Standalone (sentence-transformers) - runs locally, no external service needed`,
                    value: 'sentence-transformers',
                },
                {
                    name: `Ollama - uses Ollama for embeddings${ollamaStatus}`,
                    value: 'ollama',
                },
            ],
        });

        if (choice === 'ollama' && !ollamaRunning) {
            printBox('Warning', [
                chalk.yellow('Ollama is not running. To install:'),
                '',
                '  1. Download from https://ollama.ai/download',
                '  2. Run: ollama serve',
                '  3. Then re-run this installer',
            ]);

            const continueAnyway = await confirm({
                message: 'Continue with Ollama anyway?',
                default: false,
            });

            if (!continueAnyway) {
                // Re-prompt by continuing the loop
                continue;
            }
        }

        provider = choice;
    }

    return { provider };
}

module.exports = { promptProvider };
