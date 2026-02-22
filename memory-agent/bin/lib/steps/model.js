'use strict';

const { select, confirm } = require('@inquirer/prompts');
const chalk = require('chalk');
const { stepHeader, TOTAL_STEPS } = require('../ui.js');
const { getModelsByProvider, formatModelChoice } = require('../models.js');

/**
 * Step 4: Embedding Model
 * Prompts the user to choose an embedding model based on their selected provider.
 *
 * @param {'ollama'|'sentence-transformers'} provider
 * @param {{ ollama: { running: boolean, models: string[] } }} env
 * @returns {{ model: string, pullModel: boolean }}
 */
async function promptModel(provider, env) {
    stepHeader(4, TOTAL_STEPS, 'Embedding Model');

    console.log(chalk.dim('  Models are ordered from lightest to most powerful:\n'));

    const installedModels = (env.ollama && env.ollama.models) || [];
    const models = getModelsByProvider(provider);

    const choices = models.map((model) => ({
        name: formatModelChoice(model, installedModels),
        value: model.name,
    }));

    // Find and mark the recommended model as default
    const recommendedIndex = models.findIndex((m) => m.recommended);
    if (recommendedIndex !== -1) {
        choices[recommendedIndex].default = true;
    }

    const defaultModel = recommendedIndex !== -1 ? models[recommendedIndex].name : undefined;

    const model = await select({
        message: 'Select an embedding model:',
        choices,
        default: defaultModel,
    });

    let pullModel = false;

    if (provider === 'ollama' && !installedModels.includes(model)) {
        console.log(chalk.yellow('\n  This model is not installed in Ollama.'));

        pullModel = await confirm({
            message: `Download it now? (ollama pull ${model})`,
            default: true,
        });
    }

    return { model, pullModel };
}

module.exports = { promptModel };
