'use strict';

const chalk = require('chalk');

/**
 * Embedding models available through the Ollama provider.
 */
const OLLAMA_MODELS = [
    { name: 'all-minilm', tier: 'Light', dim: 384, size: '~46MB', speed: 5, desc: 'Quick searches, low resources' },
    { name: 'nomic-embed-text', tier: 'Standard', dim: 768, size: '~274MB', speed: 4, desc: 'General purpose (recommended)', recommended: true },
    { name: 'mxbai-embed-large', tier: 'Pro', dim: 1024, size: '~670MB', speed: 3, desc: 'High-precision recall' },
    { name: 'snowflake-arctic-embed', tier: 'Pro', dim: 1024, size: '~670MB', speed: 3, desc: 'Multilingual projects' },
    { name: 'bge-m3', tier: 'Pro', dim: 1024, size: '~1.2GB', speed: 2, desc: 'Dense retrieval, research' },
];

/**
 * Embedding models for the sentence-transformers (standalone) provider.
 */
const STANDALONE_MODELS = [
    { name: 'all-MiniLM-L6-v2', tier: 'Light', dim: 384, size: '~80MB', speed: 5, desc: 'Fastest, minimal resources' },
    { name: 'BAAI/bge-base-en-v1.5', tier: 'Standard', dim: 768, size: '~440MB', speed: 4, desc: 'Good balance (recommended)', recommended: true },
    { name: 'Alibaba-NLP/gte-large-en-v1.5', tier: 'Pro', dim: 1024, size: '~1.5GB', speed: 3, desc: 'Best quality English' },
];

/**
 * Return the model list for a given provider.
 * @param {'ollama'|'sentence-transformers'} provider
 * @returns {Array<Object>}
 */
function getModelsByProvider(provider) {
    if (provider === 'ollama') {
        return OLLAMA_MODELS;
    }
    return STANDALONE_MODELS;
}

/**
 * Build a star rating string: filled stars followed by empty stars, colored yellow.
 * @param {number} count - Number of filled stars (out of 5)
 * @returns {string}
 */
function starsString(count) {
    const filled = '\u2605'.repeat(count);
    const empty = '\u2606'.repeat(5 - count);
    return chalk.yellow(filled + empty);
}

/**
 * Format a model object into a display string suitable for inquirer choices.
 *
 * Example output:
 *   [Light]  all-minilm         384d  ~46MB   *****  Quick searches [installed]
 *
 * @param {Object} model - Model object with name, tier, dim, size, speed, desc
 * @param {string[]} installedModels - Array of model names currently installed
 * @returns {string}
 */
function formatModelChoice(model, installedModels) {
    const tierLabel = chalk.cyan('[' + model.tier + ']');
    const tierPad = 10 - model.tier.length - 2; // account for brackets
    const tierSpacing = ' '.repeat(Math.max(tierPad, 1));

    const namePad = 26 - model.name.length;
    const nameSpacing = ' '.repeat(Math.max(namePad, 1));

    const dimStr = chalk.dim(model.dim + 'd');
    const dimPad = 6 - String(model.dim).length;
    const dimSpacing = ' '.repeat(Math.max(dimPad, 1));

    const sizeStr = chalk.dim(model.size);
    const sizePad = 8 - model.size.length;
    const sizeSpacing = ' '.repeat(Math.max(sizePad, 1));

    const rating = starsString(model.speed);

    const isInstalled = Array.isArray(installedModels) && installedModels.includes(model.name);
    const installedBadge = isInstalled ? chalk.green(' [installed]') : '';

    return (
        tierLabel + tierSpacing +
        model.name + nameSpacing +
        dimStr + dimSpacing +
        sizeStr + sizeSpacing +
        rating + '  ' +
        model.desc +
        installedBadge
    );
}

module.exports = {
    OLLAMA_MODELS,
    STANDALONE_MODELS,
    getModelsByProvider,
    formatModelChoice,
};
