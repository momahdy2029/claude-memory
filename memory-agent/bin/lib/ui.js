'use strict';

const chalk = require('chalk');
const boxen = require('boxen');

/**
 * Total number of steps in the onboarding wizard.
 */
const TOTAL_STEPS = 7;

/**
 * Print a step header for the wizard.
 * Renders as:
 *
 *   Step 3/7 -- Embedding Model
 *   ----------------------------------------
 *
 * @param {number} step - Current step number
 * @param {number} total - Total number of steps
 * @param {string} title - Step title
 */
function stepHeader(step, total, title) {
    console.log('');
    console.log('  ' + chalk.bold.cyan('Step ' + step + '/' + total) + chalk.bold(' \u2014 ' + title));
    console.log('  ' + chalk.dim('\u2500'.repeat(40)));
    console.log('');
}

/**
 * Return a star rating string (filled + empty, out of 5), colored yellow.
 * @param {number} count - Number of filled stars
 * @returns {string}
 */
function stars(count) {
    const filled = '\u2605'.repeat(Math.min(count, 5));
    const empty = '\u2606'.repeat(Math.max(5 - count, 0));
    return chalk.yellow(filled + empty);
}

/**
 * Print a bordered box with a title and content lines.
 * Uses the boxen package for drawing.
 *
 * @param {string} title - Box title
 * @param {string[]} lines - Array of content lines to display inside the box
 */
function printBox(title, lines) {
    const content = lines.join('\n');
    const box = boxen(content, {
        title: title,
        titleAlignment: 'left',
        padding: 1,
        margin: { top: 0, bottom: 0, left: 2, right: 0 },
        borderColor: 'cyan',
        borderStyle: 'round',
    });
    console.log(box);
}

/**
 * Print a thin horizontal divider across the terminal width.
 * Falls back to 60 columns if terminal width is not available.
 */
function divider() {
    const width = process.stdout.columns || 60;
    console.log(chalk.dim('\u2500'.repeat(width)));
}

module.exports = {
    TOTAL_STEPS,
    stepHeader,
    stars,
    printBox,
    divider,
};
