'use strict';

const chalk = require('chalk');
const gradient = require('gradient-string');

const BANNER = `
  ██████╗██╗      █████╗ ██╗   ██╗██████╗ ███████╗
 ██╔════╝██║     ██╔══██╗██║   ██║██╔══██╗██╔════╝
 ██║     ██║     ███████║██║   ██║██║  ██║█████╗
 ██║     ██║     ██╔══██║██║   ██║██║  ██║██╔══╝
 ╚██████╗███████╗██║  ██║╚██████╔╝██████╔╝███████╗
  ╚═════╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝
 ███╗   ███╗███████╗███╗   ███╗ ██████╗ ██████╗ ██╗   ██╗
 ████╗ ████║██╔════╝████╗ ████║██╔═══██╗██╔══██╗╚██╗ ██╔╝
 ██╔████╔██║█████╗  ██╔████╔██║██║   ██║██████╔╝ ╚████╔╝
 ██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║██║   ██║██╔══██╗  ╚██╔╝
 ██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║╚██████╔╝██║  ██║   ██║
 ╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝`;

const VERSION = 'v2.1.0';

/**
 * Print the Claude Memory Agent ASCII art banner with gradient coloring.
 * Uses a cool blue-to-purple gradient (mind -> cristal).
 */
function printBanner() {
    const coolGradient = gradient(['#0575E6', '#7B68EE', '#A855F7', '#6C63FF']);
    console.log(coolGradient(BANNER));

    // Center the version string beneath the banner
    const bannerWidth = 56; // approximate width of the widest banner line
    const padding = Math.max(0, Math.floor((bannerWidth - VERSION.length) / 2));
    const centeredVersion = ' '.repeat(padding) + VERSION;
    console.log('');
    console.log(chalk.dim(centeredVersion));
    console.log('');
}

module.exports = { printBanner };
