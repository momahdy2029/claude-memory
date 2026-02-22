'use strict';

const { input, select, confirm } = require('@inquirer/prompts');
const chalk = require('chalk');
const net = require('net');
const { stepHeader, TOTAL_STEPS } = require('../ui.js');

/**
 * Check if a port is available by attempting to listen on it briefly.
 *
 * @param {number} port
 * @returns {Promise<boolean>} true if the port is available
 */
function isPortAvailable(port) {
    return new Promise((resolve) => {
        const server = net.createServer();
        server.once('error', () => resolve(false));
        server.once('listening', () => {
            server.close(() => resolve(true));
        });
        server.listen(port, '127.0.0.1');
    });
}

/**
 * Step 5: Server Settings
 * Prompts the user for port, host binding, and auto-start preference.
 *
 * @returns {{ port: number, host: string, autoStart: boolean }}
 */
async function promptServer() {
    stepHeader(5, TOTAL_STEPS, 'Server Settings');

    const portStr = await input({
        message: 'Server port:',
        default: '8102',
        validate: async (value) => {
            const num = Number(value);
            if (!Number.isInteger(num) || num < 1 || num > 65535) {
                return 'Port must be a number between 1 and 65535';
            }
            const available = await isPortAvailable(num);
            if (!available) {
                return `Port ${num} is already in use`;
            }
            return true;
        },
    });

    const port = Number(portStr);

    const host = await select({
        message: 'Dashboard binding:',
        choices: [
            {
                name: 'Localhost only (127.0.0.1) - Secure, recommended',
                value: '127.0.0.1',
            },
            {
                name: 'All interfaces (0.0.0.0) - Access from other devices',
                value: '0.0.0.0',
            },
        ],
    });

    const autoStart = await confirm({
        message: 'Start the agent after installation?',
        default: true,
    });

    return { port, host, autoStart };
}

module.exports = { promptServer };
