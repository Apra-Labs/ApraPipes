#!/usr/bin/env node

/**
 * GitHub Webhook Listener for ApraPipes CI Monitor
 *
 * Receives workflow_run events from GitHub and triggers immediate diagnosis
 * instead of waiting for polling interval.
 *
 * Usage:
 *   node webhook-listener.js
 *
 * Environment Variables:
 *   PORT - Port to listen on (default: 9876)
 *   WEBHOOK_SECRET - GitHub webhook secret for signature validation
 *   GH_TOKEN - GitHub personal access token
 */

const http = require('http');
const crypto = require('crypto');
const { exec } = require('child_process');
const util = require('util');
const execPromise = util.promisify(exec);

const PORT = process.env.PORT || 9876;
const WEBHOOK_SECRET = process.env.WEBHOOK_SECRET || '';
const MONITOR_SCRIPT = __dirname + '/ci-monitor.sh';

// Colors for console output
const colors = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
};

function log(level, message) {
  const timestamp = new Date().toISOString();
  const color = {
    INFO: colors.blue,
    WARN: colors.yellow,
    ERROR: colors.red,
    SUCCESS: colors.green,
  }[level] || colors.reset;

  console.log(`${color}[${timestamp}] [${level}]${colors.reset} ${message}`);
}

/**
 * Verify GitHub webhook signature
 */
function verifySignature(payload, signature) {
  if (!WEBHOOK_SECRET) {
    log('WARN', 'WEBHOOK_SECRET not set, skipping signature verification');
    return true;
  }

  const hmac = crypto.createHmac('sha256', WEBHOOK_SECRET);
  const digest = 'sha256=' + hmac.update(payload).digest('hex');

  return crypto.timingSafeEqual(
    Buffer.from(signature),
    Buffer.from(digest)
  );
}

/**
 * Handle workflow_run event
 */
async function handleWorkflowRun(event) {
  const { action, workflow_run } = event;
  const { id, name, status, conclusion, head_branch } = workflow_run;

  log('INFO', `Workflow: ${name}, Status: ${status}, Conclusion: ${conclusion || 'N/A'}`);
  log('INFO', `  Run ID: ${id}, Branch: ${head_branch}`);

  // Only care about completed failures
  if (action !== 'completed' || conclusion !== 'failure') {
    log('INFO', '  → Not a failure, ignoring');
    return;
  }

  log('ERROR', `  → Failure detected! Triggering diagnosis...`);

  // Call the monitor script's diagnose function directly
  try {
    const { stdout, stderr } = await execPromise(
      `bash -c 'source ${MONITOR_SCRIPT}; diagnose_and_fix "${name}" "${id}" "${head_branch}"'`
    );

    if (stdout) log('INFO', `  Diagnosis output: ${stdout.trim()}`);
    if (stderr) log('WARN', `  Diagnosis stderr: ${stderr.trim()}`);

    log('SUCCESS', `  → Diagnosis completed for run ${id}`);
  } catch (error) {
    log('ERROR', `  → Diagnosis failed: ${error.message}`);
  }
}

/**
 * HTTP request handler
 */
const server = http.createServer(async (req, res) => {
  if (req.method !== 'POST' || req.url !== '/webhook') {
    res.writeHead(404);
    res.end('Not Found');
    return;
  }

  let body = '';

  req.on('data', chunk => {
    body += chunk.toString();
  });

  req.on('end', async () => {
    const signature = req.headers['x-hub-signature-256'];
    const event = req.headers['x-github-event'];

    log('INFO', `Received ${event} event`);

    // Verify signature
    if (signature && !verifySignature(body, signature)) {
      log('ERROR', 'Invalid signature!');
      res.writeHead(401);
      res.end('Invalid signature');
      return;
    }

    // Parse payload
    let payload;
    try {
      payload = JSON.parse(body);
    } catch (error) {
      log('ERROR', `Failed to parse payload: ${error.message}`);
      res.writeHead(400);
      res.end('Bad Request');
      return;
    }

    // Handle the event
    try {
      if (event === 'workflow_run') {
        await handleWorkflowRun(payload);
      } else if (event === 'ping') {
        log('SUCCESS', 'Ping received - webhook configured correctly!');
      } else {
        log('INFO', `Ignoring event type: ${event}`);
      }

      res.writeHead(200);
      res.end('OK');
    } catch (error) {
      log('ERROR', `Error handling event: ${error.message}`);
      res.writeHead(500);
      res.end('Internal Server Error');
    }
  });
});

server.listen(PORT, () => {
  log('SUCCESS', `🚀 Webhook listener started on port ${PORT}`);
  log('INFO', `Webhook URL: http://192.168.1.102:${PORT}/webhook`);
  log('INFO', 'Waiting for GitHub workflow events...');

  if (!WEBHOOK_SECRET) {
    log('WARN', '⚠️  WEBHOOK_SECRET not set - signature verification disabled!');
    log('WARN', '   Set WEBHOOK_SECRET environment variable for production use');
  }
});

// Graceful shutdown
process.on('SIGTERM', () => {
  log('INFO', 'SIGTERM received, shutting down gracefully...');
  server.close(() => {
    log('INFO', 'Server closed');
    process.exit(0);
  });
});

process.on('SIGINT', () => {
  log('INFO', 'SIGINT received, shutting down gracefully...');
  server.close(() => {
    log('INFO', 'Server closed');
    process.exit(0);
  });
});
