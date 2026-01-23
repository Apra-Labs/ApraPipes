import express from 'express';
import cors from 'cors';
import { createServer } from 'http';
import { config } from './config.js';
import schemaRouter from './api/schema.js';
import { workspaceRouter } from './api/workspace.js';
import { createLogger } from './utils/logger.js';

const logger = createLogger('Server');

const app = express();
const server = createServer(app);

// Middleware
app.use(cors());
app.use(express.json());

// API routes
app.use('/api/schema', schemaRouter);
app.use('/api/workspace', workspaceRouter);

// Health check
app.get('/api/health', (_req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Start server
server.listen(config.port, config.host, () => {
  logger.info(`ApraPipes Studio server running at http://${config.host}:${config.port}`);
  logger.info('Press Ctrl+C to stop');
});

// Graceful shutdown
process.on('SIGINT', () => {
  logger.info('Shutting down...');
  server.close(() => {
    logger.info('Server stopped');
    process.exit(0);
  });
});

export { app, server };
