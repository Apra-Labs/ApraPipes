/**
 * Pipeline API Routes
 *
 * Endpoints for pipeline lifecycle management:
 * - POST /api/pipeline/create - Create a new pipeline
 * - POST /api/pipeline/:id/start - Start a pipeline
 * - POST /api/pipeline/:id/stop - Stop a pipeline
 * - GET /api/pipeline/:id/status - Get pipeline status
 * - DELETE /api/pipeline/:id - Delete a pipeline
 */

import { Router, Request, Response } from 'express';
import { getPipelineManager } from '../services/PipelineManager.js';
import { createLogger } from '../utils/logger.js';
import type {
  CreatePipelineRequest,
  CreatePipelineResponse,
  PipelineStatusResponse,
  OperationResponse,
  PipelineConfig,
} from '../types/pipeline.js';

const logger = createLogger('PipelineAPI');
const router = Router();

/**
 * POST /api/pipeline/create
 * Create a new pipeline from configuration
 */
router.post('/create', (req: Request, res: Response) => {
  try {
    const body = req.body as CreatePipelineRequest | PipelineConfig;

    // Support both { config: ... } and direct config
    const config: PipelineConfig = 'config' in body ? body.config : body;

    // Validate config structure
    if (!config || typeof config !== 'object') {
      res.status(400).json({
        success: false,
        message: 'Invalid request: config is required',
      });
      return;
    }

    if (!config.modules || typeof config.modules !== 'object') {
      res.status(400).json({
        success: false,
        message: 'Invalid request: config.modules is required',
      });
      return;
    }

    if (!Array.isArray(config.connections)) {
      res.status(400).json({
        success: false,
        message: 'Invalid request: config.connections must be an array',
      });
      return;
    }

    const manager = getPipelineManager();
    const pipelineId = manager.create(config);

    logger.info(`Pipeline created via API: ${pipelineId}`);

    const response: CreatePipelineResponse = {
      pipelineId,
      status: 'IDLE',
    };

    res.status(201).json(response);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    logger.error('Failed to create pipeline:', error);
    res.status(500).json({
      success: false,
      message: `Failed to create pipeline: ${message}`,
    });
  }
});

/**
 * POST /api/pipeline/:id/start
 * Start a pipeline
 */
router.post('/:id/start', async (req: Request, res: Response) => {
  try {
    const { id } = req.params;
    const manager = getPipelineManager();

    const instance = manager.get(id);
    if (!instance) {
      res.status(404).json({
        success: false,
        message: `Pipeline not found: ${id}`,
      });
      return;
    }

    await manager.start(id);

    logger.info(`Pipeline started via API: ${id}`);

    const response: OperationResponse = {
      success: true,
      status: manager.getStatus(id) || 'RUNNING',
      message: 'Pipeline started',
    };

    res.json(response);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    logger.error(`Failed to start pipeline ${req.params.id}:`, error);

    const manager = getPipelineManager();
    const status = manager.getStatus(req.params.id);

    res.status(400).json({
      success: false,
      status: status || 'ERROR',
      message,
    });
  }
});

/**
 * POST /api/pipeline/:id/stop
 * Stop a running pipeline
 */
router.post('/:id/stop', async (req: Request, res: Response) => {
  try {
    const { id } = req.params;
    const manager = getPipelineManager();

    const instance = manager.get(id);
    if (!instance) {
      res.status(404).json({
        success: false,
        message: `Pipeline not found: ${id}`,
      });
      return;
    }

    await manager.stop(id);

    logger.info(`Pipeline stopped via API: ${id}`);

    const response: OperationResponse = {
      success: true,
      status: manager.getStatus(id) || 'STOPPED',
      message: 'Pipeline stopped',
    };

    res.json(response);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    logger.error(`Failed to stop pipeline ${req.params.id}:`, error);

    const manager = getPipelineManager();
    const status = manager.getStatus(req.params.id);

    res.status(400).json({
      success: false,
      status: status || 'ERROR',
      message,
    });
  }
});

/**
 * GET /api/pipeline/:id/status
 * Get current pipeline status and metrics
 */
router.get('/:id/status', (req: Request, res: Response) => {
  try {
    const { id } = req.params;
    const manager = getPipelineManager();

    const instance = manager.get(id);
    if (!instance) {
      res.status(404).json({
        success: false,
        message: `Pipeline not found: ${id}`,
      });
      return;
    }

    const response: PipelineStatusResponse = {
      id: instance.id,
      status: instance.status,
      metrics: instance.metrics,
      errors: instance.errors,
      startTime: instance.startTime,
      duration: instance.startTime ? Date.now() - instance.startTime : undefined,
    };

    res.json(response);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    logger.error(`Failed to get pipeline status ${req.params.id}:`, error);
    res.status(500).json({
      success: false,
      message,
    });
  }
});

/**
 * DELETE /api/pipeline/:id
 * Delete a pipeline and cleanup resources
 */
router.delete('/:id', async (req: Request, res: Response) => {
  try {
    const { id } = req.params;
    const manager = getPipelineManager();

    const instance = manager.get(id);
    if (!instance) {
      res.status(404).json({
        success: false,
        message: `Pipeline not found: ${id}`,
      });
      return;
    }

    await manager.delete(id);

    logger.info(`Pipeline deleted via API: ${id}`);

    res.json({
      success: true,
      message: 'Pipeline deleted',
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    logger.error(`Failed to delete pipeline ${req.params.id}:`, error);
    res.status(500).json({
      success: false,
      message,
    });
  }
});

/**
 * GET /api/pipeline/list
 * List all pipelines
 */
router.get('/list', (_req: Request, res: Response) => {
  try {
    const manager = getPipelineManager();
    const ids = manager.list();

    const pipelines = ids.map((id) => {
      const instance = manager.get(id);
      return {
        id,
        status: instance?.status || 'UNKNOWN',
        moduleCount: instance ? Object.keys(instance.config.modules).length : 0,
        startTime: instance?.startTime,
      };
    });

    res.json({
      pipelines,
      count: pipelines.length,
      mockMode: manager.isMockMode(),
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    logger.error('Failed to list pipelines:', error);
    res.status(500).json({
      success: false,
      message,
    });
  }
});

export { router as pipelineRouter };
export default router;
