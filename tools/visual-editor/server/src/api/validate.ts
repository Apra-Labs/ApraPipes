/**
 * Validation API routes
 */

import { Router, type Request, type Response } from 'express';
import { getValidator } from '../services/Validator.js';
import { getSchemaLoader } from '../services/SchemaLoader.js';
import { logger } from '../utils/logger.js';
import type { PipelineConfig } from '../types/validation.js';

const router = Router();

/**
 * POST /api/validate
 * Validate a pipeline configuration
 *
 * Request body: PipelineConfig
 * Response: ValidationResult
 */
router.post('/', async (req: Request, res: Response) => {
  try {
    const config = req.body as PipelineConfig;

    // Basic validation of request body
    if (!config || typeof config !== 'object') {
      res.status(400).json({
        valid: false,
        issues: [
          {
            level: 'error',
            code: 'E000',
            message: 'Invalid request body: expected a pipeline configuration object',
            location: '',
          },
        ],
      });
      return;
    }

    // Ensure modules and connections exist
    if (!config.modules || typeof config.modules !== 'object') {
      res.status(400).json({
        valid: false,
        issues: [
          {
            level: 'error',
            code: 'E000',
            message: 'Invalid request body: missing "modules" object',
            location: 'modules',
          },
        ],
      });
      return;
    }

    if (!Array.isArray(config.connections)) {
      // Allow missing connections array, default to empty
      config.connections = [];
    }

    // Validate the pipeline
    const schemaLoader = getSchemaLoader();
    const validator = getValidator(schemaLoader);
    const result = await validator.validate(config);

    logger.info(`Validation complete: valid=${result.valid}, issues=${result.issues.length}`);
    res.json(result);
  } catch (error) {
    logger.error('Validation error:', error);
    res.status(500).json({
      valid: false,
      issues: [
        {
          level: 'error',
          code: 'E999',
          message: 'Internal validation error',
          location: '',
        },
      ],
    });
  }
});

export default router;
