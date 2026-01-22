import { Router, Request, Response } from 'express';
import { schemaLoader } from '../services/SchemaLoader.js';

const router = Router();

/**
 * GET /api/schema
 * Returns the module schema from schema_generator
 */
router.get('/', async (_req: Request, res: Response) => {
  try {
    const schema = await schemaLoader.load();
    res.json(schema);
  } catch (err) {
    res.status(500).json({
      error: 'Failed to load schema',
      message: err instanceof Error ? err.message : 'Unknown error',
    });
  }
});

/**
 * POST /api/schema/reload
 * Force reload the schema from disk
 */
router.post('/reload', async (_req: Request, res: Response) => {
  try {
    const schema = await schemaLoader.reload();
    res.json({
      success: true,
      moduleCount: Object.keys(schema.modules).length,
    });
  } catch (err) {
    res.status(500).json({
      error: 'Failed to reload schema',
      message: err instanceof Error ? err.message : 'Unknown error',
    });
  }
});

export default router;
