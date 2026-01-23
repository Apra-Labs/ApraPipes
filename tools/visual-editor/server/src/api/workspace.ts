import { Router, Request, Response } from 'express';
import { workspaceManager } from '../services/WorkspaceManager.js';
import { logger } from '../utils/logger.js';

export const workspaceRouter = Router();

/**
 * GET /api/workspace/list
 * List files in a workspace directory
 */
workspaceRouter.get('/list', async (req: Request, res: Response) => {
  try {
    const dirPath = (req.query.path as string) || '';
    const files = await workspaceManager.listFiles(dirPath);
    res.json({ files });
  } catch (error) {
    logger.error('Failed to list workspace files:', error);
    if (error instanceof Error && error.message === 'Path traversal detected') {
      res.status(400).json({ error: 'Invalid path' });
    } else {
      res.status(500).json({ error: 'Failed to list files' });
    }
  }
});

/**
 * POST /api/workspace/save
 * Save workspace data to files
 */
workspaceRouter.post('/save', async (req: Request, res: Response) => {
  try {
    const { path: workspacePath, data } = req.body;

    if (!workspacePath) {
      res.status(400).json({ error: 'Path is required' });
      return;
    }

    if (!data) {
      res.status(400).json({ error: 'Data is required' });
      return;
    }

    await workspaceManager.saveWorkspace(workspacePath, data);
    res.json({ success: true, path: workspacePath });
  } catch (error) {
    logger.error('Failed to save workspace:', error);
    if (error instanceof Error && error.message === 'Path traversal detected') {
      res.status(400).json({ error: 'Invalid path' });
    } else {
      res.status(500).json({ error: 'Failed to save workspace' });
    }
  }
});

/**
 * POST /api/workspace/load
 * Load workspace data from files
 */
workspaceRouter.post('/load', async (req: Request, res: Response) => {
  try {
    const { path: workspacePath } = req.body;

    if (!workspacePath) {
      res.status(400).json({ error: 'Path is required' });
      return;
    }

    const data = await workspaceManager.loadWorkspace(workspacePath);
    res.json(data);
  } catch (error) {
    logger.error('Failed to load workspace:', error);
    if (error instanceof Error) {
      if (error.message === 'Path traversal detected') {
        res.status(400).json({ error: 'Invalid path' });
      } else if (error.message.startsWith('Workspace not found')) {
        res.status(404).json({ error: error.message });
      } else if (error.message.startsWith('Invalid JSON')) {
        res.status(400).json({ error: error.message });
      } else {
        res.status(500).json({ error: 'Failed to load workspace' });
      }
    } else {
      res.status(500).json({ error: 'Failed to load workspace' });
    }
  }
});

/**
 * POST /api/workspace/create
 * Create a new workspace directory
 */
workspaceRouter.post('/create', async (req: Request, res: Response) => {
  try {
    const { path: workspacePath } = req.body;

    if (!workspacePath) {
      res.status(400).json({ error: 'Path is required' });
      return;
    }

    // Check if workspace already exists
    const exists = await workspaceManager.workspaceExists(workspacePath);
    if (exists) {
      res.status(409).json({ error: 'Workspace already exists' });
      return;
    }

    await workspaceManager.createWorkspace(workspacePath);
    res.json({ success: true, path: workspacePath });
  } catch (error) {
    logger.error('Failed to create workspace:', error);
    if (error instanceof Error && error.message === 'Path traversal detected') {
      res.status(400).json({ error: 'Invalid path' });
    } else {
      res.status(500).json({ error: 'Failed to create workspace' });
    }
  }
});

/**
 * DELETE /api/workspace
 * Delete a workspace
 */
workspaceRouter.delete('/', async (req: Request, res: Response) => {
  try {
    const { path: workspacePath } = req.body;

    if (!workspacePath) {
      res.status(400).json({ error: 'Path is required' });
      return;
    }

    await workspaceManager.deleteWorkspace(workspacePath);
    res.json({ success: true });
  } catch (error) {
    logger.error('Failed to delete workspace:', error);
    if (error instanceof Error && error.message === 'Path traversal detected') {
      res.status(400).json({ error: 'Invalid path' });
    } else {
      res.status(500).json({ error: 'Failed to delete workspace' });
    }
  }
});
