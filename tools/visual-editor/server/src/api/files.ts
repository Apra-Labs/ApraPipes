/**
 * File Browser API
 *
 * Provides endpoints for browsing the server's file system.
 * Used for file/path property selection and workspace open/save dialogs.
 */

import { Router } from 'express';
import fs from 'fs/promises';
import path from 'path';
import os from 'os';
import { logger } from '../utils/logger.js';

const router = Router();

interface FileEntry {
  name: string;
  path: string;
  type: 'file' | 'directory';
  size?: number;
  modified?: string;
}

interface ListResponse {
  path: string;
  parent: string | null;
  entries: FileEntry[];
}

/**
 * GET /api/files/list
 * List directory contents
 * Query params:
 *   - path: Directory path to list (default: home directory)
 *   - filter: File extension filter (e.g., ".json", ".jpg")
 */
router.get('/list', async (req, res) => {
  try {
    const requestedPath = (req.query.path as string) || os.homedir();
    const filter = req.query.filter as string | undefined;

    // Resolve and normalize the path
    const resolvedPath = path.resolve(requestedPath);

    // Check if path exists and is a directory
    const stat = await fs.stat(resolvedPath);
    if (!stat.isDirectory()) {
      return res.status(400).json({
        error: 'Not a directory',
        path: resolvedPath,
      });
    }

    // Read directory contents
    const entries = await fs.readdir(resolvedPath, { withFileTypes: true });

    // Build response
    const fileEntries: FileEntry[] = [];

    for (const entry of entries) {
      // Skip hidden files (starting with .)
      if (entry.name.startsWith('.')) continue;

      const entryPath = path.join(resolvedPath, entry.name);
      const isDirectory = entry.isDirectory();

      // Apply filter for files
      if (!isDirectory && filter) {
        const ext = path.extname(entry.name).toLowerCase();
        if (ext !== filter.toLowerCase()) continue;
      }

      try {
        const entryStat = await fs.stat(entryPath);
        fileEntries.push({
          name: entry.name,
          path: entryPath,
          type: isDirectory ? 'directory' : 'file',
          size: isDirectory ? undefined : entryStat.size,
          modified: entryStat.mtime.toISOString(),
        });
      } catch {
        // Skip entries we can't stat (permission issues)
      }
    }

    // Sort: directories first, then files, alphabetically
    fileEntries.sort((a, b) => {
      if (a.type !== b.type) {
        return a.type === 'directory' ? -1 : 1;
      }
      return a.name.localeCompare(b.name);
    });

    // Calculate parent directory
    const parent = resolvedPath !== path.parse(resolvedPath).root
      ? path.dirname(resolvedPath)
      : null;

    const response: ListResponse = {
      path: resolvedPath,
      parent,
      entries: fileEntries,
    };

    res.json(response);
  } catch (error) {
    logger.error('File list error:', error);
    res.status(500).json({
      error: error instanceof Error ? error.message : 'Failed to list directory',
    });
  }
});

/**
 * GET /api/files/home
 * Get home directory path
 */
router.get('/home', (_req, res) => {
  res.json({ path: os.homedir() });
});

/**
 * GET /api/files/exists
 * Check if a path exists
 * Query params:
 *   - path: Path to check
 */
router.get('/exists', async (req, res) => {
  try {
    const requestedPath = req.query.path as string;
    if (!requestedPath) {
      return res.status(400).json({ error: 'Path required' });
    }

    const resolvedPath = path.resolve(requestedPath);

    try {
      const stat = await fs.stat(resolvedPath);
      res.json({
        exists: true,
        path: resolvedPath,
        type: stat.isDirectory() ? 'directory' : 'file',
      });
    } catch {
      res.json({
        exists: false,
        path: resolvedPath,
      });
    }
  } catch (error) {
    res.status(500).json({
      error: error instanceof Error ? error.message : 'Check failed',
    });
  }
});

export default router;
