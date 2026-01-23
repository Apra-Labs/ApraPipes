import * as fs from 'fs/promises';
import * as path from 'path';
import { logger } from '../utils/logger.js';

/**
 * WorkspaceManager handles file I/O for pipeline workspaces
 * with path sanitization to prevent directory traversal attacks
 */
export class WorkspaceManager {
  private baseDir: string;

  constructor(baseDir?: string) {
    // Default to a 'workspaces' folder relative to server
    this.baseDir = baseDir || path.join(process.cwd(), 'workspaces');
    logger.info(`WorkspaceManager initialized with base directory: ${this.baseDir}`);
  }

  /**
   * Sanitize a path to prevent directory traversal
   * @param userPath - The user-provided path
   * @returns The sanitized absolute path within baseDir
   * @throws Error if path escapes baseDir
   */
  sanitizePath(userPath: string): string {
    // Remove any leading slashes and normalize
    const cleaned = userPath.replace(/^[/\\]+/, '');

    // Join with base and resolve to absolute
    const resolved = path.resolve(this.baseDir, cleaned);

    // Verify the resolved path is still within baseDir
    if (!resolved.startsWith(path.resolve(this.baseDir))) {
      throw new Error('Path traversal detected');
    }

    return resolved;
  }

  /**
   * List files in a workspace directory
   * @param dirPath - Relative path to directory
   * @returns Array of file info objects
   */
  async listFiles(dirPath: string = ''): Promise<Array<{ name: string; isDirectory: boolean }>> {
    const fullPath = this.sanitizePath(dirPath);

    try {
      const entries = await fs.readdir(fullPath, { withFileTypes: true });
      return entries.map((entry) => ({
        name: entry.name,
        isDirectory: entry.isDirectory(),
      }));
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
        // Directory doesn't exist, create it and return empty
        await fs.mkdir(fullPath, { recursive: true });
        return [];
      }
      throw error;
    }
  }

  /**
   * Save workspace data to files
   * @param workspacePath - Relative path to workspace folder
   * @param data - The workspace data to save
   */
  async saveWorkspace(workspacePath: string, data: unknown): Promise<void> {
    const fullPath = this.sanitizePath(workspacePath);

    // Ensure directory exists
    await fs.mkdir(fullPath, { recursive: true });

    // Write the workspace file as JSON
    const filePath = path.join(fullPath, 'pipeline.json');
    await fs.writeFile(filePath, JSON.stringify(data, null, 2), 'utf-8');

    logger.info(`Workspace saved to: ${filePath}`);
  }

  /**
   * Load workspace data from files
   * @param workspacePath - Relative path to workspace folder
   * @returns The loaded workspace data
   */
  async loadWorkspace(workspacePath: string): Promise<unknown> {
    const fullPath = this.sanitizePath(workspacePath);
    const filePath = path.join(fullPath, 'pipeline.json');

    try {
      const content = await fs.readFile(filePath, 'utf-8');
      return JSON.parse(content);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
        throw new Error(`Workspace not found: ${workspacePath}`);
      }
      if (error instanceof SyntaxError) {
        throw new Error(`Invalid JSON in workspace: ${workspacePath}`);
      }
      throw error;
    }
  }

  /**
   * Create a new workspace directory
   * @param workspacePath - Relative path for new workspace
   */
  async createWorkspace(workspacePath: string): Promise<void> {
    const fullPath = this.sanitizePath(workspacePath);
    await fs.mkdir(fullPath, { recursive: true });
    logger.info(`Created new workspace: ${fullPath}`);
  }

  /**
   * Check if a workspace exists
   * @param workspacePath - Relative path to workspace
   */
  async workspaceExists(workspacePath: string): Promise<boolean> {
    const fullPath = this.sanitizePath(workspacePath);
    const filePath = path.join(fullPath, 'pipeline.json');

    try {
      await fs.access(filePath);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Delete a workspace
   * @param workspacePath - Relative path to workspace
   */
  async deleteWorkspace(workspacePath: string): Promise<void> {
    const fullPath = this.sanitizePath(workspacePath);
    await fs.rm(fullPath, { recursive: true, force: true });
    logger.info(`Deleted workspace: ${fullPath}`);
  }
}

// Singleton instance
export const workspaceManager = new WorkspaceManager();
