import * as fs from 'fs/promises';
import * as path from 'path';
import { logger } from '../utils/logger.js';

/**
 * WorkspaceManager handles file I/O for pipeline workspaces
 */
export class WorkspaceManager {
  private defaultDir: string;

  constructor(defaultDir?: string) {
    // Default directory for relative paths
    this.defaultDir = defaultDir || path.join(process.cwd(), 'workspaces');
    logger.info(`WorkspaceManager initialized with default directory: ${this.defaultDir}`);
  }

  /**
   * Resolve a path - supports both absolute and relative paths
   * @param userPath - The user-provided path (absolute or relative)
   * @returns The resolved absolute path
   * @throws Error if path contains directory traversal sequences
   */
  resolvePath(userPath: string): string {
    // Check for obvious directory traversal attempts
    if (userPath.includes('..')) {
      throw new Error('Path traversal detected');
    }

    // Check if path is absolute
    if (path.isAbsolute(userPath)) {
      // Use the absolute path as-is (normalized)
      return path.normalize(userPath);
    }

    // For relative paths, resolve against the default directory
    const resolved = path.resolve(this.defaultDir, userPath);

    // Verify the resolved path is still within defaultDir (for relative paths only)
    if (!resolved.startsWith(path.resolve(this.defaultDir))) {
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
    const fullPath = this.resolvePath(dirPath);

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
    const fullPath = this.resolvePath(workspacePath);

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
    const fullPath = this.resolvePath(workspacePath);
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
    const fullPath = this.resolvePath(workspacePath);
    await fs.mkdir(fullPath, { recursive: true });
    logger.info(`Created new workspace: ${fullPath}`);
  }

  /**
   * Check if a workspace exists
   * @param workspacePath - Relative path to workspace
   */
  async workspaceExists(workspacePath: string): Promise<boolean> {
    const fullPath = this.resolvePath(workspacePath);
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
    const fullPath = this.resolvePath(workspacePath);
    await fs.rm(fullPath, { recursive: true, force: true });
    logger.info(`Deleted workspace: ${fullPath}`);
  }
}

// Singleton instance
export const workspaceManager = new WorkspaceManager();
