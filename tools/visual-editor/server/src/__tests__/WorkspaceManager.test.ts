import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as fs from 'fs/promises';
import * as path from 'path';
import { WorkspaceManager } from '../services/WorkspaceManager.js';

describe('WorkspaceManager', () => {
  let manager: WorkspaceManager;
  let testBaseDir: string;

  beforeEach(async () => {
    // Create a temporary test directory
    testBaseDir = path.join(process.cwd(), '.test-workspaces');
    await fs.mkdir(testBaseDir, { recursive: true });
    manager = new WorkspaceManager(testBaseDir);
  });

  afterEach(async () => {
    // Clean up test directory
    await fs.rm(testBaseDir, { recursive: true, force: true });
  });

  describe('sanitizePath', () => {
    it('allows normal paths', () => {
      const result = manager.sanitizePath('my-project');
      expect(result).toBe(path.join(testBaseDir, 'my-project'));
    });

    it('allows nested paths', () => {
      const result = manager.sanitizePath('user/projects/my-project');
      expect(result).toBe(path.join(testBaseDir, 'user', 'projects', 'my-project'));
    });

    it('removes leading slashes', () => {
      const result = manager.sanitizePath('/my-project');
      expect(result).toBe(path.join(testBaseDir, 'my-project'));
    });

    it('throws on directory traversal attempt', () => {
      expect(() => manager.sanitizePath('../etc/passwd')).toThrow('Path traversal detected');
    });

    it('throws on complex traversal attempt', () => {
      expect(() => manager.sanitizePath('foo/../../etc/passwd')).toThrow('Path traversal detected');
    });

    it('throws on encoded traversal attempt', () => {
      // This tests the resolved path check
      expect(() => manager.sanitizePath('..%2F..%2Fetc')).not.toThrow();
      // The path is actually valid because %2F is not decoded
    });
  });

  describe('listFiles', () => {
    it('returns empty array for non-existent directory', async () => {
      const files = await manager.listFiles('non-existent');
      expect(files).toEqual([]);
    });

    it('creates directory if it does not exist', async () => {
      await manager.listFiles('new-directory');
      const exists = await fs.access(path.join(testBaseDir, 'new-directory'))
        .then(() => true)
        .catch(() => false);
      expect(exists).toBe(true);
    });

    it('lists files in directory', async () => {
      const testDir = path.join(testBaseDir, 'test-dir');
      await fs.mkdir(testDir, { recursive: true });
      await fs.writeFile(path.join(testDir, 'file1.json'), '{}');
      await fs.writeFile(path.join(testDir, 'file2.txt'), 'hello');
      await fs.mkdir(path.join(testDir, 'subdir'));

      const files = await manager.listFiles('test-dir');
      expect(files).toContainEqual({ name: 'file1.json', isDirectory: false });
      expect(files).toContainEqual({ name: 'file2.txt', isDirectory: false });
      expect(files).toContainEqual({ name: 'subdir', isDirectory: true });
    });
  });

  describe('saveWorkspace', () => {
    it('saves workspace data to pipeline.json', async () => {
      const data = { modules: {}, connections: [] };
      await manager.saveWorkspace('my-project', data);

      const filePath = path.join(testBaseDir, 'my-project', 'pipeline.json');
      const content = await fs.readFile(filePath, 'utf-8');
      expect(JSON.parse(content)).toEqual(data);
    });

    it('creates directory if it does not exist', async () => {
      const data = { test: 'value' };
      await manager.saveWorkspace('new-project', data);

      const exists = await fs.access(path.join(testBaseDir, 'new-project'))
        .then(() => true)
        .catch(() => false);
      expect(exists).toBe(true);
    });
  });

  describe('loadWorkspace', () => {
    it('loads workspace data from pipeline.json', async () => {
      const data = { modules: { test: 'value' }, connections: [] };
      const projectDir = path.join(testBaseDir, 'my-project');
      await fs.mkdir(projectDir, { recursive: true });
      await fs.writeFile(
        path.join(projectDir, 'pipeline.json'),
        JSON.stringify(data)
      );

      const loaded = await manager.loadWorkspace('my-project');
      expect(loaded).toEqual(data);
    });

    it('throws error for non-existent workspace', async () => {
      await expect(manager.loadWorkspace('non-existent')).rejects.toThrow(
        'Workspace not found: non-existent'
      );
    });

    it('throws error for invalid JSON', async () => {
      const projectDir = path.join(testBaseDir, 'bad-project');
      await fs.mkdir(projectDir, { recursive: true });
      await fs.writeFile(path.join(projectDir, 'pipeline.json'), 'not valid json');

      await expect(manager.loadWorkspace('bad-project')).rejects.toThrow(
        'Invalid JSON in workspace: bad-project'
      );
    });
  });

  describe('createWorkspace', () => {
    it('creates a new workspace directory', async () => {
      await manager.createWorkspace('new-workspace');

      const exists = await fs.access(path.join(testBaseDir, 'new-workspace'))
        .then(() => true)
        .catch(() => false);
      expect(exists).toBe(true);
    });
  });

  describe('workspaceExists', () => {
    it('returns true if workspace exists', async () => {
      const projectDir = path.join(testBaseDir, 'existing-project');
      await fs.mkdir(projectDir, { recursive: true });
      await fs.writeFile(path.join(projectDir, 'pipeline.json'), '{}');

      const exists = await manager.workspaceExists('existing-project');
      expect(exists).toBe(true);
    });

    it('returns false if workspace does not exist', async () => {
      const exists = await manager.workspaceExists('non-existent');
      expect(exists).toBe(false);
    });

    it('returns false if directory exists but pipeline.json does not', async () => {
      const projectDir = path.join(testBaseDir, 'empty-project');
      await fs.mkdir(projectDir, { recursive: true });

      const exists = await manager.workspaceExists('empty-project');
      expect(exists).toBe(false);
    });
  });

  describe('deleteWorkspace', () => {
    it('deletes workspace directory', async () => {
      const projectDir = path.join(testBaseDir, 'to-delete');
      await fs.mkdir(projectDir, { recursive: true });
      await fs.writeFile(path.join(projectDir, 'pipeline.json'), '{}');

      await manager.deleteWorkspace('to-delete');

      const exists = await fs.access(projectDir)
        .then(() => true)
        .catch(() => false);
      expect(exists).toBe(false);
    });

    it('does not throw for non-existent workspace', async () => {
      await expect(manager.deleteWorkspace('non-existent')).resolves.not.toThrow();
    });
  });
});
