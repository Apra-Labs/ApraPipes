import { describe, it, expect, beforeEach, vi } from 'vitest';
import { useWorkspaceStore } from '../store/workspaceStore';
import { useCanvasStore } from '../store/canvasStore';
import { usePipelineStore } from '../store/pipelineStore';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('workspaceStore', () => {
  beforeEach(() => {
    // Reset all stores
    useWorkspaceStore.setState({
      currentPath: null,
      isDirty: false,
      recentFiles: [],
    });
    useCanvasStore.getState().reset();
    usePipelineStore.getState().reset();
    mockFetch.mockReset();
  });

  describe('initial state', () => {
    it('has null currentPath', () => {
      expect(useWorkspaceStore.getState().currentPath).toBeNull();
    });

    it('is not dirty initially', () => {
      expect(useWorkspaceStore.getState().isDirty).toBe(false);
    });

    it('has empty recent files', () => {
      expect(useWorkspaceStore.getState().recentFiles).toEqual([]);
    });
  });

  describe('setCurrentPath', () => {
    it('sets the current path', () => {
      useWorkspaceStore.getState().setCurrentPath('/test/path');
      expect(useWorkspaceStore.getState().currentPath).toBe('/test/path');
    });

    it('can set path to null', () => {
      useWorkspaceStore.getState().setCurrentPath('/test/path');
      useWorkspaceStore.getState().setCurrentPath(null);
      expect(useWorkspaceStore.getState().currentPath).toBeNull();
    });
  });

  describe('markDirty', () => {
    it('sets isDirty to true', () => {
      useWorkspaceStore.getState().markDirty();
      expect(useWorkspaceStore.getState().isDirty).toBe(true);
    });
  });

  describe('markClean', () => {
    it('sets isDirty to false', () => {
      useWorkspaceStore.getState().markDirty();
      useWorkspaceStore.getState().markClean();
      expect(useWorkspaceStore.getState().isDirty).toBe(false);
    });
  });

  describe('addRecentFile', () => {
    it('adds file to recent files', () => {
      useWorkspaceStore.getState().addRecentFile('/test/file1');
      expect(useWorkspaceStore.getState().recentFiles).toEqual(['/test/file1']);
    });

    it('moves existing file to front', () => {
      useWorkspaceStore.getState().addRecentFile('/test/file1');
      useWorkspaceStore.getState().addRecentFile('/test/file2');
      useWorkspaceStore.getState().addRecentFile('/test/file1');
      expect(useWorkspaceStore.getState().recentFiles).toEqual([
        '/test/file1',
        '/test/file2',
      ]);
    });

    it('keeps maximum 10 recent files', () => {
      for (let i = 0; i < 15; i++) {
        useWorkspaceStore.getState().addRecentFile(`/test/file${i}`);
      }
      expect(useWorkspaceStore.getState().recentFiles).toHaveLength(10);
      expect(useWorkspaceStore.getState().recentFiles[0]).toBe('/test/file14');
    });
  });

  describe('newWorkspace', () => {
    it('resets all stores', () => {
      // Set up some state first
      useWorkspaceStore.getState().setCurrentPath('/test/path');
      useWorkspaceStore.getState().markDirty();

      // Reset
      useWorkspaceStore.getState().newWorkspace();

      expect(useWorkspaceStore.getState().currentPath).toBeNull();
      expect(useWorkspaceStore.getState().isDirty).toBe(false);
    });
  });

  describe('saveWorkspace', () => {
    it('throws error if no path specified and no current path', async () => {
      await expect(useWorkspaceStore.getState().saveWorkspace()).rejects.toThrow(
        'No path specified for save'
      );
    });

    it('saves to specified path', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({}),
      });

      await useWorkspaceStore.getState().saveWorkspace('/test/project');

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:3000/api/workspace/save',
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
        })
      );
      expect(useWorkspaceStore.getState().currentPath).toBe('/test/project');
      expect(useWorkspaceStore.getState().isDirty).toBe(false);
    });

    it('saves to current path if not specified', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({}),
      });

      useWorkspaceStore.getState().setCurrentPath('/existing/path');
      await useWorkspaceStore.getState().saveWorkspace();

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:3000/api/workspace/save',
        expect.objectContaining({
          body: expect.stringContaining('/existing/path'),
        })
      );
    });

    it('throws on failed save', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        statusText: 'Server Error',
      });

      await expect(
        useWorkspaceStore.getState().saveWorkspace('/test/project')
      ).rejects.toThrow('Failed to save workspace: Server Error');
    });

    it('adds path to recent files on save', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({}),
      });

      await useWorkspaceStore.getState().saveWorkspace('/test/project');

      expect(useWorkspaceStore.getState().recentFiles).toContain('/test/project');
    });
  });

  describe('openWorkspace', () => {
    it('loads workspace from API', async () => {
      const mockData = {
        config: {
          modules: {},
          connections: [],
        },
        layout: {
          nodes: {},
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockData,
      });

      await useWorkspaceStore.getState().openWorkspace('/test/project');

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:3000/api/workspace/load',
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: '/test/project' }),
        })
      );
      expect(useWorkspaceStore.getState().currentPath).toBe('/test/project');
      expect(useWorkspaceStore.getState().isDirty).toBe(false);
    });

    it('throws on failed load', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        statusText: 'Not Found',
      });

      await expect(
        useWorkspaceStore.getState().openWorkspace('/test/project')
      ).rejects.toThrow('Failed to load workspace: Not Found');
    });

    it('adds path to recent files on open', async () => {
      const mockData = {
        config: {
          modules: {},
          connections: [],
        },
        layout: {
          nodes: {},
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockData,
      });

      await useWorkspaceStore.getState().openWorkspace('/test/project');

      expect(useWorkspaceStore.getState().recentFiles).toContain('/test/project');
    });
  });
});
