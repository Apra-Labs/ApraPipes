import { create } from 'zustand';
import { useCanvasStore } from './canvasStore';
import { usePipelineStore } from './pipelineStore';
import type { PipelineConfig } from '../types/schema';

/**
 * Layout data for persisting node positions
 */
interface LayoutData {
  nodes: Record<string, { x: number; y: number }>;
}

/**
 * Workspace data format saved to file
 */
export interface WorkspaceFile {
  config: PipelineConfig;
  layout: LayoutData;
}

interface WorkspaceState {
  currentPath: string | null;
  isDirty: boolean;
  recentFiles: string[];
}

interface WorkspaceActions {
  // File operations
  newWorkspace: () => void;
  openWorkspace: (path: string) => Promise<void>;
  saveWorkspace: (path?: string) => Promise<void>;
  importJSON: (jsonContent: string) => void;

  // State management
  setCurrentPath: (path: string | null) => void;
  markDirty: () => void;
  markClean: () => void;
  addRecentFile: (path: string) => void;
  clearRecentFiles: () => void;
}

const RECENT_FILES_KEY = 'aprapipes-studio-recent-files';

/**
 * Load recent files from localStorage
 */
function loadRecentFiles(): string[] {
  try {
    const stored = localStorage.getItem(RECENT_FILES_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

/**
 * Save recent files to localStorage
 */
function saveRecentFiles(files: string[]): void {
  try {
    localStorage.setItem(RECENT_FILES_KEY, JSON.stringify(files));
  } catch {
    // Storage might be full or unavailable
  }
}

const initialState: WorkspaceState = {
  currentPath: null,
  isDirty: false,
  recentFiles: loadRecentFiles(),
};

const API_BASE = 'http://localhost:3000';

/**
 * Workspace store for managing file operations
 */
export const useWorkspaceStore = create<WorkspaceState & WorkspaceActions>((set, get) => ({
  ...initialState,

  newWorkspace: () => {
    // Clear canvas store
    useCanvasStore.getState().reset();
    // Clear pipeline store
    usePipelineStore.getState().reset();
    // Reset workspace state
    set({
      currentPath: null,
      isDirty: false,
    });
  },

  openWorkspace: async (path) => {
    try {
      const response = await fetch(`${API_BASE}/api/workspace/load`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path }),
      });

      if (!response.ok) {
        throw new Error(`Failed to load workspace: ${response.statusText}`);
      }

      const data: WorkspaceFile = await response.json();

      // Load pipeline config
      usePipelineStore.getState().fromJSON(JSON.stringify(data.config));

      // Restore canvas nodes from config + layout
      const canvasStore = useCanvasStore.getState();
      canvasStore.reset();

      const schema = usePipelineStore.getState().schema;

      for (const [moduleId, moduleConfig] of Object.entries(data.config.modules)) {
        const position = data.layout?.nodes?.[moduleId] || { x: 100, y: 100 };
        const moduleSchema = schema[moduleConfig.type];

        if (moduleSchema) {
          // Add node with the ORIGINAL module ID from the saved workspace
          // This ensures connections can be restored correctly
          canvasStore.addNode(moduleConfig.type, moduleSchema, position, moduleId);
        }
      }

      // Save a snapshot after restoring all nodes
      canvasStore.saveSnapshot('Load workspace');

      // Restore connections
      for (const conn of data.config.connections) {
        const [sourceId, sourceHandle] = conn.from.split('.');
        const [targetId, targetHandle] = conn.to.split('.');
        canvasStore.addEdge({
          source: sourceId,
          target: targetId,
          sourceHandle,
          targetHandle,
        });
      }

      set({
        currentPath: path,
        isDirty: false,
      });

      get().addRecentFile(path);
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error('Failed to open workspace:', error);
      throw error;
    }
  },

  saveWorkspace: async (path) => {
    const savePath = path || get().currentPath;
    if (!savePath) {
      throw new Error('No path specified for save');
    }

    const config = usePipelineStore.getState().config;
    const nodes = useCanvasStore.getState().nodes;

    // Build layout data from canvas nodes
    const layout: LayoutData = {
      nodes: {},
    };
    for (const node of nodes) {
      layout.nodes[node.id] = node.position;
    }

    const workspaceFile: WorkspaceFile = {
      config,
      layout,
    };

    try {
      const response = await fetch(`${API_BASE}/api/workspace/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          path: savePath,
          data: workspaceFile,
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to save workspace: ${response.statusText}`);
      }

      set({
        currentPath: savePath,
        isDirty: false,
      });

      get().addRecentFile(savePath);
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error('Failed to save workspace:', error);
      throw error;
    }
  },

  setCurrentPath: (path) => {
    set({ currentPath: path });
  },

  markDirty: () => {
    set({ isDirty: true });
  },

  markClean: () => {
    set({ isDirty: false });
  },

  addRecentFile: (path) => {
    set((state) => {
      const filtered = state.recentFiles.filter((f) => f !== path);
      const newRecentFiles = [path, ...filtered].slice(0, 10); // Keep max 10 recent files
      saveRecentFiles(newRecentFiles);
      return {
        recentFiles: newRecentFiles,
      };
    });
  },

  clearRecentFiles: () => {
    saveRecentFiles([]);
    set({ recentFiles: [] });
  },

  importJSON: (jsonContent) => {
    try {
      const data = JSON.parse(jsonContent) as PipelineConfig;

      // Load pipeline config
      usePipelineStore.getState().fromJSON(jsonContent);

      // Restore canvas nodes from config
      const canvasStore = useCanvasStore.getState();
      canvasStore.reset();

      const schema = usePipelineStore.getState().schema;

      // Add nodes for each module with original IDs
      let yOffset = 100;
      for (const [moduleId, moduleConfig] of Object.entries(data.modules)) {
        const moduleSchema = schema[moduleConfig.type];
        if (moduleSchema) {
          // Position nodes in a column and preserve the original module ID
          canvasStore.addNode(moduleConfig.type, moduleSchema, { x: 100, y: yOffset }, moduleId);
          yOffset += 150;
        }
      }

      // Save a snapshot after importing
      canvasStore.saveSnapshot('Import JSON');

      // Restore connections
      for (const conn of data.connections) {
        const [sourceId, sourceHandle] = conn.from.split('.');
        const [targetId, targetHandle] = conn.to.split('.');
        canvasStore.addEdge({
          source: sourceId,
          target: targetId,
          sourceHandle,
          targetHandle,
        });
      }

      set({
        currentPath: null, // No file path for imported JSON
        isDirty: true, // Mark as dirty since it's unsaved
      });
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error('Failed to import JSON:', error);
      throw error;
    }
  },
}));
