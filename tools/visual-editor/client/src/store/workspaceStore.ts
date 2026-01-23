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

  // State management
  setCurrentPath: (path: string | null) => void;
  markDirty: () => void;
  markClean: () => void;
  addRecentFile: (path: string) => void;
}

const initialState: WorkspaceState = {
  currentPath: null,
  isDirty: false,
  recentFiles: [],
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
          // Add node using existing addNode (generates new ID if needed)
          // For simplicity, we'll update position after
          canvasStore.addNode(moduleConfig.type, moduleSchema, position);
        }
      }

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
      return {
        recentFiles: [path, ...filtered].slice(0, 10), // Keep max 10 recent files
      };
    });
  },
}));
