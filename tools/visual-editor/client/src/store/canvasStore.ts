import { create } from 'zustand';
import {
  Edge,
  Connection,
  applyNodeChanges,
  applyEdgeChanges,
  NodeChange,
  EdgeChange,
} from '@xyflow/react';
import { generateNodeId } from '../utils/id';
import { HistoryManager } from '../utils/history';
import type { ModuleSchema } from '../types/schema';

export interface ModuleNodeData {
  type: string;
  label: string;
  category: string;
  description?: string;
  inputs: Array<{ name: string; frame_types: string[] }>;
  outputs: Array<{ name: string; frame_types: string[] }>;
  properties: Record<string, unknown>;
  status: 'idle' | 'running' | 'error';
  metrics?: {
    fps: number;
    qlen: number;
    isQueueFull: boolean;
  };
  // Validation state
  validationErrors?: number;
  validationWarnings?: number;
  [key: string]: unknown; // Allow index signature for React Flow compatibility
}

export interface ModuleNode {
  id: string;
  type: string;
  position: { x: number; y: number };
  data: ModuleNodeData;
}

interface CanvasState {
  nodes: ModuleNode[];
  edges: Edge[];
  selectedNodeId: string | null;
  centerTarget: string | null; // Node ID to center on
  // History state
  canUndo: boolean;
  canRedo: boolean;
}

/**
 * Snapshot of canvas state for history
 */
interface CanvasSnapshot {
  nodes: ModuleNode[];
  edges: Edge[];
}

interface CanvasActions {
  // Node operations
  addNode: (moduleType: string, schema: ModuleSchema, position?: { x: number; y: number }, id?: string) => string;
  removeNode: (id: string) => void;
  updateNodePosition: (id: string, position: { x: number; y: number }) => void;
  updateNodeData: (id: string, data: Partial<ModuleNodeData>) => void;

  // Edge operations
  addEdge: (connection: Connection) => void;
  removeEdge: (id: string) => void;

  // Validation
  updateNodeValidation: (nodeId: string, errors: number, warnings: number) => void;
  clearAllValidation: () => void;

  // Selection
  selectNode: (id: string | null) => void;

  // Navigation
  centerOnNode: (id: string) => void;
  clearCenterTarget: () => void;

  // React Flow callbacks
  onNodesChange: (changes: NodeChange[]) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  onConnect: (connection: Connection) => void;

  // Reset
  reset: () => void;

  // History (Undo/Redo)
  undo: () => void;
  redo: () => void;
  saveSnapshot: (action?: string) => void;
}

const initialState: CanvasState = {
  nodes: [],
  edges: [],
  selectedNodeId: null,
  centerTarget: null,
  canUndo: false,
  canRedo: false,
};

// Create history manager singleton
const historyManager = new HistoryManager<CanvasSnapshot>({
  maxSize: 50,
  storageKey: 'aprapipes-studio-history',
});

export const useCanvasStore = create<CanvasState & CanvasActions>((set, get) => ({
  ...initialState,

  addNode: (moduleType, schema, position = { x: 100, y: 100 }, id?: string) => {
    // Use provided id or generate a new one
    const nodeId = id || generateNodeId(moduleType);
    // For label, use the provided id if available, otherwise extract from generated id
    const label = id || nodeId.split('_').slice(0, -1).join('_');

    const newNode: ModuleNode = {
      id: nodeId,
      type: 'module',
      position,
      data: {
        type: moduleType,
        label,
        category: schema.category,
        description: schema.description,
        inputs: schema.inputs,
        outputs: schema.outputs,
        properties: {},
        status: 'idle',
      },
    };

    set((state) => ({
      nodes: [...state.nodes, newNode],
    }));

    // Save snapshot after change (but not when restoring workspace - that's done separately)
    if (!id) {
      get().saveSnapshot(`Add ${moduleType}`);
    }

    return nodeId;
  },

  removeNode: (id) => {
    set((state) => ({
      nodes: state.nodes.filter((n) => n.id !== id),
      edges: state.edges.filter((e) => e.source !== id && e.target !== id),
      selectedNodeId: state.selectedNodeId === id ? null : state.selectedNodeId,
    }));
    get().saveSnapshot('Delete node');
  },

  updateNodePosition: (id, position) => {
    set((state) => ({
      nodes: state.nodes.map((n) =>
        n.id === id ? { ...n, position } : n
      ),
    }));
  },

  updateNodeData: (id, data) => {
    set((state) => ({
      nodes: state.nodes.map((n) =>
        n.id === id ? { ...n, data: { ...n.data, ...data } } : n
      ),
    }));
  },

  addEdge: (connection) => {
    if (!connection.source || !connection.target) return;

    const newEdge: Edge = {
      id: `${connection.source}-${connection.sourceHandle || 'default'}-${connection.target}-${connection.targetHandle || 'default'}`,
      source: connection.source,
      target: connection.target,
      sourceHandle: connection.sourceHandle,
      targetHandle: connection.targetHandle,
    };

    set((state) => ({
      edges: [...state.edges, newEdge],
    }));
    get().saveSnapshot('Connect modules');
  },

  removeEdge: (id) => {
    set((state) => ({
      edges: state.edges.filter((e) => e.id !== id),
    }));
    get().saveSnapshot('Delete connection');
  },

  updateNodeValidation: (nodeId, errors, warnings) => {
    set((state) => ({
      nodes: state.nodes.map((n) =>
        n.id === nodeId
          ? {
              ...n,
              data: {
                ...n.data,
                validationErrors: errors,
                validationWarnings: warnings,
              },
            }
          : n
      ),
    }));
  },

  clearAllValidation: () => {
    set((state) => ({
      nodes: state.nodes.map((n) => ({
        ...n,
        data: {
          ...n.data,
          validationErrors: 0,
          validationWarnings: 0,
        },
      })),
    }));
  },

  selectNode: (id) => {
    set({ selectedNodeId: id });
  },

  centerOnNode: (id) => {
    set({ centerTarget: id });
  },

  clearCenterTarget: () => {
    set({ centerTarget: null });
  },

  onNodesChange: (changes) => {
    set((state) => ({
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      nodes: applyNodeChanges(changes, state.nodes as any) as unknown as ModuleNode[],
    }));
  },

  onEdgesChange: (changes) => {
    set((state) => ({
      edges: applyEdgeChanges(changes, state.edges),
    }));
  },

  onConnect: (connection) => {
    get().addEdge(connection);
  },

  reset: () => {
    historyManager.clear();
    set(initialState);
  },

  undo: () => {
    const snapshot = historyManager.undo();
    if (snapshot) {
      set({
        nodes: snapshot.nodes,
        edges: snapshot.edges,
        canUndo: historyManager.canUndo(),
        canRedo: historyManager.canRedo(),
      });
    }
  },

  redo: () => {
    const snapshot = historyManager.redo();
    if (snapshot) {
      set({
        nodes: snapshot.nodes,
        edges: snapshot.edges,
        canUndo: historyManager.canUndo(),
        canRedo: historyManager.canRedo(),
      });
    }
  },

  saveSnapshot: (action?: string) => {
    const { nodes, edges } = get();

    // If this is the first snapshot, push an initial empty state first
    // so that undo can go back to the initial state
    if (historyManager.getCurrentState() === null) {
      historyManager.push({ nodes: [], edges: [] }, 'Initial');
    }

    historyManager.push({ nodes, edges }, action);
    set({
      canUndo: historyManager.canUndo(),
      canRedo: historyManager.canRedo(),
    });
  },
}));
