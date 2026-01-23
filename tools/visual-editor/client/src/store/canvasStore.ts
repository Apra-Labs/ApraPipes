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
}

interface CanvasActions {
  // Node operations
  addNode: (moduleType: string, schema: ModuleSchema, position?: { x: number; y: number }) => string;
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
}

const initialState: CanvasState = {
  nodes: [],
  edges: [],
  selectedNodeId: null,
  centerTarget: null,
};

export const useCanvasStore = create<CanvasState & CanvasActions>((set, get) => ({
  ...initialState,

  addNode: (moduleType, schema, position = { x: 100, y: 100 }) => {
    const id = generateNodeId(moduleType);
    const newNode: ModuleNode = {
      id,
      type: 'module',
      position,
      data: {
        type: moduleType,
        label: id.split('_').slice(0, -1).join('_'),
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

    return id;
  },

  removeNode: (id) => {
    set((state) => ({
      nodes: state.nodes.filter((n) => n.id !== id),
      edges: state.edges.filter((e) => e.source !== id && e.target !== id),
      selectedNodeId: state.selectedNodeId === id ? null : state.selectedNodeId,
    }));
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
  },

  removeEdge: (id) => {
    set((state) => ({
      edges: state.edges.filter((e) => e.id !== id),
    }));
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
    set(initialState);
  },
}));
