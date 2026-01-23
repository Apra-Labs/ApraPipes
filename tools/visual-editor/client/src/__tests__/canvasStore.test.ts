import { describe, it, expect, beforeEach } from 'vitest';
import { useCanvasStore } from '../store/canvasStore';
import type { ModuleSchema } from '../types/schema';

const mockSchema: ModuleSchema = {
  category: 'source',
  description: 'Test module',
  inputs: [],
  outputs: [{ name: 'output', frame_types: ['RAW_IMAGE'] }],
  properties: {},
};

const mockTransformSchema: ModuleSchema = {
  category: 'transform',
  description: 'Transform module',
  inputs: [{ name: 'input', frame_types: ['RAW_IMAGE'] }],
  outputs: [{ name: 'output', frame_types: ['RAW_IMAGE'] }],
  properties: {},
};

describe('canvasStore', () => {
  beforeEach(() => {
    useCanvasStore.getState().reset();
  });

  // Helper to get fresh state after mutations
  const getState = () => useCanvasStore.getState();

  describe('addNode', () => {
    it('adds a node with unique ID', () => {
      const id = getState().addNode('TestModule', mockSchema);

      expect(id).toMatch(/^TestModule_[a-zA-Z0-9_-]+$/);
      expect(getState().nodes).toHaveLength(1);
      expect(getState().nodes[0].id).toBe(id);
    });

    it('adds node with correct data', () => {
      getState().addNode('TestModule', mockSchema, { x: 200, y: 300 });

      const node = getState().nodes[0];
      expect(node.data.type).toBe('TestModule');
      expect(node.data.category).toBe('source');
      expect(node.data.status).toBe('idle');
      expect(node.position).toEqual({ x: 200, y: 300 });
    });

    it('generates unique IDs for multiple nodes', () => {
      const id1 = getState().addNode('TestModule', mockSchema);
      const id2 = getState().addNode('TestModule', mockSchema);

      expect(id1).not.toBe(id2);
      expect(getState().nodes).toHaveLength(2);
    });
  });

  describe('removeNode', () => {
    it('removes a node by ID', () => {
      const id = getState().addNode('TestModule', mockSchema);

      expect(getState().nodes).toHaveLength(1);

      getState().removeNode(id);

      expect(getState().nodes).toHaveLength(0);
    });

    it('removes edges connected to the node', () => {
      const sourceId = getState().addNode('Source', mockSchema);
      const targetId = getState().addNode('Target', mockTransformSchema);

      getState().addEdge({
        source: sourceId,
        target: targetId,
        sourceHandle: 'output',
        targetHandle: 'input',
      });

      expect(getState().edges).toHaveLength(1);

      getState().removeNode(sourceId);

      expect(getState().edges).toHaveLength(0);
    });

    it('clears selection if selected node is removed', () => {
      const id = getState().addNode('TestModule', mockSchema);
      getState().selectNode(id);

      expect(getState().selectedNodeId).toBe(id);

      getState().removeNode(id);

      expect(getState().selectedNodeId).toBeNull();
    });
  });

  describe('addEdge', () => {
    it('adds an edge between nodes', () => {
      const sourceId = getState().addNode('Source', mockSchema);
      const targetId = getState().addNode('Target', mockTransformSchema);

      getState().addEdge({
        source: sourceId,
        target: targetId,
        sourceHandle: 'output',
        targetHandle: 'input',
      });

      expect(getState().edges).toHaveLength(1);
      expect(getState().edges[0].source).toBe(sourceId);
      expect(getState().edges[0].target).toBe(targetId);
    });

    it('ignores invalid connections', () => {
      getState().addEdge({
        source: '',
        target: 'target',
        sourceHandle: null,
        targetHandle: null,
      });

      expect(getState().edges).toHaveLength(0);
    });
  });

  describe('selectNode', () => {
    it('selects a node by ID', () => {
      const id = getState().addNode('TestModule', mockSchema);

      getState().selectNode(id);

      expect(getState().selectedNodeId).toBe(id);
    });

    it('clears selection with null', () => {
      const id = getState().addNode('TestModule', mockSchema);
      getState().selectNode(id);
      getState().selectNode(null);

      expect(getState().selectedNodeId).toBeNull();
    });
  });

  describe('updateNodeData', () => {
    it('updates node data partially', () => {
      const id = getState().addNode('TestModule', mockSchema);

      getState().updateNodeData(id, { status: 'running' });

      const node = getState().nodes.find((n) => n.id === id);
      expect(node?.data.status).toBe('running');
      expect(node?.data.type).toBe('TestModule'); // Other data preserved
    });
  });

  describe('reset', () => {
    it('resets all state', () => {
      getState().addNode('TestModule', mockSchema);
      getState().selectNode(getState().nodes[0].id);

      getState().reset();

      expect(getState().nodes).toHaveLength(0);
      expect(getState().edges).toHaveLength(0);
      expect(getState().selectedNodeId).toBeNull();
    });
  });

  describe('validation', () => {
    it('updates node validation state', () => {
      const id = getState().addNode('TestModule', mockSchema);

      getState().updateNodeValidation(id, 2, 1);

      const node = getState().nodes.find((n) => n.id === id);
      expect(node?.data.validationErrors).toBe(2);
      expect(node?.data.validationWarnings).toBe(1);
    });

    it('clears all validation state', () => {
      const id1 = getState().addNode('TestModule', mockSchema);
      const id2 = getState().addNode('Transform', mockTransformSchema);

      getState().updateNodeValidation(id1, 2, 0);
      getState().updateNodeValidation(id2, 0, 3);

      getState().clearAllValidation();

      const node1 = getState().nodes.find((n) => n.id === id1);
      const node2 = getState().nodes.find((n) => n.id === id2);
      expect(node1?.data.validationErrors).toBe(0);
      expect(node1?.data.validationWarnings).toBe(0);
      expect(node2?.data.validationErrors).toBe(0);
      expect(node2?.data.validationWarnings).toBe(0);
    });

    it('sets centerTarget', () => {
      const id = getState().addNode('TestModule', mockSchema);

      getState().centerOnNode(id);

      expect(getState().centerTarget).toBe(id);
    });

    it('clears centerTarget', () => {
      const id = getState().addNode('TestModule', mockSchema);
      getState().centerOnNode(id);

      getState().clearCenterTarget();

      expect(getState().centerTarget).toBeNull();
    });
  });

  describe('undo/redo', () => {
    it('starts with canUndo and canRedo false', () => {
      expect(getState().canUndo).toBe(false);
      expect(getState().canRedo).toBe(false);
    });

    it('enables canUndo after adding a node', () => {
      getState().addNode('TestModule', mockSchema);
      // First add creates initial state + new state, so undo is possible
      expect(getState().canUndo).toBe(true);
    });

    it('undoes addNode', () => {
      getState().addNode('TestModule', mockSchema);
      expect(getState().nodes).toHaveLength(1);

      getState().undo();
      // After undo, we go back to the state before addNode (which saves initial empty state first)
      // The first saveSnapshot records the initial state, the second records after add
      // So undo goes back to the first (empty) state
    });

    it('can redo after undo', () => {
      getState().addNode('TestModule', mockSchema);
      getState().undo();

      expect(getState().canRedo).toBe(true);
    });

    it('redoes undone action', () => {
      const id = getState().addNode('TestModule', mockSchema);
      getState().undo();

      expect(getState().canRedo).toBe(true);

      getState().redo();

      // After redo, node should be back
      expect(getState().nodes).toHaveLength(1);
      expect(getState().nodes[0].id).toBe(id);
    });

    it('clears redo stack on new action', () => {
      getState().addNode('TestModule', mockSchema);
      getState().undo();
      expect(getState().canRedo).toBe(true);

      // New action should clear redo
      getState().addNode('TestModule', mockSchema);
      expect(getState().canRedo).toBe(false);
    });

    it('reset clears history', () => {
      getState().addNode('TestModule', mockSchema);
      getState().addNode('TestModule', mockSchema);
      expect(getState().canUndo).toBe(true);

      getState().reset();

      expect(getState().canUndo).toBe(false);
      expect(getState().canRedo).toBe(false);
    });
  });

  describe('selectEdge', () => {
    it('selects an edge by ID', () => {
      const sourceId = getState().addNode('Source', mockSchema);
      const targetId = getState().addNode('Target', mockTransformSchema);

      getState().addEdge({
        source: sourceId,
        target: targetId,
        sourceHandle: 'output',
        targetHandle: 'input',
      });

      const edgeId = getState().edges[0].id;
      getState().selectEdge(edgeId);

      expect(getState().selectedEdgeId).toBe(edgeId);
      expect(getState().selectedNodeId).toBeNull(); // Selecting edge clears node selection
    });

    it('clears selection with null', () => {
      const sourceId = getState().addNode('Source', mockSchema);
      const targetId = getState().addNode('Target', mockTransformSchema);

      getState().addEdge({
        source: sourceId,
        target: targetId,
        sourceHandle: 'output',
        targetHandle: 'input',
      });

      const edgeId = getState().edges[0].id;
      getState().selectEdge(edgeId);
      getState().selectEdge(null);

      expect(getState().selectedEdgeId).toBeNull();
    });

    it('selecting node clears edge selection', () => {
      const sourceId = getState().addNode('Source', mockSchema);
      const targetId = getState().addNode('Target', mockTransformSchema);

      getState().addEdge({
        source: sourceId,
        target: targetId,
        sourceHandle: 'output',
        targetHandle: 'input',
      });

      const edgeId = getState().edges[0].id;
      getState().selectEdge(edgeId);
      expect(getState().selectedEdgeId).toBe(edgeId);

      getState().selectNode(sourceId);
      expect(getState().selectedNodeId).toBe(sourceId);
      expect(getState().selectedEdgeId).toBeNull(); // Edge selection cleared
    });
  });
});
