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
});
