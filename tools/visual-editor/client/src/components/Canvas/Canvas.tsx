import { useCallback, DragEvent, useMemo } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  ReactFlowProvider,
  useReactFlow,
  NodeTypes,
  Edge,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { useCanvasStore, ModuleNodeData } from '../../store/canvasStore';
import { usePipelineStore } from '../../store/pipelineStore';
import { ModuleNode } from './ModuleNode';
import type { ModuleSchema } from '../../types/schema';

interface CanvasProps {
  schema: Record<string, ModuleSchema> | null;
}

function CanvasInner({ schema }: CanvasProps) {
  const { screenToFlowPosition } = useReactFlow();
  const {
    nodes,
    edges,
    selectedEdgeId,
    onNodesChange,
    onEdgesChange,
    onConnect: canvasOnConnect,
    addNode,
    selectNode,
    selectEdge,
  } = useCanvasStore();

  const addModuleToPipeline = usePipelineStore((state) => state.addModule);
  const addConnectionToPipeline = usePipelineStore((state) => state.addConnection);

  // Register custom node types - must be memoized
  const nodeTypes: NodeTypes = useMemo(
    () => ({
      module: ModuleNode,
    }),
    []
  );

  const handleDragOver = useCallback((event: DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'copy';
  }, []);

  const handleDrop = useCallback(
    (event: DragEvent) => {
      event.preventDefault();

      const moduleType = event.dataTransfer.getData('application/aprapipes-module');
      if (!moduleType || !schema) return;

      const moduleSchema = schema[moduleType];
      if (!moduleSchema) return;

      // Get drop position in flow coordinates
      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      // Add node to canvas store and get the generated ID
      const nodeId = addNode(moduleType, moduleSchema, position);
      // Also add to pipeline store for serialization
      addModuleToPipeline(nodeId, moduleType);
    },
    [schema, addNode, addModuleToPipeline, screenToFlowPosition]
  );

  const handleConnect = useCallback(
    (connection: { source: string; target: string; sourceHandle: string | null; targetHandle: string | null }) => {
      // Add edge to canvas store
      canvasOnConnect(connection);

      // Add connection to pipeline store in the format: "moduleId.pinName"
      if (connection.sourceHandle && connection.targetHandle) {
        const from = `${connection.source}.${connection.sourceHandle}`;
        const to = `${connection.target}.${connection.targetHandle}`;
        addConnectionToPipeline(from, to);
      }
    },
    [canvasOnConnect, addConnectionToPipeline]
  );

  const handleNodeClick = useCallback(
    (_: React.MouseEvent, node: { id: string }) => {
      selectNode(node.id);
    },
    [selectNode]
  );

  const handleEdgeClick = useCallback(
    (_: React.MouseEvent, edge: Edge) => {
      selectEdge(edge.id);
    },
    [selectEdge]
  );

  const handlePaneClick = useCallback(() => {
    selectNode(null);
    selectEdge(null);
  }, [selectNode, selectEdge]);

  // Apply selection styling to edges
  const styledEdges = useMemo(() => {
    return edges.map((edge) => ({
      ...edge,
      selected: edge.id === selectedEdgeId,
      style: {
        strokeWidth: edge.id === selectedEdgeId ? 3 : 2,
        stroke: edge.id === selectedEdgeId ? '#3b82f6' : '#888',
      },
    }));
  }, [edges, selectedEdgeId]);

  return (
    <div className="w-full h-full" onDragOver={handleDragOver} onDrop={handleDrop}>
      <ReactFlow
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        nodes={nodes as any}
        edges={styledEdges}
        nodeTypes={nodeTypes}
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        onNodesChange={onNodesChange as any}
        onEdgesChange={onEdgesChange}
        onConnect={handleConnect}
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        onNodeClick={handleNodeClick as any}
        onEdgeClick={handleEdgeClick}
        onPaneClick={handlePaneClick}
        fitView
        snapToGrid
        snapGrid={[15, 15]}
        defaultEdgeOptions={{
          animated: false,
          style: { strokeWidth: 2 },
        }}
      >
        <Background gap={15} size={1} />
        <Controls />
        <MiniMap
          nodeColor={(node) => {
            const data = node.data as ModuleNodeData;
            switch (data?.category) {
              case 'source':
                return '#3b82f6';
              case 'transform':
                return '#22c55e';
              case 'sink':
                return '#ef4444';
              default:
                return '#6b7280';
            }
          }}
        />
      </ReactFlow>
    </div>
  );
}

export function Canvas({ schema }: CanvasProps) {
  return (
    <ReactFlowProvider>
      <CanvasInner schema={schema} />
    </ReactFlowProvider>
  );
}
