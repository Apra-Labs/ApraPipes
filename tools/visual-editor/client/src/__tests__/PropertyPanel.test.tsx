import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { PropertyPanel } from '../components/Panels/PropertyPanel';
import type { ModuleSchema } from '../types/schema';

// Mock the stores
const mockCanvasStore = {
  selectedNodeId: null as string | null,
  selectedPin: null as { nodeId: string; pinName: string; pinType: 'input' | 'output' } | null,
  nodes: [] as Array<{
    id: string;
    data: {
      type: string;
      label: string;
      category: string;
      description?: string;
      inputs: Array<{ name: string; frame_types: string[] }>;
      outputs: Array<{ name: string; frame_types: string[] }>;
    };
  }>,
  edges: [] as Array<{ id: string; source: string; target: string; sourceHandle?: string; targetHandle?: string }>,
  updateNodeData: vi.fn(),
  removeNode: vi.fn(),
  centerOnNode: vi.fn(),
  selectNode: vi.fn(),
};

const mockPipelineStore = {
  config: {
    modules: {} as Record<string, { type: string; properties: Record<string, unknown> }>,
    connections: [],
  },
  updateModuleProperty: vi.fn(),
  renameModule: vi.fn(),
};

vi.mock('../store/canvasStore', () => ({
  useCanvasStore: (selector?: (state: typeof mockCanvasStore) => unknown) => {
    if (typeof selector === 'function') {
      return selector(mockCanvasStore);
    }
    return mockCanvasStore;
  },
}));

vi.mock('../store/pipelineStore', () => ({
  usePipelineStore: (selector?: (state: typeof mockPipelineStore) => unknown) => {
    if (typeof selector === 'function') {
      return selector(mockPipelineStore);
    }
    return mockPipelineStore;
  },
}));

const mockSchema: Record<string, ModuleSchema> = {
  TestSource: {
    category: 'source',
    description: 'A test source module',
    inputs: [],
    outputs: [{ name: 'output', frame_types: ['RAW_IMAGE'] }],
    properties: {
      width: { type: 'int', default: '640', min: '1', max: '4096', description: 'Frame width' },
      enabled: { type: 'bool', default: 'true' },
    },
  },
};

describe('PropertyPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockCanvasStore.selectedNodeId = null;
    mockCanvasStore.selectedPin = null;
    mockCanvasStore.nodes = [];
    mockCanvasStore.edges = [];
    mockPipelineStore.config.modules = {};
  });

  it('shows placeholder when no node selected', () => {
    render(<PropertyPanel schema={mockSchema} />);
    expect(screen.getByText('Select a module or pin to view its properties')).toBeInTheDocument();
  });

  it('shows module info when node is selected', () => {
    mockCanvasStore.selectedNodeId = 'src1';
    mockCanvasStore.nodes = [
      {
        id: 'src1',
        data: {
          type: 'TestSource',
          label: 'My Source',
          category: 'source',
          description: 'A test source module',
          inputs: [],
          outputs: [{ name: 'output', frame_types: ['RAW_IMAGE'] }],
        },
      },
    ];
    mockPipelineStore.config.modules = {
      src1: { type: 'TestSource', properties: { width: 640, enabled: true } },
    };

    render(<PropertyPanel schema={mockSchema} />);

    expect(screen.getByText('My Source')).toBeInTheDocument();
    expect(screen.getByText('TestSource')).toBeInTheDocument();
    expect(screen.getByText('source')).toBeInTheDocument();
    expect(screen.getByText('A test source module')).toBeInTheDocument();
  });

  it('renders property editors for module properties', () => {
    mockCanvasStore.selectedNodeId = 'src1';
    mockCanvasStore.nodes = [
      {
        id: 'src1',
        data: {
          type: 'TestSource',
          label: 'My Source',
          category: 'source',
          inputs: [],
          outputs: [{ name: 'output', frame_types: ['RAW_IMAGE'] }],
        },
      },
    ];
    mockPipelineStore.config.modules = {
      src1: { type: 'TestSource', properties: { width: 640, enabled: true } },
    };

    render(<PropertyPanel schema={mockSchema} />);

    // Should show the Configuration section
    expect(screen.getByText('Configuration')).toBeInTheDocument();
    // Should have property editors
    expect(screen.getByText('width')).toBeInTheDocument();
    expect(screen.getByText('enabled')).toBeInTheDocument();
  });

  it('shows module ID', () => {
    mockCanvasStore.selectedNodeId = 'src1';
    mockCanvasStore.nodes = [
      {
        id: 'src1',
        data: {
          type: 'TestSource',
          label: 'My Source',
          category: 'source',
          inputs: [],
          outputs: [{ name: 'output', frame_types: ['RAW_IMAGE'] }],
        },
      },
    ];

    render(<PropertyPanel schema={mockSchema} />);

    expect(screen.getByText('src1')).toBeInTheDocument();
  });

  it('handles null schema gracefully', () => {
    mockCanvasStore.selectedNodeId = 'src1';
    mockCanvasStore.nodes = [
      {
        id: 'src1',
        data: {
          type: 'TestSource',
          label: 'My Source',
          category: 'source',
          inputs: [],
          outputs: [{ name: 'output', frame_types: ['RAW_IMAGE'] }],
        },
      },
    ];

    render(<PropertyPanel schema={null} />);

    // Should still show basic info
    expect(screen.getByText('My Source')).toBeInTheDocument();
    expect(screen.getByText('TestSource')).toBeInTheDocument();
  });

  it('shows pin properties when pin is selected', () => {
    mockCanvasStore.selectedNodeId = 'src1';
    mockCanvasStore.selectedPin = { nodeId: 'src1', pinName: 'output', pinType: 'output' };
    mockCanvasStore.nodes = [
      {
        id: 'src1',
        data: {
          type: 'TestSource',
          label: 'My Source',
          category: 'source',
          inputs: [],
          outputs: [{ name: 'output', frame_types: ['RAW_IMAGE'] }],
        },
      },
    ];

    render(<PropertyPanel schema={mockSchema} />);

    // Should show pin properties header
    expect(screen.getByText('Pin Properties')).toBeInTheDocument();
    // Should show pin name
    expect(screen.getByText('output')).toBeInTheDocument();
    // Should show direction
    expect(screen.getByText('Output (Source)')).toBeInTheDocument();
    // Should show frame types
    expect(screen.getByText('RAW_IMAGE')).toBeInTheDocument();
    // Should show parent module
    expect(screen.getByText('My Source')).toBeInTheDocument();
  });
});
