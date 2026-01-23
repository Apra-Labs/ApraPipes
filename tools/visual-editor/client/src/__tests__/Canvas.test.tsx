import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Canvas } from '../components/Canvas/Canvas';
import type { ModuleSchema } from '../types/schema';

// Mock React Flow as it has DOM measurement issues in jsdom
vi.mock('@xyflow/react', () => ({
  ReactFlow: ({ children }: { children?: React.ReactNode }) => (
    <div data-testid="react-flow">{children}</div>
  ),
  ReactFlowProvider: ({ children }: { children?: React.ReactNode }) => (
    <div data-testid="react-flow-provider">{children}</div>
  ),
  Background: () => <div data-testid="background" />,
  Controls: () => <div data-testid="controls" />,
  MiniMap: () => <div data-testid="minimap" />,
  Handle: () => <div data-testid="handle" />,
  Position: { Left: 'left', Right: 'right' },
  useReactFlow: () => ({
    screenToFlowPosition: vi.fn((pos) => pos),
  }),
}));

// Mock the canvas store - need to handle both direct call and selector pattern
vi.mock('../store/canvasStore', () => {
  const mockState = {
    nodes: [],
    edges: [],
    onNodesChange: () => {},
    onEdgesChange: () => {},
    onConnect: () => {},
    addNode: () => 'test-id',
    selectNode: () => {},
  };

  const useCanvasStore = (selector?: (state: typeof mockState) => unknown) => {
    if (typeof selector === 'function') {
      return selector(mockState);
    }
    return mockState;
  };

  return { useCanvasStore };
});

const mockSchema: Record<string, ModuleSchema> = {
  TestModule: {
    category: 'source',
    description: 'Test module',
    inputs: [],
    outputs: [{ name: 'output', frame_types: ['RAW_IMAGE'] }],
    properties: {},
  },
};

describe('Canvas', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders with ReactFlowProvider', () => {
    render(<Canvas schema={mockSchema} />);
    expect(screen.getByTestId('react-flow-provider')).toBeInTheDocument();
  });

  it('renders ReactFlow component', () => {
    render(<Canvas schema={mockSchema} />);
    expect(screen.getByTestId('react-flow')).toBeInTheDocument();
  });

  it('renders canvas controls', () => {
    render(<Canvas schema={mockSchema} />);
    expect(screen.getByTestId('background')).toBeInTheDocument();
    expect(screen.getByTestId('controls')).toBeInTheDocument();
    expect(screen.getByTestId('minimap')).toBeInTheDocument();
  });

  it('renders with null schema', () => {
    render(<Canvas schema={null} />);
    expect(screen.getByTestId('react-flow-provider')).toBeInTheDocument();
  });
});
