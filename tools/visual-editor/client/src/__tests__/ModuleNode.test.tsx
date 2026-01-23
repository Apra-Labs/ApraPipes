import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ModuleNode } from '../components/Canvas/ModuleNode';
import type { ModuleNodeData } from '../store/canvasStore';
import { useRuntimeStore } from '../store/runtimeStore';
import { useCanvasStore } from '../store/canvasStore';

// Mock React Flow Handle component
vi.mock('@xyflow/react', () => ({
  Handle: ({ id, type }: { id: string; type: string }) => (
    <div data-testid={`handle-${type}-${id}`} />
  ),
  Position: { Left: 'left', Right: 'right' },
  useNodeId: () => 'test-node-id',
}));

// Reset stores before each test
beforeEach(() => {
  useRuntimeStore.setState({
    status: 'IDLE',
    moduleMetrics: {},
  });
  useCanvasStore.setState({
    selectedPin: null,
  });
});

const createMockData = (overrides: Partial<ModuleNodeData> = {}): ModuleNodeData => ({
  type: 'TestModule',
  label: 'Test Module',
  category: 'source',
  description: 'A test module',
  inputs: [],
  outputs: [{ name: 'output', frame_types: ['RAW_IMAGE'] }],
  properties: {},
  status: 'idle',
  ...overrides,
});

describe('ModuleNode', () => {
  it('renders module with correct label', () => {
    const data = createMockData({ label: 'My Module' });
    render(<ModuleNode data={data} />);
    expect(screen.getByText('My Module')).toBeInTheDocument();
  });

  it('renders category in title attribute (compact design)', () => {
    const data = createMockData({ category: 'transform', type: 'TestModule', label: 'Test Label' });
    render(<ModuleNode data={data} />);
    // Category is now shown only in the title attribute for compact design
    expect(screen.getByTitle('TestModule (transform)')).toBeInTheDocument();
  });

  it('renders output handles', () => {
    const data = createMockData({
      outputs: [
        { name: 'video', frame_types: ['RAW_IMAGE'] },
        { name: 'audio', frame_types: ['RAW_AUDIO'] },
      ],
    });
    render(<ModuleNode data={data} />);
    expect(screen.getByTestId('handle-source-video')).toBeInTheDocument();
    expect(screen.getByTestId('handle-source-audio')).toBeInTheDocument();
  });

  it('renders input handles', () => {
    const data = createMockData({
      inputs: [{ name: 'input', frame_types: ['RAW_IMAGE'] }],
    });
    render(<ModuleNode data={data} />);
    expect(screen.getByTestId('handle-target-input')).toBeInTheDocument();
  });

  it('shows error indicator when status is error', () => {
    const data = createMockData({ status: 'error' });
    render(<ModuleNode data={data} />);
    // StatusBadge shows !! text when error and no validation errors
    expect(screen.getByText('!!')).toBeInTheDocument();
  });

  it('does not show error indicator when status is idle', () => {
    const data = createMockData({ status: 'idle' });
    render(<ModuleNode data={data} />);
    expect(screen.queryByText('!!')).not.toBeInTheDocument();
  });

  it('shows metrics when runtime status is RUNNING', () => {
    // Set runtime store to RUNNING with metrics
    useRuntimeStore.setState({
      status: 'RUNNING',
      moduleMetrics: {
        'Test Module': { fps: 30, qlen: 5, isQueueFull: false, timestamp: Date.now() },
      },
    });

    const data = createMockData({
      status: 'idle', // data.status doesn't matter when runtime is RUNNING
    });
    render(<ModuleNode data={data} />);
    // fps is displayed with .toFixed(1)
    expect(screen.getByText('30.0')).toBeInTheDocument();
    expect(screen.getByText('5')).toBeInTheDocument();
  });

  it('does not show metrics when runtime status is IDLE', () => {
    // Runtime store defaults to IDLE in beforeEach
    const data = createMockData({
      status: 'idle',
      metrics: { fps: 30, qlen: 5, isQueueFull: false },
    });
    render(<ModuleNode data={data} />);
    // Metrics section is not shown when not running
    expect(screen.queryByText('fps')).not.toBeInTheDocument();
  });

  it('applies selected styling when selected', () => {
    const data = createMockData();
    const { container } = render(<ModuleNode data={data} selected={true} />);
    expect(container.firstChild).toHaveClass('ring-2');
  });

  it('shows placeholder when no pins defined', () => {
    const data = createMockData({ inputs: [], outputs: [] });
    render(<ModuleNode data={data} />);
    expect(screen.getByText('No pins defined')).toBeInTheDocument();
  });

  it('renders pin labels', () => {
    const data = createMockData({
      inputs: [{ name: 'video_in', frame_types: ['RAW_IMAGE'] }],
      outputs: [{ name: 'video_out', frame_types: ['RAW_IMAGE'] }],
    });
    render(<ModuleNode data={data} />);
    expect(screen.getByText('video_in')).toBeInTheDocument();
    expect(screen.getByText('video_out')).toBeInTheDocument();
  });
});
