import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { LogsPanel } from '../components/Panels/LogsPanel';
import { useRuntimeStore } from '../store/runtimeStore';
import type { LogEntry } from '../types/runtime';

// Mock URL.createObjectURL and URL.revokeObjectURL
const mockCreateObjectURL = vi.fn(() => 'blob:mock-url');
const mockRevokeObjectURL = vi.fn();
global.URL.createObjectURL = mockCreateObjectURL;
global.URL.revokeObjectURL = mockRevokeObjectURL;

// Mock clipboard
Object.assign(navigator, {
  clipboard: { writeText: vi.fn(() => Promise.resolve()) },
});

function makeLogs(entries: Partial<LogEntry>[]): LogEntry[] {
  return entries.map((e, i) => ({
    id: e.id ?? `log-${i}`,
    timestamp: e.timestamp ?? Date.now() + i,
    level: e.level ?? 'info',
    source: e.source ?? 'pipeline',
    message: e.message ?? `Log message ${i}`,
    ...e,
  }));
}

describe('LogsPanel', () => {
  beforeEach(() => {
    useRuntimeStore.setState({
      pipelineId: null,
      status: 'IDLE',
      moduleMetrics: {},
      errors: [],
      logs: [],
      connectionState: 'disconnected',
      startTime: null,
      isLoading: false,
    });
    mockCreateObjectURL.mockClear();
    mockRevokeObjectURL.mockClear();
  });

  describe('empty state', () => {
    it('shows placeholder when no logs exist', () => {
      render(<LogsPanel />);
      expect(screen.getByText(/No logs yet/)).toBeInTheDocument();
    });

    it('shows filter mismatch message when logs exist but none match filter', () => {
      useRuntimeStore.setState({
        logs: makeLogs([{ level: 'info', message: 'Hello' }]),
      });

      render(<LogsPanel />);
      // Click error filter — no error logs exist
      fireEvent.click(screen.getByText('Error (0)'));
      expect(screen.getByText(/No logs match the current filters/)).toBeInTheDocument();
    });
  });

  describe('rendering logs', () => {
    it('renders log entries with timestamp, level, source, and message', () => {
      const ts = new Date(2026, 0, 1, 10, 30, 45, 123).getTime();
      useRuntimeStore.setState({
        logs: makeLogs([
          { timestamp: ts, level: 'info', source: 'pipeline', message: 'Pipeline created' },
        ]),
      });

      render(<LogsPanel />);
      expect(screen.getByText('10:30:45.123')).toBeInTheDocument();
      expect(screen.getByText('INF')).toBeInTheDocument();
      // 'pipeline' appears in both source dropdown and log row — use getAllByText
      expect(screen.getAllByText('pipeline').length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText('Pipeline created')).toBeInTheDocument();
    });

    it('renders multiple log entries', () => {
      useRuntimeStore.setState({
        logs: makeLogs([
          { message: 'First log' },
          { message: 'Second log' },
          { message: 'Third log' },
        ]),
      });

      render(<LogsPanel />);
      expect(screen.getByText('First log')).toBeInTheDocument();
      expect(screen.getByText('Second log')).toBeInTheDocument();
      expect(screen.getByText('Third log')).toBeInTheDocument();
    });
  });

  describe('level filtering', () => {
    beforeEach(() => {
      useRuntimeStore.setState({
        logs: makeLogs([
          { level: 'debug', message: 'Debug msg' },
          { level: 'info', message: 'Info msg' },
          { level: 'warn', message: 'Warn msg' },
          { level: 'error', message: 'Error msg' },
        ]),
      });
    });

    it('shows counts in level filter buttons', () => {
      render(<LogsPanel />);
      expect(screen.getByText('All (4)')).toBeInTheDocument();
      expect(screen.getByText('Error (1)')).toBeInTheDocument();
      expect(screen.getByText('Warn (1)')).toBeInTheDocument();
      expect(screen.getByText('Info (1)')).toBeInTheDocument();
      expect(screen.getByText('Debug (1)')).toBeInTheDocument();
    });

    it('filters to show only error logs', () => {
      render(<LogsPanel />);
      fireEvent.click(screen.getByText('Error (1)'));

      expect(screen.getByText('Error msg')).toBeInTheDocument();
      expect(screen.queryByText('Info msg')).not.toBeInTheDocument();
      expect(screen.queryByText('Warn msg')).not.toBeInTheDocument();
      expect(screen.queryByText('Debug msg')).not.toBeInTheDocument();
    });

    it('filters to show only warn logs', () => {
      render(<LogsPanel />);
      fireEvent.click(screen.getByText('Warn (1)'));

      expect(screen.getByText('Warn msg')).toBeInTheDocument();
      expect(screen.queryByText('Error msg')).not.toBeInTheDocument();
    });
  });

  describe('text search', () => {
    it('filters logs by search text', () => {
      useRuntimeStore.setState({
        logs: makeLogs([
          { message: 'Pipeline created with 3 modules' },
          { message: 'fps=30.2 qlen=2' },
          { message: 'Pipeline completed' },
        ]),
      });

      render(<LogsPanel />);
      const searchInput = screen.getByPlaceholderText('Search logs...');
      fireEvent.change(searchInput, { target: { value: 'pipeline' } });

      expect(screen.getByText('Pipeline created with 3 modules')).toBeInTheDocument();
      expect(screen.getByText('Pipeline completed')).toBeInTheDocument();
      expect(screen.queryByText('fps=30.2 qlen=2')).not.toBeInTheDocument();
    });
  });

  describe('source filtering', () => {
    it('populates source dropdown with unique sources', () => {
      useRuntimeStore.setState({
        logs: makeLogs([
          { source: 'pipeline', message: 'msg1' },
          { source: 'reader1', message: 'msg2' },
          { source: 'writer1', message: 'msg3' },
          { source: 'pipeline', message: 'msg4' },
        ]),
      });

      render(<LogsPanel />);
      const select = screen.getByDisplayValue('All sources');
      expect(select).toBeInTheDocument();

      // Check option values
      const options = select.querySelectorAll('option');
      const values = Array.from(options).map((o) => o.value);
      expect(values).toContain('all');
      expect(values).toContain('pipeline');
      expect(values).toContain('reader1');
      expect(values).toContain('writer1');
    });

    it('filters by selected source', () => {
      useRuntimeStore.setState({
        logs: makeLogs([
          { source: 'pipeline', message: 'Pipeline log' },
          { source: 'reader1', message: 'Reader log' },
        ]),
      });

      render(<LogsPanel />);
      const select = screen.getByDisplayValue('All sources');
      fireEvent.change(select, { target: { value: 'reader1' } });

      expect(screen.getByText('Reader log')).toBeInTheDocument();
      expect(screen.queryByText('Pipeline log')).not.toBeInTheDocument();
    });
  });

  describe('clear and export', () => {
    it('clears logs when clear button clicked', () => {
      useRuntimeStore.setState({
        logs: makeLogs([{ message: 'Log to clear' }]),
      });

      render(<LogsPanel />);
      expect(screen.getByText('Log to clear')).toBeInTheDocument();

      fireEvent.click(screen.getByText('Clear'));
      expect(screen.queryByText('Log to clear')).not.toBeInTheDocument();
    });

    it('exports logs when export button clicked', () => {
      useRuntimeStore.setState({
        logs: makeLogs([{ message: 'Exportable log' }]),
      });

      render(<LogsPanel />);
      fireEvent.click(screen.getByText('Export'));

      expect(mockCreateObjectURL).toHaveBeenCalled();
      expect(mockRevokeObjectURL).toHaveBeenCalled();
    });
  });
});
