import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { BottomPanel } from '../components/Panels/BottomPanel';
import { usePipelineStore } from '../store/pipelineStore';
import { useRuntimeStore } from '../store/runtimeStore';

// Mock URL.createObjectURL and URL.revokeObjectURL (needed by child panels)
global.URL.createObjectURL = vi.fn(() => 'blob:mock-url');
global.URL.revokeObjectURL = vi.fn();

describe('BottomPanel', () => {
  beforeEach(() => {
    usePipelineStore.setState({
      validationResult: null,
      isValidating: false,
    });
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
  });

  describe('tab bar', () => {
    it('renders Problems and Logs tabs', () => {
      render(<BottomPanel />);
      expect(screen.getByText('Problems')).toBeInTheDocument();
      expect(screen.getByText('Logs')).toBeInTheDocument();
    });

    it('shows Problems tab active by default', () => {
      render(<BottomPanel />);
      // ProblemsPanel renders its filter bar when active
      expect(screen.getByText('All (0)')).toBeInTheDocument();
    });

    it('shows badge counts for problems', () => {
      usePipelineStore.setState({
        validationResult: {
          valid: false,
          issues: [
            { level: 'error', code: 'E101', message: 'err', location: 'modules.a' },
            { level: 'warning', code: 'W201', message: 'warn', location: 'modules.b' },
          ],
        },
      });
      useRuntimeStore.setState({
        errors: [{ moduleId: 'a', message: 'runtime err', timestamp: Date.now() }],
      });

      render(<BottomPanel />);
      // 2 validation + 1 runtime = 3 total
      expect(screen.getByText('3')).toBeInTheDocument();
    });
  });

  describe('tab switching', () => {
    it('switches to Logs panel when Logs tab clicked', () => {
      useRuntimeStore.setState({
        logs: [
          { id: 'l1', timestamp: Date.now(), level: 'info', source: 'pipeline', message: 'Test log' },
        ],
      });

      render(<BottomPanel />);

      // Click Logs tab
      fireEvent.click(screen.getByText('Logs'));

      // LogsPanel content should be visible
      expect(screen.getByText('Test log')).toBeInTheDocument();
    });

    it('switches back to Problems panel', () => {
      render(<BottomPanel />);

      // Switch to Logs
      fireEvent.click(screen.getByText('Logs'));
      // LogsPanel empty state
      expect(screen.getByText(/No logs yet/)).toBeInTheDocument();

      // Switch back to Problems
      fireEvent.click(screen.getByText('Problems'));
      // ProblemsPanel content
      expect(screen.getByText(/Click "Validate"/)).toBeInTheDocument();
    });
  });

  describe('collapse/expand', () => {
    it('collapses when collapse button clicked', () => {
      render(<BottomPanel />);

      // Click collapse
      fireEvent.click(screen.getByLabelText('Collapse panel'));

      // Should still show tab labels in collapsed bar and expand button
      expect(screen.getByText('Problems')).toBeInTheDocument();
      expect(screen.getByText('Logs')).toBeInTheDocument();
      expect(screen.getByLabelText('Expand panel')).toBeInTheDocument();

      // ProblemsPanel filter bar should NOT be visible
      expect(screen.queryByText('All (0)')).not.toBeInTheDocument();
    });

    it('expands when collapsed tab is clicked', () => {
      render(<BottomPanel />);

      // Collapse
      fireEvent.click(screen.getByLabelText('Collapse panel'));

      // Click on the Logs tab in collapsed bar
      fireEvent.click(screen.getByText('Logs'));

      // Should expand and show LogsPanel
      expect(screen.getByText(/No logs yet/)).toBeInTheDocument();
    });
  });
});
