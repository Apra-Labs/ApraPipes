import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ProblemsPanel } from '../components/Panels/ProblemsPanel';
import { usePipelineStore } from '../store/pipelineStore';
import { useCanvasStore } from '../store/canvasStore';
import { useRuntimeStore } from '../store/runtimeStore';

// Mock URL.createObjectURL and URL.revokeObjectURL
const mockCreateObjectURL = vi.fn(() => 'blob:mock-url');
const mockRevokeObjectURL = vi.fn();
global.URL.createObjectURL = mockCreateObjectURL;
global.URL.revokeObjectURL = mockRevokeObjectURL;

describe('ProblemsPanel', () => {
  beforeEach(() => {
    // Reset all stores
    usePipelineStore.setState({
      validationResult: null,
      isValidating: false,
    });
    useCanvasStore.setState({
      nodes: [],
      edges: [],
      selectedNodeId: null,
    });
    useRuntimeStore.setState({
      status: 'IDLE',
      errors: [],
      pipelineId: null,
      moduleMetrics: {},
      connectionState: 'disconnected',
      startTime: null,
      isLoading: false,
    });

    // Reset mocks
    mockCreateObjectURL.mockClear();
    mockRevokeObjectURL.mockClear();
  });

  describe('validation issues', () => {
    it('shows validation placeholder when no validation result', () => {
      render(<ProblemsPanel />);
      expect(screen.getByText('Click "Validate" to check your pipeline')).toBeInTheDocument();
    });

    it('shows validation issues', () => {
      usePipelineStore.setState({
        validationResult: {
          valid: false,
          issues: [
            {
              level: 'error',
              code: 'E101',
              message: 'Test error message',
              location: 'modules.test',
            },
          ],
        },
      });

      render(<ProblemsPanel />);
      expect(screen.getByText('E101')).toBeInTheDocument();
      expect(screen.getByText('Test error message')).toBeInTheDocument();
    });

    it('displays no issues message when valid', () => {
      usePipelineStore.setState({
        validationResult: {
          valid: true,
          issues: [],
        },
      });

      render(<ProblemsPanel />);
      expect(screen.getByText(/No issues found/)).toBeInTheDocument();
    });
  });

  describe('runtime errors', () => {
    it('displays runtime errors in the list', () => {
      useRuntimeStore.setState({
        errors: [
          {
            moduleId: 'test-module',
            message: 'Runtime error occurred',
            timestamp: Date.now(),
            code: 'R001',
          },
        ],
      });

      render(<ProblemsPanel />);
      expect(screen.getByText('Runtime error occurred')).toBeInTheDocument();
      expect(screen.getByText('runtime')).toBeInTheDocument(); // Runtime badge
    });

    it('shows runtime count in filter button', () => {
      useRuntimeStore.setState({
        errors: [
          {
            moduleId: 'test1',
            message: 'Error 1',
            timestamp: Date.now(),
          },
          {
            moduleId: 'test2',
            message: 'Error 2',
            timestamp: Date.now(),
          },
        ],
      });

      render(<ProblemsPanel />);
      expect(screen.getByText('Runtime (2)')).toBeInTheDocument();
    });

    it('filters to show only runtime errors', () => {
      usePipelineStore.setState({
        validationResult: {
          valid: false,
          issues: [
            {
              level: 'error',
              code: 'E101',
              message: 'Validation error',
              location: 'modules.test',
            },
          ],
        },
      });
      useRuntimeStore.setState({
        errors: [
          {
            moduleId: 'test-module',
            message: 'Runtime error',
            timestamp: Date.now(),
          },
        ],
      });

      render(<ProblemsPanel />);

      // Both issues visible initially
      expect(screen.getByText('Validation error')).toBeInTheDocument();
      expect(screen.getByText('Runtime error')).toBeInTheDocument();

      // Click runtime filter
      fireEvent.click(screen.getByText('Runtime (1)'));

      // Only runtime error visible
      expect(screen.queryByText('Validation error')).not.toBeInTheDocument();
      expect(screen.getByText('Runtime error')).toBeInTheDocument();
    });

    it('shows clear button when runtime errors exist', () => {
      useRuntimeStore.setState({
        errors: [
          {
            moduleId: 'test',
            message: 'Error',
            timestamp: Date.now(),
          },
        ],
      });

      render(<ProblemsPanel />);
      expect(screen.getByText('Clear')).toBeInTheDocument();
    });

    it('clears runtime errors when clear button clicked', () => {
      useRuntimeStore.setState({
        errors: [
          {
            moduleId: 'test',
            message: 'Error to clear',
            timestamp: Date.now(),
          },
        ],
      });

      render(<ProblemsPanel />);
      expect(screen.getByText('Error to clear')).toBeInTheDocument();

      fireEvent.click(screen.getByText('Clear'));

      expect(screen.queryByText('Error to clear')).not.toBeInTheDocument();
    });
  });

  describe('export logs', () => {
    it('shows export button when there are issues', () => {
      usePipelineStore.setState({
        validationResult: {
          valid: false,
          issues: [
            {
              level: 'warning',
              code: 'W201',
              message: 'Warning',
              location: 'modules.test',
            },
          ],
        },
      });

      render(<ProblemsPanel />);
      expect(screen.getByText('Export')).toBeInTheDocument();
    });

    it('calls createObjectURL when export button clicked', () => {
      usePipelineStore.setState({
        validationResult: {
          valid: false,
          issues: [
            {
              level: 'error',
              code: 'E101',
              message: 'Export test error',
              location: 'modules.test',
            },
          ],
        },
      });

      render(<ProblemsPanel />);
      fireEvent.click(screen.getByText('Export'));

      // Verify that URL.createObjectURL was called for the blob download
      expect(mockCreateObjectURL).toHaveBeenCalled();
      expect(mockRevokeObjectURL).toHaveBeenCalled();
    });
  });

  describe('collapsed state', () => {
    it('can be collapsed', () => {
      useRuntimeStore.setState({
        errors: [
          {
            moduleId: 'test',
            message: 'Error',
            timestamp: Date.now(),
          },
        ],
      });

      render(<ProblemsPanel />);

      // Panel is expanded initially, collapse button is visible
      expect(screen.getByLabelText('Collapse panel')).toBeInTheDocument();
    });
  });

  describe('issue click handling', () => {
    it('clicking issue calls selectNode', () => {
      const selectNode = vi.fn();
      const centerOnNode = vi.fn();

      // Get current state and merge in our mocks
      const currentState = useCanvasStore.getState();
      useCanvasStore.setState({
        ...currentState,
        selectNode,
        centerOnNode,
      });

      useRuntimeStore.setState({
        errors: [
          {
            moduleId: 'test-module',
            message: 'Clickable error',
            timestamp: Date.now(),
          },
        ],
      });

      render(<ProblemsPanel />);
      fireEvent.click(screen.getByText('Clickable error'));

      expect(selectNode).toHaveBeenCalledWith('test-module');
      expect(centerOnNode).toHaveBeenCalledWith('test-module');
    });
  });
});
