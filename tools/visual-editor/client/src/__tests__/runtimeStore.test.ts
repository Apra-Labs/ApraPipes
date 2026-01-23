import { describe, it, expect, beforeEach, vi } from 'vitest';
import { useRuntimeStore } from '../store/runtimeStore';
import type { HealthMessage, ErrorMessage, StatusMessage } from '../types/runtime';

// Mock fetch
const mockFetch = vi.fn();
global.fetch = mockFetch;

// Mock WebSocket client
vi.mock('../services/websocket', () => ({
  getWebSocketClient: vi.fn(() => ({
    setHandlers: vi.fn(),
    connect: vi.fn(),
    disconnect: vi.fn(),
    subscribe: vi.fn(),
    unsubscribe: vi.fn(),
  })),
}));

describe('runtimeStore', () => {
  beforeEach(() => {
    useRuntimeStore.getState().reset();
    mockFetch.mockReset();
  });

  // Helper to get fresh state
  const getState = () => useRuntimeStore.getState();

  describe('initial state', () => {
    it('has correct initial values', () => {
      expect(getState().pipelineId).toBeNull();
      expect(getState().status).toBe('IDLE');
      expect(getState().moduleMetrics).toEqual({});
      expect(getState().errors).toEqual([]);
      expect(getState().connectionState).toBe('disconnected');
      expect(getState().startTime).toBeNull();
      expect(getState().isLoading).toBe(false);
    });
  });

  describe('createPipeline', () => {
    it('creates a pipeline and sets state', async () => {
      const mockConfig = {
        modules: { source: { type: 'Test' } },
        connections: [],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ pipelineId: 'test-123' }),
      });

      const id = await getState().createPipeline(mockConfig);

      expect(id).toBe('test-123');
      expect(getState().pipelineId).toBe('test-123');
      expect(getState().status).toBe('IDLE');
      expect(getState().moduleMetrics.source).toBeDefined();
    });

    it('throws error on failure', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        statusText: 'Bad Request',
      });

      await expect(getState().createPipeline({})).rejects.toThrow('Failed to create pipeline');
    });
  });

  describe('startPipeline', () => {
    it('starts pipeline and updates status', async () => {
      // Set up state with a pipeline
      useRuntimeStore.setState({ pipelineId: 'test-123', status: 'IDLE' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, status: 'RUNNING' }),
      });

      await getState().startPipeline();

      expect(getState().status).toBe('RUNNING');
      expect(getState().startTime).not.toBeNull();
    });

    it('throws error if no pipeline', async () => {
      await expect(getState().startPipeline()).rejects.toThrow('No pipeline created');
    });
  });

  describe('stopPipeline', () => {
    it('stops pipeline and updates status', async () => {
      useRuntimeStore.setState({
        pipelineId: 'test-123',
        status: 'RUNNING',
        startTime: Date.now(),
      });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, status: 'STOPPED' }),
      });

      await getState().stopPipeline();

      expect(getState().status).toBe('STOPPED');
    });

    it('throws error if no pipeline', async () => {
      await expect(getState().stopPipeline()).rejects.toThrow('No pipeline created');
    });
  });

  describe('deletePipeline', () => {
    it('deletes pipeline and resets state', async () => {
      useRuntimeStore.setState({
        pipelineId: 'test-123',
        status: 'STOPPED',
        moduleMetrics: { source: { fps: 30, qlen: 5, isQueueFull: false, timestamp: 0 } },
      });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true }),
      });

      await getState().deletePipeline();

      expect(getState().pipelineId).toBeNull();
      expect(getState().status).toBe('IDLE');
      expect(getState().moduleMetrics).toEqual({});
    });
  });

  describe('onHealthEvent', () => {
    it('updates module metrics', () => {
      useRuntimeStore.setState({ pipelineId: 'test-123' });

      const message: HealthMessage = {
        event: 'health',
        pipelineId: 'test-123',
        data: {
          moduleId: 'source',
          fps: 30,
          qlen: 5,
          isQueueFull: false,
        },
      };

      getState().onHealthEvent(message);

      expect(getState().moduleMetrics.source).toBeDefined();
      expect(getState().moduleMetrics.source.fps).toBe(30);
      expect(getState().moduleMetrics.source.qlen).toBe(5);
    });

    it('ignores events for other pipelines', () => {
      useRuntimeStore.setState({ pipelineId: 'test-123' });

      const message: HealthMessage = {
        event: 'health',
        pipelineId: 'other-pipeline',
        data: {
          moduleId: 'source',
          fps: 30,
          qlen: 5,
          isQueueFull: false,
        },
      };

      getState().onHealthEvent(message);

      expect(getState().moduleMetrics.source).toBeUndefined();
    });
  });

  describe('onErrorEvent', () => {
    it('adds runtime error', () => {
      useRuntimeStore.setState({ pipelineId: 'test-123' });

      const message: ErrorMessage = {
        event: 'error',
        pipelineId: 'test-123',
        data: {
          moduleId: 'source',
          message: 'Test error',
          code: 'E001',
        },
      };

      getState().onErrorEvent(message);

      expect(getState().errors).toHaveLength(1);
      expect(getState().errors[0].moduleId).toBe('source');
      expect(getState().errors[0].message).toBe('Test error');
    });

    it('ignores events for other pipelines', () => {
      useRuntimeStore.setState({ pipelineId: 'test-123' });

      const message: ErrorMessage = {
        event: 'error',
        pipelineId: 'other-pipeline',
        data: {
          moduleId: 'source',
          message: 'Test error',
        },
      };

      getState().onErrorEvent(message);

      expect(getState().errors).toHaveLength(0);
    });
  });

  describe('onStatusEvent', () => {
    it('updates status', () => {
      useRuntimeStore.setState({ pipelineId: 'test-123', status: 'IDLE' });

      const message: StatusMessage = {
        event: 'status',
        pipelineId: 'test-123',
        data: { status: 'RUNNING' },
      };

      getState().onStatusEvent(message);

      expect(getState().status).toBe('RUNNING');
    });

    it('sets startTime when status changes to RUNNING', () => {
      useRuntimeStore.setState({
        pipelineId: 'test-123',
        status: 'IDLE',
        startTime: null,
      });

      const message: StatusMessage = {
        event: 'status',
        pipelineId: 'test-123',
        data: { status: 'RUNNING' },
      };

      getState().onStatusEvent(message);

      expect(getState().startTime).not.toBeNull();
    });
  });

  describe('clearErrors', () => {
    it('clears all errors', () => {
      useRuntimeStore.setState({
        errors: [
          { moduleId: 'source', message: 'Error 1', timestamp: 0 },
          { moduleId: 'sink', message: 'Error 2', timestamp: 1 },
        ],
      });

      getState().clearErrors();

      expect(getState().errors).toHaveLength(0);
    });
  });

  describe('reset', () => {
    it('resets all state to initial values', () => {
      useRuntimeStore.setState({
        pipelineId: 'test-123',
        status: 'RUNNING',
        moduleMetrics: { source: { fps: 30, qlen: 5, isQueueFull: false, timestamp: 0 } },
        errors: [{ moduleId: 'source', message: 'Error', timestamp: 0 }],
        startTime: Date.now(),
      });

      getState().reset();

      expect(getState().pipelineId).toBeNull();
      expect(getState().status).toBe('IDLE');
      expect(getState().moduleMetrics).toEqual({});
      expect(getState().errors).toEqual([]);
      expect(getState().startTime).toBeNull();
    });
  });

  describe('getDuration', () => {
    it('returns null when not running', () => {
      useRuntimeStore.setState({ status: 'IDLE', startTime: null });

      expect(getState().getDuration()).toBeNull();
    });

    it('returns duration when running', () => {
      const startTime = Date.now() - 5000; // 5 seconds ago
      useRuntimeStore.setState({ status: 'RUNNING', startTime });

      const duration = getState().getDuration();
      expect(duration).toBeGreaterThanOrEqual(4900);
      expect(duration).toBeLessThanOrEqual(6000);
    });
  });
});
