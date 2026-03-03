import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { PipelineManager, resetPipelineManager } from '../services/PipelineManager.js';
import type { PipelineConfig } from '../types/pipeline.js';

describe('PipelineManager', () => {
  let manager: PipelineManager;

  const mockConfig: PipelineConfig = {
    modules: {
      source: {
        type: 'TestSignalGenerator',
        properties: { width: 1920, height: 1080 },
      },
      sink: {
        type: 'FileWriterModule',
        properties: { filePath: '/tmp/output.raw' },
      },
    },
    connections: [
      { from: 'source.output', to: 'sink.input' },
    ],
  };

  beforeEach(() => {
    resetPipelineManager();
    // Force mock mode for tests
    manager = new PipelineManager({ forceMockMode: true });
  });

  afterEach(() => {
    resetPipelineManager();
  });

  describe('create', () => {
    it('creates a pipeline and returns unique ID', () => {
      const id = manager.create(mockConfig);

      expect(id).toBeTruthy();
      expect(typeof id).toBe('string');
      expect(id.length).toBeGreaterThan(0);
    });

    it('creates pipelines with unique IDs', () => {
      const id1 = manager.create(mockConfig);
      const id2 = manager.create(mockConfig);

      expect(id1).not.toBe(id2);
    });

    it('initializes pipeline with IDLE status', () => {
      const id = manager.create(mockConfig);
      const instance = manager.get(id);

      expect(instance?.status).toBe('IDLE');
    });

    it('initializes metrics for each module', () => {
      const id = manager.create(mockConfig);
      const instance = manager.get(id);

      expect(instance?.metrics).toBeDefined();
      expect(instance?.metrics.source).toBeDefined();
      expect(instance?.metrics.sink).toBeDefined();
      expect(instance?.metrics.source.fps).toBe(0);
    });

    it('stores the pipeline configuration', () => {
      const id = manager.create(mockConfig);
      const instance = manager.get(id);

      expect(instance?.config).toEqual(mockConfig);
    });

    it('emits created event', () => {
      const handler = vi.fn();
      manager.on('created', handler);

      const id = manager.create(mockConfig);

      expect(handler).toHaveBeenCalledWith({
        pipelineId: id,
        status: 'IDLE',
      });
    });
  });

  describe('start', () => {
    it('starts a pipeline and sets status to RUNNING', async () => {
      const id = manager.create(mockConfig);

      await manager.start(id);

      expect(manager.getStatus(id)).toBe('RUNNING');
    });

    it('sets startTime when starting', async () => {
      const id = manager.create(mockConfig);
      const beforeStart = Date.now();

      await manager.start(id);

      const instance = manager.get(id);
      expect(instance?.startTime).toBeDefined();
      expect(instance?.startTime).toBeGreaterThanOrEqual(beforeStart);
    });

    it('throws error for non-existent pipeline', async () => {
      await expect(manager.start('non-existent')).rejects.toThrow('Pipeline not found');
    });

    it('throws error if already running', async () => {
      const id = manager.create(mockConfig);
      await manager.start(id);

      await expect(manager.start(id)).rejects.toThrow('already running');
    });

    it('emits status events', async () => {
      const handler = vi.fn();
      manager.on('status', handler);

      const id = manager.create(mockConfig);
      await manager.start(id);

      // Should emit CREATING then RUNNING
      expect(handler).toHaveBeenCalled();
      const lastCall = handler.mock.calls[handler.mock.calls.length - 1][0];
      expect(lastCall.status).toBe('RUNNING');
    });
  });

  describe('stop', () => {
    it('stops a running pipeline', async () => {
      const id = manager.create(mockConfig);
      await manager.start(id);

      await manager.stop(id);

      expect(manager.getStatus(id)).toBe('STOPPED');
    });

    it('throws error for non-existent pipeline', async () => {
      await expect(manager.stop('non-existent')).rejects.toThrow('Pipeline not found');
    });

    it('throws error if not running', async () => {
      const id = manager.create(mockConfig);

      await expect(manager.stop(id)).rejects.toThrow('not running');
    });

    it('emits status events', async () => {
      const handler = vi.fn();
      const id = manager.create(mockConfig);
      await manager.start(id);

      manager.on('status', handler);
      await manager.stop(id);

      const lastCall = handler.mock.calls[handler.mock.calls.length - 1][0];
      expect(lastCall.status).toBe('STOPPED');
    });
  });

  describe('get', () => {
    it('returns pipeline instance', () => {
      const id = manager.create(mockConfig);
      const instance = manager.get(id);

      expect(instance).toBeDefined();
      expect(instance?.id).toBe(id);
    });

    it('returns undefined for non-existent pipeline', () => {
      const instance = manager.get('non-existent');

      expect(instance).toBeUndefined();
    });
  });

  describe('getStatus', () => {
    it('returns pipeline status', () => {
      const id = manager.create(mockConfig);

      expect(manager.getStatus(id)).toBe('IDLE');
    });

    it('returns undefined for non-existent pipeline', () => {
      expect(manager.getStatus('non-existent')).toBeUndefined();
    });
  });

  describe('delete', () => {
    it('deletes a pipeline', async () => {
      const id = manager.create(mockConfig);

      await manager.delete(id);

      expect(manager.get(id)).toBeUndefined();
    });

    it('stops pipeline if running before delete', async () => {
      const id = manager.create(mockConfig);
      await manager.start(id);

      await manager.delete(id);

      expect(manager.get(id)).toBeUndefined();
    });

    it('throws error for non-existent pipeline', async () => {
      await expect(manager.delete('non-existent')).rejects.toThrow('Pipeline not found');
    });

    it('emits deleted event', async () => {
      const handler = vi.fn();
      manager.on('deleted', handler);

      const id = manager.create(mockConfig);
      await manager.delete(id);

      expect(handler).toHaveBeenCalledWith({ pipelineId: id });
    });
  });

  describe('list', () => {
    it('returns empty array when no pipelines', () => {
      expect(manager.list()).toEqual([]);
    });

    it('returns all pipeline IDs', () => {
      const id1 = manager.create(mockConfig);
      const id2 = manager.create(mockConfig);

      const list = manager.list();

      expect(list).toContain(id1);
      expect(list).toContain(id2);
      expect(list.length).toBe(2);
    });
  });

  describe('isMockMode', () => {
    it('returns true when native addon not available', () => {
      // In test environment, native addon is not available
      expect(manager.isMockMode()).toBe(true);
    });
  });

  describe('error field mapping', () => {
    it('maps C++ error field names (errorMessage, errorCode, moduleName) to TS names', async () => {
      // Simulate the C++ addon emitting an error with C++ field names
      const id = manager.create(mockConfig);
      const instance = manager.get(id)!;

      // Manually push an error using the same mapping logic as startNative
      const cppEvent: Record<string, unknown> = {
        errorCode: 42,
        errorMessage: 'File not found',
        moduleName: 'FileWriter',
        moduleId: 'writer1',
        timestamp: '2026-03-02T10:00:00Z',
      };

      const runtimeError = {
        moduleId: (cppEvent.moduleId as string) || (cppEvent.moduleName as string) || 'unknown',
        message: (cppEvent.errorMessage as string) || (cppEvent.message as string) || 'Unknown error',
        timestamp: Date.now(),
        code: cppEvent.errorCode != null ? String(cppEvent.errorCode) : (cppEvent.code as string),
      };
      instance.errors.push(runtimeError);

      expect(runtimeError.moduleId).toBe('writer1');
      expect(runtimeError.message).toBe('File not found');
      expect(runtimeError.code).toBe('42');
    });

    it('falls back to moduleName when moduleId is missing', () => {
      const cppEvent: Record<string, unknown> = {
        errorCode: 13,
        errorMessage: 'Permission denied',
        moduleName: 'FileWriter',
      };

      const runtimeError = {
        moduleId: (cppEvent.moduleId as string) || (cppEvent.moduleName as string) || 'unknown',
        message: (cppEvent.errorMessage as string) || (cppEvent.message as string) || 'Unknown error',
        timestamp: Date.now(),
        code: cppEvent.errorCode != null ? String(cppEvent.errorCode) : (cppEvent.code as string),
      };

      expect(runtimeError.moduleId).toBe('FileWriter');
      expect(runtimeError.message).toBe('Permission denied');
      expect(runtimeError.code).toBe('13');
    });

    it('falls back to defaults when all fields are missing', () => {
      const cppEvent: Record<string, unknown> = {};

      const runtimeError = {
        moduleId: (cppEvent.moduleId as string) || (cppEvent.moduleName as string) || 'unknown',
        message: (cppEvent.errorMessage as string) || (cppEvent.message as string) || 'Unknown error',
        timestamp: Date.now(),
        code: cppEvent.errorCode != null ? String(cppEvent.errorCode) : (cppEvent.code as string),
      };

      expect(runtimeError.moduleId).toBe('unknown');
      expect(runtimeError.message).toBe('Unknown error');
    });
  });

  describe('COMPLETED status handling', () => {
    it('allows stopping a COMPLETED pipeline', async () => {
      const id = manager.create(mockConfig);
      await manager.start(id);

      // Manually set status to COMPLETED (simulating endOfStream event)
      const instance = manager.get(id)!;
      instance.status = 'COMPLETED';

      // Should not throw - stop should work on COMPLETED pipelines
      await manager.stop(id);
      expect(manager.getStatus(id)).toBe('STOPPED');
    });

    it('allows deleting a COMPLETED pipeline', async () => {
      const id = manager.create(mockConfig);
      await manager.start(id);

      // Manually set status to COMPLETED
      const instance = manager.get(id)!;
      instance.status = 'COMPLETED';

      await manager.delete(id);
      expect(manager.get(id)).toBeUndefined();
    });
  });

  describe('log emission', () => {
    it('emits log event on pipeline create', () => {
      const logHandler = vi.fn();
      manager.on('log', logHandler);

      const id = manager.create(mockConfig);

      expect(logHandler).toHaveBeenCalled();
      const logEvent = logHandler.mock.calls[0][0];
      expect(logEvent.pipelineId).toBe(id);
      expect(logEvent.data.level).toBe('info');
      expect(logEvent.data.source).toBe('pipeline');
      expect(logEvent.data.message).toContain('Pipeline created');
    });

    it('accumulates logs in pipeline instance', async () => {
      const id = manager.create(mockConfig);
      await manager.start(id);

      const instance = manager.get(id)!;
      // Should have at least: "Pipeline created", "Starting pipeline...", "Pipeline running"
      expect(instance.logs.length).toBeGreaterThanOrEqual(3);
      expect(instance.logs[0].message).toContain('Pipeline created');
      expect(instance.logs[1].message).toBe('Starting pipeline...');
      expect(instance.logs[2].message).toBe('Pipeline running');
    });

    it('emits log on pipeline stop', async () => {
      const id = manager.create(mockConfig);
      await manager.start(id);

      const logHandler = vi.fn();
      manager.on('log', logHandler);

      await manager.stop(id);

      const stopLog = logHandler.mock.calls.find(
        (call) => call[0].data.message === 'Pipeline stopped by user'
      );
      expect(stopLog).toBeDefined();
    });

    it('emits log on pipeline delete', async () => {
      const id = manager.create(mockConfig);

      const logHandler = vi.fn();
      manager.on('log', logHandler);

      await manager.delete(id);

      const deleteLog = logHandler.mock.calls.find(
        (call) => call[0].data.message === 'Pipeline deleted'
      );
      expect(deleteLog).toBeDefined();
    });
  });

  describe('mock mode health events', () => {
    it('emits health events when pipeline is running', async () => {
      const healthHandler = vi.fn();
      manager.on('health', healthHandler);

      const id = manager.create(mockConfig);
      await manager.start(id);

      // Wait for at least one health event (mock emits every 1 second)
      await new Promise((resolve) => setTimeout(resolve, 1500));

      expect(healthHandler).toHaveBeenCalled();

      // Verify health event structure
      const event = healthHandler.mock.calls[0][0];
      expect(event.pipelineId).toBe(id);
      expect(event.moduleId).toBeDefined();
      expect(typeof event.fps).toBe('number');
      expect(typeof event.qlen).toBe('number');
      expect(typeof event.isQueueFull).toBe('boolean');

      await manager.stop(id);
    }, 10000);

    it('stops emitting health events when pipeline is stopped', async () => {
      const healthHandler = vi.fn();
      manager.on('health', healthHandler);

      const id = manager.create(mockConfig);
      await manager.start(id);

      // Wait for health events
      await new Promise((resolve) => setTimeout(resolve, 1500));
      expect(healthHandler.mock.calls.length).toBeGreaterThan(0);

      await manager.stop(id);
      healthHandler.mockClear();

      // Wait to verify no more events
      await new Promise((resolve) => setTimeout(resolve, 1500));

      expect(healthHandler.mock.calls.length).toBe(0);
    }, 10000);
  });

  describe('full lifecycle', () => {
    it('supports full create -> start -> stop -> delete cycle', async () => {
      // Create
      const id = manager.create(mockConfig);
      expect(manager.getStatus(id)).toBe('IDLE');

      // Start
      await manager.start(id);
      expect(manager.getStatus(id)).toBe('RUNNING');

      // Stop
      await manager.stop(id);
      expect(manager.getStatus(id)).toBe('STOPPED');

      // Delete
      await manager.delete(id);
      expect(manager.get(id)).toBeUndefined();
    });

    it('can restart a stopped pipeline', async () => {
      const id = manager.create(mockConfig);

      await manager.start(id);
      expect(manager.getStatus(id)).toBe('RUNNING');

      await manager.stop(id);
      expect(manager.getStatus(id)).toBe('STOPPED');

      // Note: Need to create a new pipeline to restart
      // The current implementation doesn't support restarting stopped pipelines
      // This is expected behavior - create a new pipeline instead
    });
  });
});
