/**
 * Pipeline Manager Service
 *
 * Manages pipeline lifecycle: create, start, stop, delete
 * Supports both native aprapipes addon and mock mode for development
 */

import { EventEmitter } from 'events';
import { randomUUID } from 'crypto';
import { createLogger } from '../utils/logger.js';
import type {
  PipelineInstance,
  PipelineStatus,
  PipelineConfig,
  ModuleMetrics,
  RuntimeError,
  HealthEvent,
  ErrorEvent,
} from '../types/pipeline.js';

const logger = createLogger('PipelineManager');

/**
 * Try to load the native aprapipes addon
 */
function tryLoadNativeAddon(): unknown | null {
  try {
    // Try multiple paths for the native addon
    const paths = [
      '../../bin/aprapipes.node',
      '../../../bin/aprapipes.node',
      '../../../../bin/aprapipes.node',
    ];

    for (const addonPath of paths) {
      try {
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        const addon = require(addonPath);
        if (addon && typeof addon.createPipeline === 'function') {
          logger.info(`Native addon loaded from ${addonPath}`);
          return addon;
        }
      } catch {
        // Try next path
      }
    }

    logger.warn('Native aprapipes addon not found, using mock mode');
    return null;
  } catch (error) {
    logger.warn('Failed to load native addon, using mock mode:', error);
    return null;
  }
}

/**
 * Pipeline Manager class
 * Extends EventEmitter to broadcast pipeline events to subscribers
 */
export class PipelineManager extends EventEmitter {
  private pipelines: Map<string, PipelineInstance> = new Map();
  private nativeAddon: unknown | null;
  private useMockMode: boolean;

  constructor() {
    super();

    // Add default error listener to prevent unhandled error crashes
    // Actual error handling is done by subscribers (MetricsStream, etc.)
    this.on('error', () => {
      // Error events are logged elsewhere, this just prevents crash
    });

    this.nativeAddon = tryLoadNativeAddon();
    this.useMockMode = this.nativeAddon === null;

    if (this.useMockMode) {
      logger.info('PipelineManager initialized in MOCK mode');
    } else {
      logger.info('PipelineManager initialized with native addon');
    }
  }

  /**
   * Check if running in mock mode
   */
  isMockMode(): boolean {
    return this.useMockMode;
  }

  /**
   * Create a new pipeline from configuration
   */
  create(config: PipelineConfig): string {
    const id = randomUUID();

    const instance: PipelineInstance = {
      id,
      status: 'IDLE',
      config,
      metrics: {},
      errors: [],
    };

    // Initialize metrics for each module
    for (const moduleId of Object.keys(config.modules)) {
      instance.metrics[moduleId] = {
        fps: 0,
        qlen: 0,
        isQueueFull: false,
        timestamp: Date.now(),
      };
    }

    this.pipelines.set(id, instance);
    logger.info(`Pipeline created: ${id} with ${Object.keys(config.modules).length} modules`);

    this.emit('created', { pipelineId: id, status: instance.status });
    return id;
  }

  /**
   * Start a pipeline
   */
  async start(id: string): Promise<void> {
    const instance = this.pipelines.get(id);
    if (!instance) {
      throw new Error(`Pipeline not found: ${id}`);
    }

    if (instance.status === 'RUNNING') {
      throw new Error(`Pipeline already running: ${id}`);
    }

    instance.status = 'CREATING';
    this.emit('status', { pipelineId: id, status: instance.status });

    try {
      if (this.useMockMode) {
        await this.startMock(instance);
      } else {
        await this.startNative(instance);
      }

      instance.status = 'RUNNING';
      instance.startTime = Date.now();
      logger.info(`Pipeline started: ${id}`);
      this.emit('status', { pipelineId: id, status: instance.status });
    } catch (error) {
      instance.status = 'ERROR';
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      instance.errors.push({
        moduleId: 'pipeline',
        message: `Failed to start: ${errorMessage}`,
        timestamp: Date.now(),
      });
      logger.error(`Pipeline start failed: ${id}`, error);
      this.emit('status', { pipelineId: id, status: instance.status });
      this.emit('error', { pipelineId: id, moduleId: 'pipeline', message: errorMessage });
      throw error;
    }
  }

  /**
   * Stop a pipeline
   */
  async stop(id: string): Promise<void> {
    const instance = this.pipelines.get(id);
    if (!instance) {
      throw new Error(`Pipeline not found: ${id}`);
    }

    if (instance.status !== 'RUNNING') {
      throw new Error(`Pipeline not running: ${id}`);
    }

    instance.status = 'STOPPING';
    this.emit('status', { pipelineId: id, status: instance.status });

    try {
      if (this.useMockMode) {
        this.stopMock(instance);
      } else {
        await this.stopNative(instance);
      }

      instance.status = 'STOPPED';
      logger.info(`Pipeline stopped: ${id}`);
      this.emit('status', { pipelineId: id, status: instance.status });
    } catch (error) {
      instance.status = 'ERROR';
      logger.error(`Pipeline stop failed: ${id}`, error);
      this.emit('status', { pipelineId: id, status: instance.status });
      throw error;
    }
  }

  /**
   * Get pipeline instance by ID
   */
  get(id: string): PipelineInstance | undefined {
    return this.pipelines.get(id);
  }

  /**
   * Get pipeline status
   */
  getStatus(id: string): PipelineStatus | undefined {
    return this.pipelines.get(id)?.status;
  }

  /**
   * Delete a pipeline and cleanup resources
   */
  async delete(id: string): Promise<void> {
    const instance = this.pipelines.get(id);
    if (!instance) {
      throw new Error(`Pipeline not found: ${id}`);
    }

    // Stop if running
    if (instance.status === 'RUNNING') {
      await this.stop(id);
    }

    // Cleanup mock timer if exists
    if (instance.mockTimerId) {
      clearInterval(instance.mockTimerId);
    }

    // Cleanup native pipeline if exists
    if (instance.nativePipeline && !this.useMockMode) {
      try {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (instance.nativePipeline as any).destroy?.();
      } catch (error) {
        logger.warn(`Failed to destroy native pipeline: ${id}`, error);
      }
    }

    this.pipelines.delete(id);
    logger.info(`Pipeline deleted: ${id}`);
    this.emit('deleted', { pipelineId: id });
  }

  /**
   * List all pipeline IDs
   */
  list(): string[] {
    return Array.from(this.pipelines.keys());
  }

  /**
   * Start pipeline in mock mode - simulates metrics
   */
  private async startMock(instance: PipelineInstance): Promise<void> {
    // Simulate startup delay
    await new Promise((resolve) => setTimeout(resolve, 100));

    // Start mock metrics generation
    instance.mockTimerId = setInterval(() => {
      if (instance.status !== 'RUNNING') {
        return;
      }

      // Generate mock health events for each module
      for (const moduleId of Object.keys(instance.config.modules)) {
        const metrics: ModuleMetrics = {
          fps: 25 + Math.random() * 10, // 25-35 fps
          qlen: Math.floor(Math.random() * 10), // 0-9 queue length
          isQueueFull: Math.random() < 0.05, // 5% chance of full queue
          timestamp: Date.now(),
        };

        instance.metrics[moduleId] = metrics;

        const healthEvent: HealthEvent = {
          moduleId,
          fps: metrics.fps,
          qlen: metrics.qlen,
          isQueueFull: metrics.isQueueFull,
        };

        this.emit('health', { pipelineId: instance.id, ...healthEvent });
      }

      // Occasionally emit a mock error (1% chance)
      if (Math.random() < 0.01) {
        const moduleIds = Object.keys(instance.config.modules);
        if (moduleIds.length > 0) {
          const randomModule = moduleIds[Math.floor(Math.random() * moduleIds.length)];
          const errorEvent: ErrorEvent = {
            moduleId: randomModule,
            message: 'Mock transient error',
            code: 'MOCK_ERROR',
          };

          const runtimeError: RuntimeError = {
            moduleId: randomModule,
            message: errorEvent.message,
            timestamp: Date.now(),
            code: errorEvent.code,
          };

          instance.errors.push(runtimeError);
          this.emit('error', { pipelineId: instance.id, ...errorEvent });
        }
      }
    }, 1000); // Update every second
  }

  /**
   * Stop mock pipeline
   */
  private stopMock(instance: PipelineInstance): void {
    if (instance.mockTimerId) {
      clearInterval(instance.mockTimerId);
      instance.mockTimerId = undefined;
    }

    // Reset metrics
    for (const moduleId of Object.keys(instance.metrics)) {
      instance.metrics[moduleId] = {
        fps: 0,
        qlen: 0,
        isQueueFull: false,
        timestamp: Date.now(),
      };
    }
  }

  /**
   * Start pipeline with native addon
   */
  private async startNative(instance: PipelineInstance): Promise<void> {
    if (!this.nativeAddon) {
      throw new Error('Native addon not available');
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const addon = this.nativeAddon as any;

    // Create native pipeline
    const pipeline = addon.createPipeline(JSON.stringify(instance.config));
    instance.nativePipeline = pipeline;

    // Set up event listeners
    pipeline.on('health', (event: HealthEvent) => {
      if (instance.status !== 'RUNNING') return;

      instance.metrics[event.moduleId] = {
        fps: event.fps,
        qlen: event.qlen,
        isQueueFull: event.isQueueFull,
        timestamp: Date.now(),
      };

      this.emit('health', { pipelineId: instance.id, ...event });
    });

    pipeline.on('error', (event: ErrorEvent) => {
      instance.errors.push({
        moduleId: event.moduleId,
        message: event.message,
        timestamp: Date.now(),
        code: event.code,
      });

      this.emit('error', { pipelineId: instance.id, ...event });
    });

    // Start the pipeline
    await pipeline.start();
  }

  /**
   * Stop native pipeline
   */
  private async stopNative(instance: PipelineInstance): Promise<void> {
    if (!instance.nativePipeline) {
      return;
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const pipeline = instance.nativePipeline as any;
    await pipeline.stop();
  }
}

// Singleton instance
let pipelineManagerInstance: PipelineManager | null = null;

/**
 * Get the singleton PipelineManager instance
 */
export function getPipelineManager(): PipelineManager {
  if (!pipelineManagerInstance) {
    pipelineManagerInstance = new PipelineManager();
  }
  return pipelineManagerInstance;
}

/**
 * Reset the pipeline manager (for testing)
 */
export function resetPipelineManager(): void {
  if (pipelineManagerInstance) {
    // Stop and delete all pipelines
    for (const id of pipelineManagerInstance.list()) {
      try {
        const instance = pipelineManagerInstance.get(id);
        if (instance?.mockTimerId) {
          clearInterval(instance.mockTimerId);
        }
      } catch {
        // Ignore errors during cleanup
      }
    }
    pipelineManagerInstance.removeAllListeners();
    pipelineManagerInstance = null;
  }
}

export default PipelineManager;
