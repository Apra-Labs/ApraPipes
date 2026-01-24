/**
 * Pipeline Manager Service
 *
 * Manages pipeline lifecycle: create, start, stop, delete
 * Supports both native aprapipes addon and mock mode for development
 */

import { EventEmitter } from 'events';
import { randomUUID } from 'crypto';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { createRequire } from 'module';
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

// Create require function for loading native addons in ESM context
const require = createRequire(import.meta.url);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const logger = createLogger('PipelineManager');

/**
 * Try to load the native aprapipes addon
 */
function tryLoadNativeAddon(): NativeAddon | null {
  try {
    // Try multiple paths for the native addon
    const addonPaths = [
      path.resolve(process.cwd(), 'aprapipes.node'),
      path.resolve(process.cwd(), '..', 'aprapipes.node'),
      path.resolve(process.cwd(), '..', '..', 'aprapipes.node'),
      path.resolve(process.cwd(), '..', '..', '..', 'aprapipes.node'),
      path.resolve(__dirname, '..', '..', '..', '..', '..', 'aprapipes.node'),
    ];

    for (const addonPath of addonPaths) {
      try {
        const addon = require(addonPath);
        if (addon && typeof addon.createPipeline === 'function') {
          logger.info(`Native addon loaded from ${addonPath}`);
          return addon as NativeAddon;
        }
      } catch (err) {
        // Log error only if the file exists but failed to load
        if (fs.existsSync(addonPath)) {
          const errorMessage = err instanceof Error ? err.message : String(err);
          logger.warn(`Failed to load addon from ${addonPath}: ${errorMessage}`);
        }
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
 * Type for the native addon
 */
interface NativeAddon {
  createPipeline: (config: string | object) => NativePipeline;
  validatePipeline: (config: string | object) => { valid: boolean; issues: unknown[] };
}

/**
 * Type for a native pipeline instance
 */
interface NativePipeline {
  init: () => Promise<boolean>;
  run: (options?: { pauseSupport?: boolean }) => Promise<boolean>;
  stop: () => Promise<boolean>;
  terminate: () => Promise<boolean>;
  pause: () => void;
  play: () => void;
  getStatus: () => string;
  getName: () => string;
  getModuleIds: () => string[];
  on: (event: string, callback: (data: unknown) => void) => NativePipeline;
  off: (event: string, callback: (data: unknown) => void) => NativePipeline;
  removeAllListeners: (event?: string) => NativePipeline;
}

/**
 * Options for PipelineManager constructor
 */
interface PipelineManagerOptions {
  /** Force mock mode even if native addon is available (for testing) */
  forceMockMode?: boolean;
}

/**
 * Pipeline Manager class
 * Extends EventEmitter to broadcast pipeline events to subscribers
 */
export class PipelineManager extends EventEmitter {
  private pipelines: Map<string, PipelineInstance> = new Map();
  private nativeAddon: NativeAddon | null;
  private useMockMode: boolean;

  constructor(options: PipelineManagerOptions = {}) {
    super();

    // Add default error listener to prevent unhandled error crashes
    // Actual error handling is done by subscribers (MetricsStream, etc.)
    this.on('error', () => {
      // Error events are logged elsewhere, this just prevents crash
    });

    if (options.forceMockMode) {
      this.nativeAddon = null;
      this.useMockMode = true;
      logger.info('PipelineManager initialized in MOCK mode (forced)');
    } else {
      this.nativeAddon = tryLoadNativeAddon();
      this.useMockMode = this.nativeAddon === null;

      if (this.useMockMode) {
        logger.info('PipelineManager initialized in MOCK mode');
      } else {
        logger.info('PipelineManager initialized with native addon');
      }
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
        const pipeline = instance.nativePipeline as NativePipeline;
        pipeline.removeAllListeners();
        await pipeline.terminate();
      } catch (error) {
        logger.warn(`Failed to terminate native pipeline: ${id}`, error);
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

    // Convert config to the format expected by the addon
    const pipelineConfig = this.convertToPipelineConfig(instance.config);

    // Create native pipeline
    const pipeline = this.nativeAddon.createPipeline(pipelineConfig);
    instance.nativePipeline = pipeline;

    // Set up event listeners
    pipeline.on('health', (event: unknown) => {
      if (instance.status !== 'RUNNING') return;

      const healthEvent = event as HealthEvent;
      instance.metrics[healthEvent.moduleId] = {
        fps: healthEvent.fps,
        qlen: healthEvent.qlen,
        isQueueFull: healthEvent.isQueueFull,
        timestamp: Date.now(),
      };

      this.emit('health', { pipelineId: instance.id, ...healthEvent });
    });

    pipeline.on('error', (event: unknown) => {
      const errorEvent = event as ErrorEvent;
      instance.errors.push({
        moduleId: errorEvent.moduleId,
        message: errorEvent.message,
        timestamp: Date.now(),
        code: errorEvent.code,
      });

      this.emit('error', { pipelineId: instance.id, ...errorEvent });
    });

    // Initialize and run the pipeline
    await pipeline.init();
    await pipeline.run({ pauseSupport: true });
  }

  /**
   * Convert PipelineConfig to the format expected by aprapipes.node
   */
  private convertToPipelineConfig(config: PipelineConfig): string {
    const pipelineObj = {
      modules: {} as Record<string, { type: string; props?: Record<string, unknown> }>,
      connections: config.connections.map((conn) => ({
        from: conn.from,
        to: conn.to,
      })),
    };

    for (const [moduleId, moduleConfig] of Object.entries(config.modules)) {
      pipelineObj.modules[moduleId] = {
        type: moduleConfig.type,
        ...(moduleConfig.properties && Object.keys(moduleConfig.properties).length > 0
          ? { props: moduleConfig.properties }
          : {}),
      };
    }

    return JSON.stringify(pipelineObj);
  }

  /**
   * Stop native pipeline
   */
  private async stopNative(instance: PipelineInstance): Promise<void> {
    if (!instance.nativePipeline) {
      return;
    }

    const pipeline = instance.nativePipeline as NativePipeline;
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
