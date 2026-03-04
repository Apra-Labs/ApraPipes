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
import { getSchemaLoader } from './SchemaLoader.js';
import type { ModuleSchema } from './SchemaLoader.js';
import type {
  PipelineInstance,
  PipelineStatus,
  PipelineConfig,
  ModuleMetrics,
  RuntimeError,
  HealthEvent,
  ErrorEvent,
  LogEntry,
  LogLevel,
} from '../types/pipeline.js';

// Create require function for loading native addons in ESM context
const require = createRequire(import.meta.url);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const logger = createLogger('PipelineManager');

/** Maximum log entries per pipeline instance */
const LOG_BUFFER_MAX = 1000;

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
  private moduleSchemas: Record<string, ModuleSchema> | null = null;

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

    // Load module schemas asynchronously for type coercion
    this.loadSchemas();
  }

  /**
   * Load module schemas for property type coercion
   */
  private async loadSchemas(): Promise<void> {
    try {
      this.moduleSchemas = await getSchemaLoader().getSchema();
      logger.info(`Loaded schemas for ${Object.keys(this.moduleSchemas).length} module types`);
    } catch (error) {
      logger.warn('Failed to load module schemas for type coercion:', error);
    }
  }

  /**
   * Check if running in mock mode
   */
  isMockMode(): boolean {
    return this.useMockMode;
  }

  /**
   * Add a log entry to a pipeline instance and emit it
   */
  private addLog(instance: PipelineInstance, level: LogLevel, source: string, message: string, details?: Record<string, unknown>): void {
    const entry: LogEntry = {
      id: randomUUID(),
      timestamp: Date.now(),
      level,
      source,
      message,
      ...(details && { details }),
    };

    instance.logs.push(entry);
    // Ring buffer: keep only the last LOG_BUFFER_MAX entries
    if (instance.logs.length > LOG_BUFFER_MAX) {
      instance.logs = instance.logs.slice(instance.logs.length - LOG_BUFFER_MAX);
    }

    this.emit('log', { pipelineId: instance.id, data: entry });
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
      logs: [],
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
    const moduleCount = Object.keys(config.modules).length;
    logger.info(`Pipeline created: ${id} with ${moduleCount} modules`);

    this.addLog(instance, 'info', 'pipeline', `Pipeline created with ${moduleCount} modules`);
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
    this.addLog(instance, 'info', 'pipeline', 'Starting pipeline...');

    try {
      if (this.useMockMode) {
        await this.startMock(instance);
      } else {
        await this.startNative(instance);
      }

      instance.status = 'RUNNING';
      instance.startTime = Date.now();
      logger.info(`Pipeline started: ${id}`);
      this.addLog(instance, 'info', 'pipeline', 'Pipeline running');
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
      this.addLog(instance, 'error', 'pipeline', `Failed to start: ${errorMessage}`);
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

    if (instance.status !== 'RUNNING' && instance.status !== 'COMPLETED') {
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
      this.addLog(instance, 'info', 'pipeline', 'Pipeline stopped by user');
      this.emit('status', { pipelineId: id, status: instance.status });
    } catch (error) {
      instance.status = 'ERROR';
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      logger.error(`Pipeline stop failed: ${id}`, error);
      this.addLog(instance, 'error', 'pipeline', `Failed to stop: ${errorMessage}`);
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

    // Stop if running or completed
    if (instance.status === 'RUNNING' || instance.status === 'COMPLETED') {
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

    this.addLog(instance, 'info', 'pipeline', 'Pipeline deleted');
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
      const moduleIds = Object.keys(instance.config.modules);
      for (const moduleId of moduleIds) {
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

      // Emit a debug-level health tick log
      this.addLog(instance, 'debug', 'pipeline', `Health tick: ${moduleIds.length} modules reporting`);

      // Occasionally emit a mock error (1% chance)
      if (Math.random() < 0.01) {
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
    this.addLog(instance, 'debug', 'pipeline', `Config sent to addon (${pipelineConfig.length} bytes)`);

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
      // Map C++ field names (errorMessage, errorCode, moduleName) to TS names (message, code, moduleId)
      const raw = event as Record<string, unknown>;
      const runtimeError = {
        moduleId: (raw.moduleId as string) || (raw.moduleName as string) || 'unknown',
        message: (raw.errorMessage as string) || (raw.message as string) || 'Unknown error',
        timestamp: Date.now(),
        code: raw.errorCode != null ? String(raw.errorCode) : (raw.code as string),
      };
      instance.errors.push(runtimeError);

      const codeStr = runtimeError.code ? ` (code: ${runtimeError.code})` : '';
      this.addLog(instance, 'error', runtimeError.moduleId, `${runtimeError.message}${codeStr}`);
      this.emit('error', { pipelineId: instance.id, ...runtimeError });
    });

    // Set up lifecycle event listeners
    pipeline.on('endOfStream', () => {
      if (instance.status === 'RUNNING') {
        instance.status = 'COMPLETED';
        logger.info(`Pipeline completed (end of stream): ${instance.id}`);
        this.addLog(instance, 'info', 'pipeline', 'Pipeline completed \u2014 end of stream');
        this.emit('status', { pipelineId: instance.id, status: 'COMPLETED' });
      }
    });

    pipeline.on('stopped', () => {
      if (instance.status === 'RUNNING') {
        instance.status = 'STOPPED';
        logger.info(`Pipeline stopped (native event): ${instance.id}`);
        this.addLog(instance, 'info', 'pipeline', 'Pipeline stopped (native event)');
        this.emit('status', { pipelineId: instance.id, status: 'STOPPED' });
      }
    });

    // Initialize and run the pipeline
    await pipeline.init();
    await pipeline.run({ pauseSupport: true });
    pipeline.play();  // Resume from initial pause — run_all_threaded_withpause starts paused
  }

  /**
   * Coerce a property value to the type declared in the schema.
   * The C++ PipelineValidator rejects type mismatches (e.g. string "320" for an int prop).
   */
  private coercePropertyValue(value: unknown, schemaType: string): unknown {
    if (value === undefined || value === null) return value;

    switch (schemaType) {
      case 'int': {
        if (typeof value === 'number') return Math.round(value);
        if (typeof value === 'string') {
          const parsed = parseInt(value, 10);
          return isNaN(parsed) ? value : parsed;
        }
        return value;
      }
      case 'float': {
        if (typeof value === 'number') return value;
        if (typeof value === 'string') {
          const parsed = parseFloat(value);
          return isNaN(parsed) ? value : parsed;
        }
        return value;
      }
      case 'bool': {
        if (typeof value === 'boolean') return value;
        if (typeof value === 'string') return value === 'true';
        return value;
      }
      default:
        // string, enum, json — pass as-is
        return value;
    }
  }

  /**
   * Convert PipelineConfig to the format expected by aprapipes.node
   * Coerces property value types based on module schema to prevent E201 validation errors.
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
      let props: Record<string, unknown> | undefined;

      if (moduleConfig.properties && Object.keys(moduleConfig.properties).length > 0) {
        props = { ...moduleConfig.properties };

        // Coerce types using schema if available
        const schema = this.moduleSchemas?.[moduleConfig.type];
        if (schema) {
          for (const [key, value] of Object.entries(props)) {
            const propSchema = schema.properties[key];
            if (propSchema) {
              props[key] = this.coercePropertyValue(value, propSchema.type);
            }
          }
        }
      }

      pipelineObj.modules[moduleId] = {
        type: moduleConfig.type,
        ...(props ? { props } : {}),
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
