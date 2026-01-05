/**
 * @apralabs/aprapipes - Native Node.js bindings for ApraPipes
 *
 * Video processing pipeline framework with GPU acceleration
 */

declare module '@apralabs/aprapipes' {
  /**
   * Get the addon version
   */
  export function getVersion(): string;

  /**
   * Get list of all registered module types
   * @returns Array of module type names
   */
  export function listModules(): string[];

  /**
   * Get detailed information about a module type
   * @param moduleName Name of the module type
   * @returns Module metadata object
   */
  export function describeModule(moduleName: string): ModuleInfo;

  /**
   * Validate a pipeline configuration without running it
   * @param config Pipeline configuration (JSON string or object)
   * @returns Validation result with any issues found
   */
  export function validatePipeline(config: string | PipelineConfig): ValidationResult;

  // ============================================================
  // Type definitions
  // ============================================================

  export interface ModuleInfo {
    name: string;
    description: string;
    version: string;
    category: 'source' | 'sink' | 'transform' | 'analytics' | 'utility' | 'unknown';
    tags: string[];
    properties: PropertyInfo[];
    inputs: PinInfo[];
    outputs: PinInfo[];
  }

  export interface PropertyInfo {
    name: string;
    type: string;
    required: boolean;
    description: string;
  }

  export interface PinInfo {
    name: string;
    required?: boolean;
    frameTypes: string[];
  }

  export interface ValidationResult {
    valid: boolean;
    issues: ValidationIssue[];
  }

  export interface ValidationIssue {
    level: 'error' | 'warning' | 'info';
    code: string;
    message: string;
    location: string;
    suggestion?: string;
  }

  // ============================================================
  // Pipeline configuration types
  // ============================================================

  export interface PipelineConfig {
    pipeline?: PipelineSettings;
    modules: { [id: string]: ModuleConfig };
    connections: ConnectionConfig[];
  }

  export interface PipelineSettings {
    name?: string;
    description?: string;
    queue_size?: number;
    on_error?: 'stop_pipeline' | 'restart_module' | 'skip';
  }

  export interface ModuleConfig {
    type: string;
    props?: { [key: string]: any };
  }

  export interface ConnectionConfig {
    from: string;
    to: string;
  }

  // ============================================================
  // Future Phase 3+ types (not yet implemented)
  // ============================================================

  // export function createPipeline(config: string | PipelineConfig): Pipeline;

  // export interface Pipeline {
  //   init(): Promise<void>;
  //   run(): Promise<void>;
  //   stop(): Promise<void>;
  //   pause(): void;
  //   play(): void;
  //   getModule(id: string): ModuleHandle | null;
  //   getStatus(): PipelineStatus;
  //   on(event: 'error', handler: (error: PipelineError) => void): void;
  //   on(event: 'stats', handler: (stats: PipelineStats) => void): void;
  //   on(event: 'health', handler: (health: HealthReport) => void): void;
  //   off(event: string, handler: Function): void;
  // }

  // export interface ModuleHandle {
  //   id: string;
  //   type: string;
  //   getProperty<T = any>(name: string): T;
  //   setProperty<T = any>(name: string, value: T): void;
  //   getStatus(): ModuleStatus;
  // }

  // export interface PipelineStatus {
  //   state: 'idle' | 'running' | 'paused' | 'stopped' | 'error';
  //   modules: { [id: string]: ModuleStatus };
  // }

  // export interface ModuleStatus {
  //   state: 'idle' | 'running' | 'paused' | 'error';
  //   framesProcessed: number;
  //   currentFps: number;
  //   errorCount: number;
  // }

  // export interface PipelineError {
  //   moduleId: string;
  //   errorCode: number;
  //   errorMessage: string;
  //   timestamp: Date;
  // }

  // export interface PipelineStats {
  //   timestamp: Date;
  //   data: PipelineStatus;
  // }

  // export interface HealthReport {
  //   timestamp: Date;
  //   healthy: boolean;
  //   modules: { [id: string]: boolean };
  // }
}
