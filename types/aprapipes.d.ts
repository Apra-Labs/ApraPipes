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
  // Pipeline creation and control (Phase 3)
  // ============================================================

  /**
   * Create a pipeline from configuration
   * @param config Pipeline configuration (JSON string or object)
   * @returns Pipeline instance ready for init/run
   * @throws Error if config is invalid or build fails
   */
  export function createPipeline(config: string | PipelineConfig): Pipeline;

  /**
   * Pipeline class - wraps a running pipeline
   */
  export class Pipeline {
    /**
     * Initialize the pipeline (must be called before run)
     * @returns Promise that resolves when init completes
     */
    init(): Promise<boolean>;

    /**
     * Start the pipeline running
     * @param options Run options
     * @returns Promise that resolves when pipeline stops
     */
    run(options?: { pauseSupport?: boolean }): Promise<boolean>;

    /**
     * Stop the pipeline
     * @returns Promise that resolves when stop completes
     */
    stop(): Promise<boolean>;

    /**
     * Terminate and cleanup the pipeline
     * @returns Promise that resolves when terminate completes
     */
    terminate(): Promise<boolean>;

    /**
     * Pause the pipeline (requires pauseSupport in run options)
     */
    pause(): void;

    /**
     * Resume a paused pipeline
     */
    play(): void;

    /**
     * Step one frame (when paused)
     */
    step(): void;

    /**
     * Get current pipeline status
     * @returns Status string (PL_CREATED, PL_INITED, PL_RUNNING, etc.)
     */
    getStatus(): string;

    /**
     * Get pipeline name
     */
    getName(): string;

    /**
     * Get list of module instance IDs
     */
    getModuleIds(): string[];

    /**
     * Get a module handle by ID
     * @param id Module instance ID
     * @returns Module handle or null if not found
     */
    getModule(id: string): ModuleHandle | null;
  }

  // ============================================================
  // Module Handle (Phase 3)
  // ============================================================

  /**
   * ModuleHandle - provides access to individual module info
   */
  export class ModuleHandle {
    /** Instance ID (e.g., "source", "decoder") */
    readonly id: string;

    /** Module type (e.g., "TestSignalGenerator", "H264Decoder") */
    readonly type: string;

    /** Internal module name (may be null if module not accessible) */
    readonly name: string | null;

    /**
     * Get module properties (fps, qlen, etc.)
     */
    getProps(): ModuleProps;

    /**
     * Check if module's input queue is full
     */
    isInputQueFull(): boolean;

    /**
     * Check if module is running
     */
    isRunning(): boolean;
  }

  export interface ModuleProps {
    fps: number;
    qlen: number;
    logHealth: boolean;
    logHealthFrequency: number;
    maxConcurrentFrames: number;
    enableHealthCallBack: boolean;
    healthUpdateIntervalInSec: number;
  }

  // Event system (Phase 4)
  // export interface PipelineError {
  //   moduleId: string;
  //   errorCode: number;
  //   errorMessage: string;
  //   timestamp: Date;
  // }

  // export interface PipelineStats {
  //   timestamp: Date;
  //   fps: number;
  //   framesProcessed: number;
  // }
}
