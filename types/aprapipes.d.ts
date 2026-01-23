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
   * Get detailed information about ALL registered module types
   * Used for schema generation - same structure as CLI `describe --all --json`
   * @returns Object containing all modules keyed by name
   */
  export function describeAllModules(): AllModulesSchema;

  /**
   * Validate a pipeline configuration without running it
   * @param config Pipeline configuration (JSON string or object)
   * @returns Validation result with any issues found
   */
  export function validatePipeline(config: string | PipelineConfig): ValidationResult;

  // ============================================================
  // Type definitions
  // ============================================================

  /**
   * Schema containing all registered modules and frame types
   * Returned by describeAllModules() and CLI `describe --all --json`
   */
  export interface AllModulesSchema {
    modules: { [moduleName: string]: ModuleInfo };
    frameTypes: { [typeName: string]: FrameTypeInfo };
  }

  /**
   * Frame type metadata from FrameTypeRegistry
   */
  export interface FrameTypeInfo {
    parent: string;
    description: string;
    tags: string[];
    attributes?: { [attrName: string]: FrameTypeAttribute };
    ancestors?: string[];
    subtypes?: string[];
  }

  export interface FrameTypeAttribute {
    type: string;
    required: boolean;
    description: string;
    enumValues?: string[];
  }

  export interface ModuleInfo {
    name: string;
    description: string;
    version: string;
    category: 'source' | 'sink' | 'transform' | 'analytics' | 'controller' | 'utility' | 'unknown';
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
    mutability: string;
    default: string;
    min?: string;
    max?: string;
    enumValues?: string[];
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

    // ============================================================
    // Event methods (Phase 4)
    // ============================================================

    /**
     * Register an event listener
     * @param event Event name ('error', 'health', 'started', 'stopped', etc.)
     * @param callback Function to call when event occurs
     * @returns this for chaining
     */
    on(event: 'error', callback: (error: PipelineError) => void): this;
    on(event: 'health', callback: (health: PipelineHealth) => void): this;
    on(event: 'started' | 'stopped' | 'paused' | 'resumed', callback: (event: LifecycleEvent) => void): this;
    on(event: string, callback: (data: any) => void): this;

    /**
     * Remove an event listener
     * @param event Event name
     * @param callback Function to remove
     * @returns this for chaining
     */
    off(event: string, callback: (...args: any[]) => void): this;

    /**
     * Remove all listeners for an event (or all events if no event specified)
     * @param event Optional event name
     * @returns this for chaining
     */
    removeAllListeners(event?: string): this;
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

    // ============================================================
    // Dynamic Property Methods
    // ============================================================

    /**
     * Check if module supports dynamic properties
     * @returns true if getProperty/setProperty methods are available
     */
    hasDynamicProperties(): boolean;

    /**
     * Get list of dynamic property names
     * @returns Array of property names (e.g., ["roiX", "roiY", "roiWidth", "roiHeight"])
     */
    getDynamicPropertyNames(): string[];

    /**
     * Get a dynamic property value
     * @param name Property name
     * @returns Property value (number, boolean, or string)
     * @throws Error if property doesn't exist or module doesn't support dynamic properties
     */
    getProperty(name: string): number | boolean | string;

    /**
     * Set a dynamic property value (takes effect at runtime)
     * @param name Property name
     * @param value New value
     * @returns true if successfully set
     * @throws Error if property doesn't exist or module doesn't support dynamic properties
     */
    setProperty(name: string, value: number | boolean | string): boolean;
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

  // ============================================================
  // Event types (Phase 4)
  // ============================================================

  export interface PipelineError {
    errorCode: number;
    errorMessage: string;
    moduleName: string;
    moduleId: string;
    timestamp: string;
  }

  export interface PipelineHealth {
    moduleId: string;
    timestamp: string;
  }

  export interface LifecycleEvent {
    event: string;
    timestamp: string;
  }
}
