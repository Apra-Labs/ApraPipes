/**
 * Pipeline runtime types for ApraPipes Studio
 */

/**
 * Pipeline execution status
 */
export type PipelineStatus = 'IDLE' | 'CREATING' | 'RUNNING' | 'STOPPING' | 'STOPPED' | 'ERROR';

/**
 * Module runtime metrics from health events
 */
export interface ModuleMetrics {
  /** Frames per second */
  fps: number;
  /** Current queue length */
  qlen: number;
  /** Whether the queue is full */
  isQueueFull: boolean;
  /** Last update timestamp */
  timestamp: number;
}

/**
 * Runtime error from a module
 */
export interface RuntimeError {
  /** Module ID that generated the error */
  moduleId: string;
  /** Error message */
  message: string;
  /** When the error occurred */
  timestamp: number;
  /** Optional error code */
  code?: string;
}

/**
 * Health event from aprapipes
 */
export interface HealthEvent {
  /** Module ID */
  moduleId: string;
  /** Frames per second */
  fps: number;
  /** Queue length */
  qlen: number;
  /** Is queue full */
  isQueueFull: boolean;
}

/**
 * Error event from aprapipes
 */
export interface ErrorEvent {
  /** Module ID */
  moduleId: string;
  /** Error message */
  message: string;
  /** Error code */
  code?: string;
}

/**
 * Pipeline instance interface
 */
export interface PipelineInstance {
  /** Unique pipeline ID */
  id: string;
  /** Current status */
  status: PipelineStatus;
  /** Pipeline configuration */
  config: PipelineConfig;
  /** Module metrics (keyed by module ID) */
  metrics: Record<string, ModuleMetrics>;
  /** Runtime errors */
  errors: RuntimeError[];
  /** Start time (if running) */
  startTime?: number;
  /** Native pipeline reference (if using real aprapipes) */
  nativePipeline?: unknown;
  /** Mock timer ID (if using mock) */
  mockTimerId?: NodeJS.Timeout;
}

/**
 * Pipeline configuration (re-exported for convenience)
 */
export interface PipelineConfig {
  modules: Record<string, ModuleConfig>;
  connections: ConnectionConfig[];
}

/**
 * Module configuration
 */
export interface ModuleConfig {
  type: string;
  properties?: Record<string, unknown>;
}

/**
 * Connection configuration
 */
export interface ConnectionConfig {
  from: string;
  to: string;
}

/**
 * Event listener type for pipeline events
 */
export type PipelineEventListener = (event: HealthEvent | ErrorEvent | PipelineStatus) => void;

/**
 * Create pipeline request
 */
export interface CreatePipelineRequest {
  config: PipelineConfig;
}

/**
 * Create pipeline response
 */
export interface CreatePipelineResponse {
  pipelineId: string;
  status: PipelineStatus;
}

/**
 * Pipeline status response
 */
export interface PipelineStatusResponse {
  id: string;
  status: PipelineStatus;
  metrics: Record<string, ModuleMetrics>;
  errors: RuntimeError[];
  startTime?: number;
  duration?: number;
}

/**
 * Simple operation response
 */
export interface OperationResponse {
  success: boolean;
  status: PipelineStatus;
  message?: string;
}
