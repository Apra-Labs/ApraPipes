/**
 * Runtime types for pipeline execution monitoring
 */

/**
 * Pipeline execution status
 */
export type PipelineStatus = 'IDLE' | 'CREATING' | 'RUNNING' | 'STOPPING' | 'STOPPED' | 'COMPLETED' | 'ERROR';

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
 * Log level for pipeline log entries
 */
export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

/**
 * Structured log entry from pipeline execution
 */
export interface LogEntry {
  /** Unique ID for React keys */
  id: string;
  /** Unix timestamp in milliseconds */
  timestamp: number;
  /** Log severity level */
  level: LogLevel;
  /** Source: moduleId, 'pipeline', or 'addon' */
  source: string;
  /** Human-readable log message */
  message: string;
  /** Optional structured data */
  details?: Record<string, unknown>;
}

/**
 * WebSocket message types
 */
export type WebSocketMessageType =
  | 'subscribe'
  | 'unsubscribe'
  | 'health'
  | 'error'
  | 'status'
  | 'log'
  | 'subscribed'
  | 'unsubscribed'
  | 'error_message';

/**
 * Base WebSocket message
 */
export interface WebSocketMessage {
  event: WebSocketMessageType;
  pipelineId?: string;
  data?: unknown;
}

/**
 * Health event message
 */
export interface HealthMessage extends WebSocketMessage {
  event: 'health';
  pipelineId: string;
  data: {
    moduleId: string;
    fps: number;
    qlen: number;
    isQueueFull: boolean;
  };
}

/**
 * Error event message
 */
export interface ErrorMessage extends WebSocketMessage {
  event: 'error';
  pipelineId: string;
  data: {
    moduleId: string;
    message: string;
    code?: string;
  };
}

/**
 * Status event message
 */
export interface StatusMessage extends WebSocketMessage {
  event: 'status';
  pipelineId: string;
  data: {
    status: PipelineStatus;
  };
}

/**
 * Log event message
 */
export interface LogMessage extends WebSocketMessage {
  event: 'log';
  pipelineId: string;
  data: LogEntry;
}

/**
 * Connection state
 */
export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'reconnecting';
