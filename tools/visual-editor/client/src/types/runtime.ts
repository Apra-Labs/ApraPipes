/**
 * Runtime types for pipeline execution monitoring
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
 * WebSocket message types
 */
export type WebSocketMessageType =
  | 'subscribe'
  | 'unsubscribe'
  | 'health'
  | 'error'
  | 'status'
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
 * Connection state
 */
export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'reconnecting';
