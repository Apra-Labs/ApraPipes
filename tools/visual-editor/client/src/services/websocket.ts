/**
 * WebSocket Client Service
 *
 * Manages WebSocket connection to the metrics stream server.
 * Handles auto-reconnect with exponential backoff.
 */

import type {
  WebSocketMessage,
  HealthMessage,
  ErrorMessage,
  StatusMessage,
  ConnectionState,
} from '../types/runtime';

const WS_URL = 'ws://localhost:3000/ws';

// Reconnect settings
const INITIAL_RECONNECT_DELAY = 1000; // 1 second
const MAX_RECONNECT_DELAY = 30000; // 30 seconds
const RECONNECT_MULTIPLIER = 1.5;

/**
 * Event handlers for WebSocket events
 */
export interface WebSocketHandlers {
  onConnect?: () => void;
  onDisconnect?: () => void;
  onHealth?: (message: HealthMessage) => void;
  onError?: (message: ErrorMessage) => void;
  onStatus?: (message: StatusMessage) => void;
  onConnectionStateChange?: (state: ConnectionState) => void;
}

/**
 * WebSocket Client class
 */
export class WebSocketClient {
  private socket: WebSocket | null = null;
  private handlers: WebSocketHandlers = {};
  private reconnectDelay = INITIAL_RECONNECT_DELAY;
  private reconnectTimer: number | null = null;
  private connectionState: ConnectionState = 'disconnected';
  private subscriptions: Set<string> = new Set();
  private pendingSubscriptions: Set<string> = new Set();

  /**
   * Set event handlers
   */
  setHandlers(handlers: WebSocketHandlers): void {
    this.handlers = handlers;
  }

  /**
   * Get current connection state
   */
  getConnectionState(): ConnectionState {
    return this.connectionState;
  }

  /**
   * Connect to WebSocket server
   */
  connect(): void {
    if (this.socket?.readyState === WebSocket.OPEN) {
      return;
    }

    this.setConnectionState('connecting');
    this.socket = new WebSocket(WS_URL);

    this.socket.onopen = () => {
      this.handleOpen();
    };

    this.socket.onclose = () => {
      this.handleClose();
    };

    this.socket.onerror = (event) => {
      this.handleError(event);
    };

    this.socket.onmessage = (event) => {
      this.handleMessage(event);
    };
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnect(): void {
    this.cancelReconnect();
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
    this.setConnectionState('disconnected');
  }

  /**
   * Subscribe to a pipeline's events
   */
  subscribe(pipelineId: string): void {
    this.subscriptions.add(pipelineId);

    if (this.socket?.readyState === WebSocket.OPEN) {
      this.sendSubscribe(pipelineId);
    } else {
      this.pendingSubscriptions.add(pipelineId);
    }
  }

  /**
   * Unsubscribe from a pipeline's events
   */
  unsubscribe(pipelineId: string): void {
    this.subscriptions.delete(pipelineId);
    this.pendingSubscriptions.delete(pipelineId);

    if (this.socket?.readyState === WebSocket.OPEN) {
      this.sendUnsubscribe(pipelineId);
    }
  }

  /**
   * Unsubscribe from all pipelines
   */
  unsubscribeAll(): void {
    for (const pipelineId of this.subscriptions) {
      this.unsubscribe(pipelineId);
    }
  }

  /**
   * Handle WebSocket open event
   */
  private handleOpen(): void {
    this.setConnectionState('connected');
    this.reconnectDelay = INITIAL_RECONNECT_DELAY;

    // Send pending subscriptions
    for (const pipelineId of this.pendingSubscriptions) {
      this.sendSubscribe(pipelineId);
    }
    this.pendingSubscriptions.clear();

    // Re-subscribe to existing subscriptions (after reconnect)
    for (const pipelineId of this.subscriptions) {
      this.sendSubscribe(pipelineId);
    }

    this.handlers.onConnect?.();
  }

  /**
   * Handle WebSocket close event
   */
  private handleClose(): void {
    this.socket = null;

    if (this.connectionState !== 'disconnected') {
      this.setConnectionState('reconnecting');
      this.scheduleReconnect();
    }

    this.handlers.onDisconnect?.();
  }

  /**
   * Handle WebSocket error event
   */
  private handleError(_event: Event): void {
    // Error handling is done in onclose
  }

  /**
   * Handle incoming WebSocket message
   */
  private handleMessage(event: MessageEvent): void {
    try {
      const message = JSON.parse(event.data) as WebSocketMessage;

      switch (message.event) {
        case 'health':
          this.handlers.onHealth?.(message as HealthMessage);
          break;
        case 'error':
          this.handlers.onError?.(message as ErrorMessage);
          break;
        case 'status':
          this.handlers.onStatus?.(message as StatusMessage);
          break;
        case 'subscribed':
        case 'unsubscribed':
          // Confirmation messages, no action needed
          break;
        case 'error_message':
          // eslint-disable-next-line no-console
          console.warn('WebSocket error:', message.data);
          break;
      }
    } catch {
      // eslint-disable-next-line no-console
      console.warn('Failed to parse WebSocket message');
    }
  }

  /**
   * Send subscribe message
   */
  private sendSubscribe(pipelineId: string): void {
    this.send({ event: 'subscribe', pipelineId });
  }

  /**
   * Send unsubscribe message
   */
  private sendUnsubscribe(pipelineId: string): void {
    this.send({ event: 'unsubscribe', pipelineId });
  }

  /**
   * Send message to server
   */
  private send(message: Partial<WebSocketMessage>): void {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(message));
    }
  }

  /**
   * Schedule reconnection attempt
   */
  private scheduleReconnect(): void {
    this.cancelReconnect();

    this.reconnectTimer = window.setTimeout(() => {
      this.connect();
      this.reconnectDelay = Math.min(
        this.reconnectDelay * RECONNECT_MULTIPLIER,
        MAX_RECONNECT_DELAY
      );
    }, this.reconnectDelay);
  }

  /**
   * Cancel scheduled reconnection
   */
  private cancelReconnect(): void {
    if (this.reconnectTimer !== null) {
      window.clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  /**
   * Update and broadcast connection state
   */
  private setConnectionState(state: ConnectionState): void {
    this.connectionState = state;
    this.handlers.onConnectionStateChange?.(state);
  }
}

// Singleton instance
let wsClientInstance: WebSocketClient | null = null;

/**
 * Get the singleton WebSocket client instance
 */
export function getWebSocketClient(): WebSocketClient {
  if (!wsClientInstance) {
    wsClientInstance = new WebSocketClient();
  }
  return wsClientInstance;
}

/**
 * Reset the WebSocket client (for testing)
 */
export function resetWebSocketClient(): void {
  if (wsClientInstance) {
    wsClientInstance.disconnect();
    wsClientInstance = null;
  }
}

export default WebSocketClient;
