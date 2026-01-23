/**
 * Metrics Stream WebSocket Service
 *
 * Provides real-time pipeline metrics to connected clients via WebSocket.
 * Clients subscribe to specific pipelines and receive health/error events.
 */

import { WebSocketServer, WebSocket } from 'ws';
import type { Server } from 'http';
import { createLogger } from '../utils/logger.js';
import { getPipelineManager } from '../services/PipelineManager.js';
import type { HealthEvent, ErrorEvent, PipelineStatus } from '../types/pipeline.js';

const logger = createLogger('MetricsStream');

/**
 * WebSocket message types
 */
export type WebSocketMessageType = 'subscribe' | 'unsubscribe' | 'health' | 'error' | 'status' | 'subscribed' | 'unsubscribed' | 'error_message';

/**
 * Base message structure
 */
export interface WebSocketMessage {
  event: WebSocketMessageType;
  pipelineId?: string;
  data?: unknown;
}

/**
 * Subscribe message from client
 */
export interface SubscribeMessage extends WebSocketMessage {
  event: 'subscribe';
  pipelineId: string;
}

/**
 * Unsubscribe message from client
 */
export interface UnsubscribeMessage extends WebSocketMessage {
  event: 'unsubscribe';
  pipelineId: string;
}

/**
 * Health event message to client
 */
export interface HealthMessage extends WebSocketMessage {
  event: 'health';
  pipelineId: string;
  data: HealthEvent;
}

/**
 * Error event message to client
 */
export interface ErrorMessage extends WebSocketMessage {
  event: 'error';
  pipelineId: string;
  data: ErrorEvent;
}

/**
 * Status event message to client
 */
export interface StatusMessage extends WebSocketMessage {
  event: 'status';
  pipelineId: string;
  data: { status: PipelineStatus };
}

/**
 * Client connection with subscriptions
 */
interface ClientConnection {
  socket: WebSocket;
  subscriptions: Set<string>;
}

/* eslint-disable @typescript-eslint/no-explicit-any */
type EventHandler = (...args: any[]) => void;
/* eslint-enable @typescript-eslint/no-explicit-any */

/**
 * MetricsStream class manages WebSocket connections and broadcasts
 */
export class MetricsStream {
  private wss: WebSocketServer | null = null;
  private clients: Map<WebSocket, ClientConnection> = new Map();
  private pipelineSubscribers: Map<string, Set<WebSocket>> = new Map();
  private eventHandlers: Map<string, EventHandler> = new Map();

  /**
   * Initialize WebSocket server and attach to HTTP server
   */
  initialize(server: Server): void {
    this.wss = new WebSocketServer({ server, path: '/ws' });

    this.wss.on('connection', (socket: WebSocket) => {
      this.handleConnection(socket);
    });

    // Subscribe to PipelineManager events
    const manager = getPipelineManager();

    const healthHandler = (event: { pipelineId: string } & HealthEvent) => {
      this.broadcastHealth(event.pipelineId, event);
    };
    manager.on('health', healthHandler);
    this.eventHandlers.set('health', healthHandler);

    const errorHandler = (event: { pipelineId: string } & ErrorEvent) => {
      this.broadcastError(event.pipelineId, event);
    };
    manager.on('error', errorHandler);
    this.eventHandlers.set('error', errorHandler);

    const statusHandler = (event: { pipelineId: string; status: PipelineStatus }) => {
      this.broadcastStatus(event.pipelineId, event.status);
    };
    manager.on('status', statusHandler);
    this.eventHandlers.set('status', statusHandler);

    logger.info('MetricsStream WebSocket server initialized on /ws');
  }

  /**
   * Handle new WebSocket connection
   */
  private handleConnection(socket: WebSocket): void {
    const connection: ClientConnection = {
      socket,
      subscriptions: new Set(),
    };
    this.clients.set(socket, connection);

    logger.info(`Client connected. Total clients: ${this.clients.size}`);

    socket.on('message', (data: Buffer) => {
      try {
        const message = JSON.parse(data.toString()) as WebSocketMessage;
        this.handleMessage(socket, message);
      } catch (error) {
        logger.warn('Invalid WebSocket message:', error);
        this.sendError(socket, 'Invalid message format');
      }
    });

    socket.on('close', () => {
      this.handleDisconnect(socket);
    });

    socket.on('error', (error) => {
      logger.error('WebSocket error:', error);
    });
  }

  /**
   * Handle incoming message from client
   */
  private handleMessage(socket: WebSocket, message: WebSocketMessage): void {
    switch (message.event) {
      case 'subscribe':
        this.handleSubscribe(socket, message as SubscribeMessage);
        break;
      case 'unsubscribe':
        this.handleUnsubscribe(socket, message as UnsubscribeMessage);
        break;
      default:
        logger.warn(`Unknown message event: ${message.event}`);
        this.sendError(socket, `Unknown event: ${message.event}`);
    }
  }

  /**
   * Handle subscribe request
   */
  private handleSubscribe(socket: WebSocket, message: SubscribeMessage): void {
    const { pipelineId } = message;

    if (!pipelineId) {
      this.sendError(socket, 'Missing pipelineId');
      return;
    }

    const connection = this.clients.get(socket);
    if (!connection) {
      return;
    }

    // Add to client subscriptions
    connection.subscriptions.add(pipelineId);

    // Add to pipeline subscribers
    if (!this.pipelineSubscribers.has(pipelineId)) {
      this.pipelineSubscribers.set(pipelineId, new Set());
    }
    this.pipelineSubscribers.get(pipelineId)!.add(socket);

    // Send confirmation
    this.send(socket, {
      event: 'subscribed',
      pipelineId,
    });

    logger.info(`Client subscribed to pipeline: ${pipelineId}`);
  }

  /**
   * Handle unsubscribe request
   */
  private handleUnsubscribe(socket: WebSocket, message: UnsubscribeMessage): void {
    const { pipelineId } = message;

    if (!pipelineId) {
      this.sendError(socket, 'Missing pipelineId');
      return;
    }

    const connection = this.clients.get(socket);
    if (!connection) {
      return;
    }

    // Remove from client subscriptions
    connection.subscriptions.delete(pipelineId);

    // Remove from pipeline subscribers
    const subscribers = this.pipelineSubscribers.get(pipelineId);
    if (subscribers) {
      subscribers.delete(socket);
      if (subscribers.size === 0) {
        this.pipelineSubscribers.delete(pipelineId);
      }
    }

    // Send confirmation
    this.send(socket, {
      event: 'unsubscribed',
      pipelineId,
    });

    logger.info(`Client unsubscribed from pipeline: ${pipelineId}`);
  }

  /**
   * Handle client disconnect
   */
  private handleDisconnect(socket: WebSocket): void {
    const connection = this.clients.get(socket);
    if (connection) {
      // Remove from all pipeline subscriptions
      for (const pipelineId of connection.subscriptions) {
        const subscribers = this.pipelineSubscribers.get(pipelineId);
        if (subscribers) {
          subscribers.delete(socket);
          if (subscribers.size === 0) {
            this.pipelineSubscribers.delete(pipelineId);
          }
        }
      }
    }

    this.clients.delete(socket);
    logger.info(`Client disconnected. Total clients: ${this.clients.size}`);
  }

  /**
   * Broadcast health event to pipeline subscribers
   */
  private broadcastHealth(pipelineId: string, event: HealthEvent): void {
    const subscribers = this.pipelineSubscribers.get(pipelineId);
    if (!subscribers || subscribers.size === 0) {
      return;
    }

    const message: HealthMessage = {
      event: 'health',
      pipelineId,
      data: event,
    };

    this.broadcast(subscribers, message);
  }

  /**
   * Broadcast error event to pipeline subscribers
   */
  private broadcastError(pipelineId: string, event: ErrorEvent): void {
    const subscribers = this.pipelineSubscribers.get(pipelineId);
    if (!subscribers || subscribers.size === 0) {
      return;
    }

    const message: ErrorMessage = {
      event: 'error',
      pipelineId,
      data: event,
    };

    this.broadcast(subscribers, message);
  }

  /**
   * Broadcast status change to pipeline subscribers
   */
  private broadcastStatus(pipelineId: string, status: PipelineStatus): void {
    const subscribers = this.pipelineSubscribers.get(pipelineId);
    if (!subscribers || subscribers.size === 0) {
      return;
    }

    const message: StatusMessage = {
      event: 'status',
      pipelineId,
      data: { status },
    };

    this.broadcast(subscribers, message);
  }

  /**
   * Broadcast message to set of sockets
   */
  private broadcast(sockets: Set<WebSocket>, message: WebSocketMessage): void {
    const data = JSON.stringify(message);
    for (const socket of sockets) {
      if (socket.readyState === WebSocket.OPEN) {
        socket.send(data);
      }
    }
  }

  /**
   * Send message to single socket
   */
  private send(socket: WebSocket, message: WebSocketMessage): void {
    if (socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(message));
    }
  }

  /**
   * Send error message to socket
   */
  private sendError(socket: WebSocket, error: string): void {
    this.send(socket, {
      event: 'error_message',
      data: { error },
    });
  }

  /**
   * Close WebSocket server and cleanup
   */
  close(): void {
    // Remove event handlers from PipelineManager
    const manager = getPipelineManager();
    for (const [eventName, handler] of this.eventHandlers) {
      manager.off(eventName, handler);
    }
    this.eventHandlers.clear();

    // Close all client connections
    for (const [socket] of this.clients) {
      socket.close();
    }
    this.clients.clear();
    this.pipelineSubscribers.clear();

    // Close WebSocket server
    if (this.wss) {
      this.wss.close();
      this.wss = null;
    }

    logger.info('MetricsStream closed');
  }

  /**
   * Get number of connected clients
   */
  getClientCount(): number {
    return this.clients.size;
  }

  /**
   * Get number of subscribers for a pipeline
   */
  getSubscriberCount(pipelineId: string): number {
    return this.pipelineSubscribers.get(pipelineId)?.size || 0;
  }
}

// Singleton instance
let metricsStreamInstance: MetricsStream | null = null;

/**
 * Get the singleton MetricsStream instance
 */
export function getMetricsStream(): MetricsStream {
  if (!metricsStreamInstance) {
    metricsStreamInstance = new MetricsStream();
  }
  return metricsStreamInstance;
}

/**
 * Reset the metrics stream (for testing)
 */
export function resetMetricsStream(): void {
  if (metricsStreamInstance) {
    metricsStreamInstance.close();
    metricsStreamInstance = null;
  }
}

export default MetricsStream;
