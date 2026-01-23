/**
 * Runtime Store
 *
 * Manages pipeline runtime state including:
 * - Pipeline ID and status
 * - Module metrics (FPS, queue length)
 * - Runtime errors
 * - WebSocket connection
 */

import { create } from 'zustand';
import { getWebSocketClient } from '../services/websocket';
import type {
  PipelineStatus,
  ModuleMetrics,
  RuntimeError,
  ConnectionState,
  HealthMessage,
  ErrorMessage,
  StatusMessage,
} from '../types/runtime';

const API_BASE = 'http://localhost:3000';

/**
 * Runtime store state
 */
interface RuntimeState {
  /** Current pipeline ID */
  pipelineId: string | null;
  /** Pipeline execution status */
  status: PipelineStatus;
  /** Module metrics keyed by module ID */
  moduleMetrics: Record<string, ModuleMetrics>;
  /** Runtime errors */
  errors: RuntimeError[];
  /** WebSocket connection state */
  connectionState: ConnectionState;
  /** Pipeline start time */
  startTime: number | null;
  /** Is an operation in progress */
  isLoading: boolean;
}

/**
 * Runtime store actions
 */
interface RuntimeActions {
  // Pipeline lifecycle
  createPipeline: (config: unknown) => Promise<string>;
  startPipeline: () => Promise<void>;
  stopPipeline: () => Promise<void>;
  deletePipeline: () => Promise<void>;

  // WebSocket connection
  connect: () => void;
  disconnect: () => void;

  // Event handlers
  onHealthEvent: (message: HealthMessage) => void;
  onErrorEvent: (message: ErrorMessage) => void;
  onStatusEvent: (message: StatusMessage) => void;
  onConnectionStateChange: (state: ConnectionState) => void;

  // Utilities
  clearErrors: () => void;
  reset: () => void;
  getDuration: () => number | null;
}

const initialState: RuntimeState = {
  pipelineId: null,
  status: 'IDLE',
  moduleMetrics: {},
  errors: [],
  connectionState: 'disconnected',
  startTime: null,
  isLoading: false,
};

/**
 * Runtime store for managing pipeline execution
 */
export const useRuntimeStore = create<RuntimeState & RuntimeActions>((set, get) => {
  // Set up WebSocket handlers
  const wsClient = getWebSocketClient();

  wsClient.setHandlers({
    onConnect: () => {
      // Subscribe to current pipeline if we have one
      const { pipelineId } = get();
      if (pipelineId) {
        wsClient.subscribe(pipelineId);
      }
    },
    onHealth: (message) => get().onHealthEvent(message),
    onError: (message) => get().onErrorEvent(message),
    onStatus: (message) => get().onStatusEvent(message),
    onConnectionStateChange: (state) => get().onConnectionStateChange(state),
  });

  return {
    ...initialState,

    createPipeline: async (config) => {
      set({ isLoading: true });

      try {
        const response = await fetch(`${API_BASE}/api/pipeline/create`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(config),
        });

        if (!response.ok) {
          throw new Error(`Failed to create pipeline: ${response.statusText}`);
        }

        const data = await response.json();
        const pipelineId = data.pipelineId;

        // Initialize metrics for all modules
        const moduleMetrics: Record<string, ModuleMetrics> = {};
        if (typeof config === 'object' && config !== null && 'modules' in config) {
          const modules = (config as { modules: Record<string, unknown> }).modules;
          for (const moduleId of Object.keys(modules)) {
            moduleMetrics[moduleId] = {
              fps: 0,
              qlen: 0,
              isQueueFull: false,
              timestamp: Date.now(),
            };
          }
        }

        set({
          pipelineId,
          status: 'IDLE',
          moduleMetrics,
          errors: [],
          isLoading: false,
        });

        // Subscribe to pipeline events via WebSocket
        wsClient.subscribe(pipelineId);

        return pipelineId;
      } catch (error) {
        set({ isLoading: false });
        throw error;
      }
    },

    startPipeline: async () => {
      const { pipelineId } = get();
      if (!pipelineId) {
        throw new Error('No pipeline created');
      }

      set({ isLoading: true });

      try {
        const response = await fetch(`${API_BASE}/api/pipeline/${pipelineId}/start`, {
          method: 'POST',
        });

        if (!response.ok) {
          const data = await response.json();
          throw new Error(data.message || 'Failed to start pipeline');
        }

        set({
          status: 'RUNNING',
          startTime: Date.now(),
          isLoading: false,
        });
      } catch (error) {
        set({ isLoading: false });
        throw error;
      }
    },

    stopPipeline: async () => {
      const { pipelineId } = get();
      if (!pipelineId) {
        throw new Error('No pipeline created');
      }

      set({ isLoading: true });

      try {
        const response = await fetch(`${API_BASE}/api/pipeline/${pipelineId}/stop`, {
          method: 'POST',
        });

        if (!response.ok) {
          const data = await response.json();
          throw new Error(data.message || 'Failed to stop pipeline');
        }

        set({
          status: 'STOPPED',
          isLoading: false,
        });
      } catch (error) {
        set({ isLoading: false });
        throw error;
      }
    },

    deletePipeline: async () => {
      const { pipelineId } = get();
      if (!pipelineId) {
        return;
      }

      set({ isLoading: true });

      try {
        // Unsubscribe from WebSocket
        wsClient.unsubscribe(pipelineId);

        const response = await fetch(`${API_BASE}/api/pipeline/${pipelineId}`, {
          method: 'DELETE',
        });

        if (!response.ok) {
          const data = await response.json();
          throw new Error(data.message || 'Failed to delete pipeline');
        }

        set({
          pipelineId: null,
          status: 'IDLE',
          moduleMetrics: {},
          errors: [],
          startTime: null,
          isLoading: false,
        });
      } catch (error) {
        set({ isLoading: false });
        throw error;
      }
    },

    connect: () => {
      wsClient.connect();
    },

    disconnect: () => {
      wsClient.disconnect();
    },

    onHealthEvent: (message) => {
      const { pipelineId: currentId } = get();
      if (message.pipelineId !== currentId) {
        return;
      }

      const { moduleId, fps, qlen, isQueueFull } = message.data;

      set((state) => ({
        moduleMetrics: {
          ...state.moduleMetrics,
          [moduleId]: {
            fps,
            qlen,
            isQueueFull,
            timestamp: Date.now(),
          },
        },
      }));
    },

    onErrorEvent: (message) => {
      const { pipelineId: currentId } = get();
      if (message.pipelineId !== currentId) {
        return;
      }

      const { moduleId, message: errorMessage, code } = message.data;

      const error: RuntimeError = {
        moduleId,
        message: errorMessage,
        timestamp: Date.now(),
        code,
      };

      set((state) => ({
        errors: [...state.errors, error],
      }));
    },

    onStatusEvent: (message) => {
      const { pipelineId: currentId } = get();
      if (message.pipelineId !== currentId) {
        return;
      }

      const { status } = message.data;

      set((state) => ({
        status,
        // Update startTime if pipeline is starting
        startTime: status === 'RUNNING' && !state.startTime ? Date.now() : state.startTime,
      }));
    },

    onConnectionStateChange: (connectionState) => {
      set({ connectionState });
    },

    clearErrors: () => {
      set({ errors: [] });
    },

    reset: () => {
      const { pipelineId } = get();
      if (pipelineId) {
        wsClient.unsubscribe(pipelineId);
      }
      set(initialState);
    },

    getDuration: () => {
      const { startTime, status } = get();
      if (!startTime || status !== 'RUNNING') {
        return null;
      }
      return Date.now() - startTime;
    },
  };
});

export default useRuntimeStore;
