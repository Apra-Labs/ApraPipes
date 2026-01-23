import { create } from 'zustand';
import type { ModuleSchema, ModuleConfig, PipelineConfig } from '../types/schema';

/**
 * Connection in the pipeline format expected by ApraPipes
 */
export interface PipelineConnection {
  from: string; // Format: "moduleId.outputPinName"
  to: string;   // Format: "moduleId.inputPinName"
}

/**
 * Pipeline store state
 */
interface PipelineState {
  config: PipelineConfig;
  schema: Record<string, ModuleSchema>;
  isDirty: boolean;
}

/**
 * Pipeline store actions
 */
interface PipelineActions {
  // Schema management
  setSchema: (schema: Record<string, ModuleSchema>) => void;

  // Module operations
  addModule: (moduleId: string, moduleType: string) => void;
  removeModule: (moduleId: string) => void;
  renameModule: (oldId: string, newId: string) => void;
  updateModuleProperty: (moduleId: string, key: string, value: unknown) => void;

  // Connection operations
  addConnection: (from: string, to: string) => void;
  removeConnection: (connectionIndex: number) => void;

  // Serialization
  toJSON: () => string;
  fromJSON: (json: string) => void;

  // State management
  reset: () => void;
  markClean: () => void;
}

const initialConfig: PipelineConfig = {
  modules: {},
  connections: [],
};

const initialState: PipelineState = {
  config: initialConfig,
  schema: {},
  isDirty: false,
};

/**
 * Pipeline store for managing the pipeline configuration
 * This store holds the serializable pipeline state that can be saved as JSON
 */
export const usePipelineStore = create<PipelineState & PipelineActions>((set, get) => ({
  ...initialState,

  setSchema: (schema) => {
    set({ schema });
  },

  addModule: (moduleId, moduleType) => {
    const { schema } = get();
    const moduleSchema = schema[moduleType];

    // Initialize properties with defaults from schema, parsing to correct types
    const properties: Record<string, unknown> = {};
    if (moduleSchema?.properties) {
      for (const [key, propDef] of Object.entries(moduleSchema.properties)) {
        if (propDef.default !== undefined) {
          // Parse default value based on type
          switch (propDef.type) {
            case 'int':
              properties[key] = parseInt(propDef.default, 10);
              break;
            case 'float':
              properties[key] = parseFloat(propDef.default);
              break;
            case 'bool':
              properties[key] = propDef.default === 'true';
              break;
            case 'json':
              try {
                properties[key] = JSON.parse(propDef.default);
              } catch {
                properties[key] = {};
              }
              break;
            default:
              properties[key] = propDef.default;
          }
        }
      }
    }

    const moduleConfig: ModuleConfig = {
      type: moduleType,
      properties,
    };

    set((state) => ({
      config: {
        ...state.config,
        modules: {
          ...state.config.modules,
          [moduleId]: moduleConfig,
        },
      },
      isDirty: true,
    }));
  },

  removeModule: (moduleId) => {
    set((state) => {
      // Remove module
      const { [moduleId]: _removed, ...remainingModules } = state.config.modules;

      // Remove connections involving this module
      const filteredConnections = state.config.connections.filter(
        (conn) => !conn.from.startsWith(`${moduleId}.`) && !conn.to.startsWith(`${moduleId}.`)
      );

      return {
        config: {
          modules: remainingModules,
          connections: filteredConnections,
        },
        isDirty: true,
      };
    });
  },

  renameModule: (oldId, newId) => {
    if (oldId === newId) return;

    set((state) => {
      const moduleConfig = state.config.modules[oldId];
      if (!moduleConfig) return state;

      // Remove old key, add new key
      const { [oldId]: _oldModule, ...remainingModules } = state.config.modules;
      const updatedModules = {
        ...remainingModules,
        [newId]: moduleConfig,
      };

      // Update connections
      const updatedConnections = state.config.connections.map((conn) => ({
        from: conn.from.startsWith(`${oldId}.`)
          ? `${newId}.${conn.from.slice(oldId.length + 1)}`
          : conn.from,
        to: conn.to.startsWith(`${oldId}.`)
          ? `${newId}.${conn.to.slice(oldId.length + 1)}`
          : conn.to,
      }));

      return {
        config: {
          modules: updatedModules,
          connections: updatedConnections,
        },
        isDirty: true,
      };
    });
  },

  updateModuleProperty: (moduleId, key, value) => {
    set((state) => {
      const moduleConfig = state.config.modules[moduleId];
      if (!moduleConfig) return state;

      return {
        config: {
          ...state.config,
          modules: {
            ...state.config.modules,
            [moduleId]: {
              ...moduleConfig,
              properties: {
                ...moduleConfig.properties,
                [key]: value,
              },
            },
          },
        },
        isDirty: true,
      };
    });
  },

  addConnection: (from, to) => {
    // Check for duplicate connections
    const { config } = get();
    const exists = config.connections.some(
      (conn) => conn.from === from && conn.to === to
    );
    if (exists) return;

    set((state) => ({
      config: {
        ...state.config,
        connections: [...state.config.connections, { from, to }],
      },
      isDirty: true,
    }));
  },

  removeConnection: (connectionIndex) => {
    set((state) => ({
      config: {
        ...state.config,
        connections: state.config.connections.filter((_, i) => i !== connectionIndex),
      },
      isDirty: true,
    }));
  },

  toJSON: () => {
    const { config } = get();
    return JSON.stringify(config, null, 2);
  },

  fromJSON: (json) => {
    try {
      const config = JSON.parse(json) as PipelineConfig;
      set({ config, isDirty: false });
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error('Failed to parse pipeline JSON:', error);
    }
  },

  reset: () => {
    set(initialState);
  },

  markClean: () => {
    set({ isDirty: false });
  },
}));
