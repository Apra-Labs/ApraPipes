/**
 * Pipeline configuration types
 */

export interface ModuleConfig {
  type: string;
  props?: Record<string, unknown>;
}

export interface Connection {
  from: string;
  to: string;
  sieve?: boolean;
}

export interface PipelineConfig {
  modules: Record<string, ModuleConfig>;
  connections: Connection[];
}
