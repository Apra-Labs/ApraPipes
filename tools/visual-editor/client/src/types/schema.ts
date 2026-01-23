/**
 * Type definitions for module schema from aprapipes.node addon
 */

export interface Pin {
  name: string;
  frame_types: string[];
  required?: boolean;
}

export interface PropertySchema {
  type: 'int' | 'float' | 'bool' | 'string' | 'enum' | 'json';
  default?: string;
  min?: string;
  max?: string;
  enum_values?: string[];
  description?: string;
  required?: boolean;
  mutability?: string;
}

export interface ModuleSchema {
  category: string;
  description: string;
  version?: string;
  tags?: string[];
  inputs: Pin[];
  outputs: Pin[];
  properties: Record<string, PropertySchema>;
}

export interface FrameTypeSchema {
  parent: string;
  description: string;
  tags: string[];
  attributes?: Record<string, {
    type: string;
    required: boolean;
    description: string;
    enumValues?: string[];
  }>;
  ancestors?: string[];
  subtypes?: string[];
}

export interface SchemaResponse {
  modules: Record<string, ModuleSchema>;
  frameTypes: Record<string, FrameTypeSchema>;
}

/**
 * Pipeline configuration types for serialization
 */
export interface ModuleConfig {
  type: string;
  properties: Record<string, unknown>;
}

export interface PipelineConnection {
  from: string; // Format: "moduleId.outputPinName"
  to: string;   // Format: "moduleId.inputPinName"
}

export interface PipelineConfig {
  modules: Record<string, ModuleConfig>;
  connections: PipelineConnection[];
}

/**
 * Module categories for grouping in palette
 */
export type ModuleCategory = 'source' | 'transform' | 'sink' | 'cuda' | 'other';

/**
 * Get the category color class for Tailwind
 */
export function getCategoryColor(category: string): string {
  switch (category.toLowerCase()) {
    case 'source':
      return 'bg-source text-white';
    case 'transform':
      return 'bg-transform text-white';
    case 'sink':
      return 'bg-sink text-white';
    case 'cuda':
      return 'bg-purple-500 text-white';
    default:
      return 'bg-gray-500 text-white';
  }
}
