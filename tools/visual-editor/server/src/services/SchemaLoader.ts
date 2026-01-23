/**
 * SchemaLoader Service
 *
 * Loads module and frame type schemas from the aprapipes.node addon.
 * Falls back to schema.json when the addon is not available.
 */

import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { createLogger } from '../utils/logger.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const logger = createLogger('SchemaLoader');

// ============================================================
// Type definitions matching the client-side expected format
// ============================================================

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

export interface Schema {
  modules: Record<string, ModuleSchema>;
  frameTypes: Record<string, FrameTypeSchema>;
}

// ============================================================
// Types from the aprapipes.node addon
// ============================================================

interface AddonPropertyInfo {
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

interface AddonPinInfo {
  name: string;
  required?: boolean;
  frameTypes: string[];
}

interface AddonModuleInfo {
  name: string;
  description: string;
  version: string;
  category: string;
  tags: string[];
  properties: AddonPropertyInfo[];
  inputs: AddonPinInfo[];
  outputs: AddonPinInfo[];
}

interface AddonFrameTypeInfo {
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

interface AddonSchema {
  modules: Record<string, AddonModuleInfo>;
  frameTypes: Record<string, AddonFrameTypeInfo>;
}

// ============================================================
// Schema.json fallback loading
// ============================================================

/**
 * Load schema from the schema.json file
 */
function loadSchemaFromFile(): Schema | null {
  // Try multiple possible locations for schema.json
  const schemaPaths = [
    path.resolve(process.cwd(), 'schema.json'),
    path.resolve(process.cwd(), '..', 'schema.json'),
    path.resolve(__dirname, '..', '..', '..', 'schema.json'),
    path.resolve(__dirname, '..', '..', '..', '..', 'schema.json'),
  ];

  for (const schemaPath of schemaPaths) {
    try {
      if (fs.existsSync(schemaPath)) {
        const content = fs.readFileSync(schemaPath, 'utf-8');
        const data = JSON.parse(content) as AddonSchema;
        logger.info(`Loaded schema from ${schemaPath}`);
        return transformSchema(data);
      }
    } catch (err) {
      logger.warn(`Failed to load schema from ${schemaPath}: ${err}`);
    }
  }

  return null;
}

// ============================================================
// Addon loading
// ============================================================

let aprapipesAddon: {
  describeAllModules: () => AddonSchema;
  validatePipeline: (config: string | object) => { valid: boolean; issues: unknown[] };
} | null = null;

function loadAddon(): boolean {
  if (aprapipesAddon !== null) {
    return true;
  }

  // Try to load the addon from known locations
  const addonPaths = [
    path.resolve(process.cwd(), 'aprapipes.node'),
    path.resolve(process.cwd(), '..', 'aprapipes.node'),
    path.resolve(process.cwd(), '..', '..', 'aprapipes.node'),
    path.resolve(process.cwd(), '..', '..', '..', 'aprapipes.node'),
    path.resolve(__dirname, '..', '..', '..', '..', '..', 'aprapipes.node'),
  ];

  for (const addonPath of addonPaths) {
    try {
      // Dynamic require for native addon
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      aprapipesAddon = require(addonPath);
      logger.info(`Loaded aprapipes.node from ${addonPath}`);
      return true;
    } catch {
      // Continue to next path
    }
  }

  logger.warn('Could not load aprapipes.node addon, will try schema.json fallback');
  return false;
}

/**
 * Transform addon module format to client-expected format
 */
function transformModuleSchema(addonModule: AddonModuleInfo): ModuleSchema {
  // Convert properties from array to object
  const properties: Record<string, PropertySchema> = {};
  for (const prop of addonModule.properties) {
    // Map addon type to our type
    let type: PropertySchema['type'] = 'string';
    switch (prop.type) {
      case 'int':
        type = 'int';
        break;
      case 'float':
        type = 'float';
        break;
      case 'bool':
        type = 'bool';
        break;
      case 'string':
        type = 'string';
        break;
      case 'enum':
        type = 'enum';
        break;
      case 'json':
        type = 'json';
        break;
      default:
        type = 'string';
    }

    properties[prop.name] = {
      type,
      default: prop.default,
      min: prop.min,
      max: prop.max,
      enum_values: prop.enumValues,
      description: prop.description,
      required: prop.required,
      mutability: prop.mutability,
    };
  }

  // Convert pins (frameTypes -> frame_types)
  const inputs: Pin[] = addonModule.inputs.map((pin) => ({
    name: pin.name,
    frame_types: pin.frameTypes,
    required: pin.required,
  }));

  const outputs: Pin[] = addonModule.outputs.map((pin) => ({
    name: pin.name,
    frame_types: pin.frameTypes,
  }));

  return {
    category: addonModule.category,
    description: addonModule.description,
    version: addonModule.version,
    tags: addonModule.tags,
    inputs,
    outputs,
    properties,
  };
}

/**
 * Transform full addon schema to client-expected format
 */
function transformSchema(addonSchema: AddonSchema): Schema {
  const modules: Record<string, ModuleSchema> = {};
  for (const [name, module] of Object.entries(addonSchema.modules)) {
    modules[name] = transformModuleSchema(module);
  }

  // Frame types don't need transformation (already in the right format)
  const frameTypes: Record<string, FrameTypeSchema> = addonSchema.frameTypes || {};

  return { modules, frameTypes };
}

// ============================================================
// SchemaLoader class
// ============================================================

/**
 * Service for loading and caching the module schema
 */
class SchemaLoader {
  private schema: Schema | null = null;
  private addonLoaded: boolean = false;

  constructor() {
    this.addonLoaded = loadAddon();
  }

  /**
   * Load the schema from addon or fallback to mock
   */
  async load(): Promise<Schema> {
    if (this.schema) {
      return this.schema;
    }

    // Try to load from addon
    if (this.addonLoaded && aprapipesAddon) {
      try {
        const addonSchema = aprapipesAddon.describeAllModules();
        this.schema = transformSchema(addonSchema);
        logger.info(`Loaded schema from aprapipes.node: ${Object.keys(this.schema.modules).length} modules, ${Object.keys(this.schema.frameTypes).length} frame types`);
        return this.schema;
      } catch (err) {
        logger.error(`Error loading schema from addon: ${err}`);
      }
    }

    // Fallback to schema.json file
    const fileSchema = loadSchemaFromFile();
    if (fileSchema) {
      this.schema = fileSchema;
      logger.info(`Loaded schema from file: ${Object.keys(this.schema.modules).length} modules, ${Object.keys(this.schema.frameTypes).length} frame types`);
      return this.schema;
    }

    // Ultimate fallback - empty schema
    logger.error('No schema available - neither addon nor schema.json found');
    this.schema = { modules: {}, frameTypes: {} };
    return this.schema;
  }

  /**
   * Force reload the schema
   */
  async reload(): Promise<Schema> {
    this.schema = null;
    // Try reloading the addon
    this.addonLoaded = loadAddon();
    return this.load();
  }

  /**
   * Get a specific module schema
   */
  async getModule(name: string): Promise<ModuleSchema | undefined> {
    const schema = await this.load();
    return schema.modules[name];
  }

  /**
   * Get the full schema as a flat map of module names to schemas
   */
  async getSchema(): Promise<Record<string, ModuleSchema>> {
    const schema = await this.load();
    return schema.modules;
  }

  /**
   * Get all frame types
   */
  async getFrameTypes(): Promise<Record<string, FrameTypeSchema>> {
    const schema = await this.load();
    return schema.frameTypes;
  }

  /**
   * Get the full schema including frameTypes
   */
  async getFullSchema(): Promise<Schema> {
    return this.load();
  }

  /**
   * Check if addon is loaded
   */
  isAddonLoaded(): boolean {
    return this.addonLoaded;
  }

  /**
   * Check if two frame types are compatible (considering hierarchy)
   */
  async areFrameTypesCompatible(outputType: string, inputType: string): Promise<boolean> {
    // Exact match
    if (outputType === inputType) {
      return true;
    }

    // Generic "Frame" accepts anything
    if (inputType === 'Frame') {
      return true;
    }

    // Check if outputType is a subtype of inputType
    const frameTypes = await this.getFrameTypes();
    const outputInfo = frameTypes[outputType];

    if (outputInfo?.ancestors) {
      return outputInfo.ancestors.includes(inputType);
    }

    return false;
  }
}

// Export class for testing and direct instantiation
export { SchemaLoader };

// Export singleton instance
const schemaLoader = new SchemaLoader();

export function getSchemaLoader(): SchemaLoader {
  return schemaLoader;
}

export { schemaLoader };

// Export addon for use in Validator
export function getAprapipesAddon() {
  if (!aprapipesAddon) {
    loadAddon();
  }
  return aprapipesAddon;
}
