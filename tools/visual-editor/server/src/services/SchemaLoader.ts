import fs from 'fs/promises';
import path from 'path';
import { config } from '../config.js';
import { createLogger } from '../utils/logger.js';

const logger = createLogger('SchemaLoader');

export interface Pin {
  name: string;
  frame_types: string[];
}

export interface PropertySchema {
  type: 'int' | 'float' | 'bool' | 'string' | 'enum' | 'json';
  default?: string;
  min?: string;
  max?: string;
  enum_values?: string[];
  description?: string;
}

export interface ModuleSchema {
  category: string;
  description: string;
  inputs: Pin[];
  outputs: Pin[];
  properties: Record<string, PropertySchema>;
}

export interface Schema {
  modules: Record<string, ModuleSchema>;
}

/**
 * Mock schema data for development when modules.json is not available
 */
const MOCK_SCHEMA: Schema = {
  modules: {
    TestSignalGenerator: {
      category: 'source',
      description: 'Generates test video signals for debugging',
      inputs: [],
      outputs: [{ name: 'output', frame_types: ['RAW_IMAGE'] }],
      properties: {
        width: { type: 'int', default: '640', min: '1', max: '4096' },
        height: { type: 'int', default: '480', min: '1', max: '4096' },
        fps: { type: 'int', default: '30', min: '1', max: '120' },
      },
    },
    FileReaderModule: {
      category: 'source',
      description: 'Reads frames from image files',
      inputs: [],
      outputs: [{ name: 'output', frame_types: ['RAW_IMAGE', 'ENCODED_IMAGE'] }],
      properties: {
        strFullFileNameWithPattern: {
          type: 'string',
          description: 'File path pattern (e.g., frame_*.jpg)',
        },
        loop: { type: 'bool', default: 'false' },
      },
    },
    FileWriterModule: {
      category: 'sink',
      description: 'Writes frames to image files',
      inputs: [{ name: 'input', frame_types: ['RAW_IMAGE', 'ENCODED_IMAGE'] }],
      outputs: [],
      properties: {
        strFullFileNameWithPattern: {
          type: 'string',
          description: 'Output file path pattern',
        },
      },
    },
    ResizeModule: {
      category: 'transform',
      description: 'Resizes images to specified dimensions',
      inputs: [{ name: 'input', frame_types: ['RAW_IMAGE'] }],
      outputs: [{ name: 'output', frame_types: ['RAW_IMAGE'] }],
      properties: {
        width: { type: 'int', default: '640', min: '1', max: '8192' },
        height: { type: 'int', default: '480', min: '1', max: '8192' },
      },
    },
    ColorConversionModule: {
      category: 'transform',
      description: 'Converts between color formats',
      inputs: [{ name: 'input', frame_types: ['RAW_IMAGE'] }],
      outputs: [{ name: 'output', frame_types: ['RAW_IMAGE'] }],
      properties: {
        conversionType: {
          type: 'enum',
          enum_values: ['BGR_TO_RGB', 'RGB_TO_BGR', 'BGR_TO_GRAY', 'RGB_TO_GRAY'],
          default: 'BGR_TO_RGB',
        },
      },
    },
  },
};

/**
 * Service for loading and caching the module schema
 */
class SchemaLoader {
  private schema: Schema | null = null;
  private schemaPath: string | null = null;

  constructor() {
    this.schemaPath = this.findSchemaPath();
  }

  /**
   * Find the schema file in expected locations
   */
  private findSchemaPath(): string | null {
    // Environment override takes precedence
    if (config.schemaPath) {
      return config.schemaPath;
    }

    // Search common locations
    const candidates = [
      './modules.json',
      '../modules.json',
      '../data/modules.json',
      '../../data/modules.json',
      '../../../build/modules.json',
      path.join(process.cwd(), 'modules.json'),
    ];

    for (const candidate of candidates) {
      try {
        // Use synchronous check for initialization
        const resolved = path.resolve(candidate);
        // We'll check existence later during load
        return resolved;
      } catch {
        // Continue to next candidate
      }
    }

    return null;
  }

  /**
   * Load the schema, with fallback to mock data
   */
  async load(): Promise<Schema> {
    if (this.schema) {
      return this.schema;
    }

    // Try to load from file
    if (this.schemaPath) {
      try {
        const content = await fs.readFile(this.schemaPath, 'utf-8');
        this.schema = JSON.parse(content) as Schema;
        logger.info(`Loaded schema from ${this.schemaPath}`);
        return this.schema;
      } catch (err) {
        logger.warn(`Could not load schema from ${this.schemaPath}: ${err}`);
      }
    }

    // Fallback to mock data
    logger.warn('Using mock schema data (modules.json not found)');
    this.schema = MOCK_SCHEMA;
    return this.schema;
  }

  /**
   * Force reload the schema from disk
   */
  async reload(): Promise<Schema> {
    this.schema = null;
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
}

// Export class for testing and direct instantiation
export { SchemaLoader };

// Export singleton instance
const schemaLoader = new SchemaLoader();

export function getSchemaLoader(): SchemaLoader {
  return schemaLoader;
}

export { schemaLoader };
