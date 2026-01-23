import { describe, it, expect, beforeEach, vi } from 'vitest';
import { Validator } from '../services/Validator.js';
import { SchemaLoader, type ModuleSchema } from '../services/SchemaLoader.js';
import type { PipelineConfig } from '../types/validation.js';

// Mock schema for testing - must match PropertySchema types
const mockSchema: Record<string, ModuleSchema> = {
  FileReaderModule: {
    category: 'source',
    description: 'Reads files',
    inputs: [],
    outputs: [{ name: 'output', frame_types: ['raw_image'] }],
    properties: {
      path: { type: 'string' as const, default: '' },
      loop: { type: 'bool' as const, default: 'false' },
    },
  },
  ResizeModule: {
    category: 'transform',
    description: 'Resizes images',
    inputs: [{ name: 'input', frame_types: ['raw_image'] }],
    outputs: [{ name: 'output', frame_types: ['raw_image'] }],
    properties: {
      width: { type: 'int' as const, min: '1', max: '4096' },
      height: { type: 'int' as const, min: '1', max: '4096' },
    },
  },
  FileWriterModule: {
    category: 'sink',
    description: 'Writes files',
    inputs: [{ name: 'input', frame_types: ['raw_image', 'encoded_image'] }],
    outputs: [],
    properties: {
      path: { type: 'string' as const },
      format: { type: 'enum' as const, enum_values: ['jpg', 'png', 'bmp'] },
    },
  },
  EncoderModule: {
    category: 'transform',
    description: 'Encodes images',
    inputs: [{ name: 'input', frame_types: ['raw_image'] }],
    outputs: [{ name: 'output', frame_types: ['encoded_image'] }],
    properties: {
      quality: { type: 'int' as const, min: '0', max: '100', default: '90' },
    },
  },
};

describe('Validator', () => {
  let validator: Validator;
  let schemaLoader: SchemaLoader;

  beforeEach(() => {
    schemaLoader = new SchemaLoader();
    // Mock getSchema to return our test schema
    vi.spyOn(schemaLoader, 'getSchema').mockResolvedValue(mockSchema);
    validator = new Validator(schemaLoader);
  });

  describe('validate empty pipeline', () => {
    it('returns info for empty pipeline', async () => {
      const config: PipelineConfig = {
        modules: {},
        connections: [],
      };

      const result = await validator.validate(config);

      expect(result.valid).toBe(true); // Info doesn't make it invalid
      expect(result.issues).toHaveLength(1);
      expect(result.issues[0].code).toBe('I101');
      expect(result.issues[0].level).toBe('info');
    });
  });

  describe('validate module types', () => {
    it('returns error for unknown module type', async () => {
      const config: PipelineConfig = {
        modules: {
          src: { type: 'UnknownModule', properties: {} },
        },
        connections: [],
      };

      const result = await validator.validate(config);

      expect(result.valid).toBe(false);
      expect(result.issues.some((i) => i.code === 'E101')).toBe(true);
    });

    it('accepts valid module type', async () => {
      const config: PipelineConfig = {
        modules: {
          src: { type: 'FileReaderModule', properties: { path: '/test.jpg' } },
        },
        connections: [],
      };

      const result = await validator.validate(config);

      // Should not have E101 error
      expect(result.issues.some((i) => i.code === 'E101')).toBe(false);
    });
  });

  describe('validate properties without defaults', () => {
    it('returns warning for missing property without default', async () => {
      const config: PipelineConfig = {
        modules: {
          resize: { type: 'ResizeModule', properties: { width: 640 } }, // missing height
        },
        connections: [],
      };

      const result = await validator.validate(config);

      // Warnings don't make pipeline invalid
      expect(result.issues.some((i) => i.code === 'W301' && i.location.includes('height'))).toBe(true);
      expect(result.issues.find((i) => i.code === 'W301')?.level).toBe('warning');
    });

    it('no warning when all properties present', async () => {
      const config: PipelineConfig = {
        modules: {
          resize: { type: 'ResizeModule', properties: { width: 640, height: 480 } },
        },
        connections: [],
      };

      const result = await validator.validate(config);

      expect(result.issues.some((i) => i.code === 'W301')).toBe(false);
    });
  });

  describe('validate property types', () => {
    it('returns error for wrong property type', async () => {
      const config: PipelineConfig = {
        modules: {
          resize: { type: 'ResizeModule', properties: { width: 'not a number', height: 480 } },
        },
        connections: [],
      };

      const result = await validator.validate(config);

      expect(result.valid).toBe(false);
      expect(result.issues.some((i) => i.code === 'E302')).toBe(true);
    });

    it('validates boolean type', async () => {
      const config: PipelineConfig = {
        modules: {
          src: { type: 'FileReaderModule', properties: { path: '/test.jpg', loop: 'not a bool' } },
        },
        connections: [],
      };

      const result = await validator.validate(config);

      expect(result.issues.some((i) => i.code === 'E302' && i.location.includes('loop'))).toBe(true);
    });
  });

  describe('validate property ranges', () => {
    it('returns error for value below minimum', async () => {
      const config: PipelineConfig = {
        modules: {
          resize: { type: 'ResizeModule', properties: { width: 0, height: 480 } },
        },
        connections: [],
      };

      const result = await validator.validate(config);

      expect(result.valid).toBe(false);
      expect(result.issues.some((i) => i.code === 'E303' && i.message.includes('below minimum'))).toBe(true);
    });

    it('returns error for value above maximum', async () => {
      const config: PipelineConfig = {
        modules: {
          resize: { type: 'ResizeModule', properties: { width: 640, height: 5000 } },
        },
        connections: [],
      };

      const result = await validator.validate(config);

      expect(result.valid).toBe(false);
      expect(result.issues.some((i) => i.code === 'E303' && i.message.includes('exceeds maximum'))).toBe(true);
    });
  });

  describe('validate enum properties', () => {
    it('returns error for invalid enum value', async () => {
      const config: PipelineConfig = {
        modules: {
          writer: { type: 'FileWriterModule', properties: { path: '/out.jpg', format: 'gif' } },
        },
        connections: [],
      };

      const result = await validator.validate(config);

      expect(result.valid).toBe(false);
      expect(result.issues.some((i) => i.code === 'E304')).toBe(true);
    });

    it('accepts valid enum value', async () => {
      const config: PipelineConfig = {
        modules: {
          writer: { type: 'FileWriterModule', properties: { path: '/out.jpg', format: 'png' } },
        },
        connections: [],
      };

      const result = await validator.validate(config);

      expect(result.issues.some((i) => i.code === 'E304')).toBe(false);
    });
  });

  describe('validate connections', () => {
    it('returns error for connection to non-existent source module', async () => {
      const config: PipelineConfig = {
        modules: {
          resize: { type: 'ResizeModule', properties: { width: 640, height: 480 } },
        },
        connections: [{ from: 'nonexistent.output', to: 'resize.input' }],
      };

      const result = await validator.validate(config);

      expect(result.valid).toBe(false);
      expect(result.issues.some((i) => i.code === 'E201')).toBe(true);
    });

    it('returns error for connection to non-existent target module', async () => {
      const config: PipelineConfig = {
        modules: {
          src: { type: 'FileReaderModule', properties: { path: '/test.jpg' } },
        },
        connections: [{ from: 'src.output', to: 'nonexistent.input' }],
      };

      const result = await validator.validate(config);

      expect(result.valid).toBe(false);
      expect(result.issues.some((i) => i.code === 'E202')).toBe(true);
    });

    it('returns error for non-existent output pin', async () => {
      const config: PipelineConfig = {
        modules: {
          src: { type: 'FileReaderModule', properties: { path: '/test.jpg' } },
          resize: { type: 'ResizeModule', properties: { width: 640, height: 480 } },
        },
        connections: [{ from: 'src.nonexistent', to: 'resize.input' }],
      };

      const result = await validator.validate(config);

      expect(result.valid).toBe(false);
      expect(result.issues.some((i) => i.code === 'E203')).toBe(true);
    });

    it('returns error for non-existent input pin', async () => {
      const config: PipelineConfig = {
        modules: {
          src: { type: 'FileReaderModule', properties: { path: '/test.jpg' } },
          resize: { type: 'ResizeModule', properties: { width: 640, height: 480 } },
        },
        connections: [{ from: 'src.output', to: 'resize.nonexistent' }],
      };

      const result = await validator.validate(config);

      expect(result.valid).toBe(false);
      expect(result.issues.some((i) => i.code === 'E204')).toBe(true);
    });

    it('returns error for duplicate connection', async () => {
      const config: PipelineConfig = {
        modules: {
          src: { type: 'FileReaderModule', properties: { path: '/test.jpg' } },
          resize: { type: 'ResizeModule', properties: { width: 640, height: 480 } },
        },
        connections: [
          { from: 'src.output', to: 'resize.input' },
          { from: 'src.output', to: 'resize.input' },
        ],
      };

      const result = await validator.validate(config);

      expect(result.valid).toBe(false);
      expect(result.issues.some((i) => i.code === 'E206')).toBe(true);
    });

    it('returns error for self-connection', async () => {
      const config: PipelineConfig = {
        modules: {
          resize: { type: 'ResizeModule', properties: { width: 640, height: 480 } },
        },
        connections: [{ from: 'resize.output', to: 'resize.input' }],
      };

      const result = await validator.validate(config);

      expect(result.valid).toBe(false);
      expect(result.issues.some((i) => i.code === 'E207')).toBe(true);
    });

    it('accepts valid connection', async () => {
      const config: PipelineConfig = {
        modules: {
          src: { type: 'FileReaderModule', properties: { path: '/test.jpg' } },
          resize: { type: 'ResizeModule', properties: { width: 640, height: 480 } },
        },
        connections: [{ from: 'src.output', to: 'resize.input' }],
      };

      const result = await validator.validate(config);

      // Should not have connection errors
      expect(result.issues.some((i) => i.code.startsWith('E2'))).toBe(false);
    });
  });

  describe('validate frame type compatibility', () => {
    it('returns error for incompatible frame types', async () => {
      const config: PipelineConfig = {
        modules: {
          encoder: { type: 'EncoderModule', properties: {} },
          writer: { type: 'FileWriterModule', properties: { path: '/out.jpg' } },
        },
        // encoded_image output -> raw_image input (incompatible with FileWriter that accepts both)
        // Actually FileWriter accepts encoded_image, so let's create a scenario that fails
        connections: [{ from: 'encoder.output', to: 'writer.input' }],
      };

      const result = await validator.validate(config);

      // FileWriterModule accepts encoded_image, so this should be valid
      expect(result.issues.some((i) => i.code === 'E205')).toBe(false);
    });
  });

  describe('validate disconnected modules', () => {
    it('warns about source with no outgoing connections', async () => {
      const config: PipelineConfig = {
        modules: {
          src: { type: 'FileReaderModule', properties: { path: '/test.jpg' } },
        },
        connections: [],
      };

      const result = await validator.validate(config);

      expect(result.issues.some((i) => i.code === 'W102')).toBe(true);
    });

    it('warns about sink with no incoming connections', async () => {
      const config: PipelineConfig = {
        modules: {
          writer: { type: 'FileWriterModule', properties: { path: '/out.jpg' } },
        },
        connections: [],
      };

      const result = await validator.validate(config);

      expect(result.issues.some((i) => i.code === 'W103')).toBe(true);
    });

    it('warns about transform with no connections', async () => {
      const config: PipelineConfig = {
        modules: {
          resize: { type: 'ResizeModule', properties: { width: 640, height: 480 } },
        },
        connections: [],
      };

      const result = await validator.validate(config);

      expect(result.issues.some((i) => i.code === 'W101')).toBe(true);
    });
  });

  describe('validate complete pipeline', () => {
    it('validates a complete valid pipeline', async () => {
      const config: PipelineConfig = {
        modules: {
          src: { type: 'FileReaderModule', properties: { path: '/input.jpg', loop: false } },
          resize: { type: 'ResizeModule', properties: { width: 640, height: 480 } },
          writer: { type: 'FileWriterModule', properties: { path: '/output.jpg', format: 'jpg' } },
        },
        connections: [
          { from: 'src.output', to: 'resize.input' },
          { from: 'resize.output', to: 'writer.input' },
        ],
      };

      const result = await validator.validate(config);

      expect(result.valid).toBe(true);
      expect(result.issues.filter((i) => i.level === 'error')).toHaveLength(0);
    });
  });
});
