import { describe, it, expect, beforeEach } from 'vitest';
import { usePipelineStore } from '../store/pipelineStore';
import type { ModuleSchema } from '../types/schema';

const mockSchema: Record<string, ModuleSchema> = {
  TestSource: {
    category: 'source',
    description: 'Test source module',
    inputs: [],
    outputs: [{ name: 'output', frame_types: ['RAW_IMAGE'] }],
    properties: {
      width: { type: 'int', default: '640', min: '1', max: '4096' },
      height: { type: 'int', default: '480', min: '1', max: '4096' },
    },
  },
  TestTransform: {
    category: 'transform',
    description: 'Test transform module',
    inputs: [{ name: 'input', frame_types: ['RAW_IMAGE'] }],
    outputs: [{ name: 'output', frame_types: ['RAW_IMAGE'] }],
    properties: {
      enabled: { type: 'bool', default: 'true' },
    },
  },
};

describe('pipelineStore', () => {
  beforeEach(() => {
    usePipelineStore.getState().reset();
  });

  const getState = () => usePipelineStore.getState();

  describe('setSchema', () => {
    it('sets the schema', () => {
      getState().setSchema(mockSchema);
      expect(getState().schema).toEqual(mockSchema);
    });
  });

  describe('addModule', () => {
    it('adds a module to config', () => {
      getState().setSchema(mockSchema);
      getState().addModule('src1', 'TestSource');

      expect(getState().config.modules['src1']).toBeDefined();
      expect(getState().config.modules['src1'].type).toBe('TestSource');
    });

    it('initializes properties with defaults', () => {
      getState().setSchema(mockSchema);
      getState().addModule('src1', 'TestSource');

      const module = getState().config.modules['src1'];
      expect(module.properties.width).toBe(640);
      expect(module.properties.height).toBe(480);
    });

    it('marks state as dirty', () => {
      getState().setSchema(mockSchema);
      expect(getState().isDirty).toBe(false);

      getState().addModule('src1', 'TestSource');
      expect(getState().isDirty).toBe(true);
    });
  });

  describe('removeModule', () => {
    it('removes a module from config', () => {
      getState().setSchema(mockSchema);
      getState().addModule('src1', 'TestSource');
      expect(getState().config.modules['src1']).toBeDefined();

      getState().removeModule('src1');
      expect(getState().config.modules['src1']).toBeUndefined();
    });

    it('removes connections involving the module', () => {
      getState().setSchema(mockSchema);
      getState().addModule('src1', 'TestSource');
      getState().addModule('xform1', 'TestTransform');
      getState().addConnection('src1.output', 'xform1.input');

      expect(getState().config.connections).toHaveLength(1);

      getState().removeModule('src1');
      expect(getState().config.connections).toHaveLength(0);
    });
  });

  describe('renameModule', () => {
    it('renames a module', () => {
      getState().setSchema(mockSchema);
      getState().addModule('src1', 'TestSource');

      getState().renameModule('src1', 'mySource');

      expect(getState().config.modules['src1']).toBeUndefined();
      expect(getState().config.modules['mySource']).toBeDefined();
      expect(getState().config.modules['mySource'].type).toBe('TestSource');
    });

    it('updates connections when renaming', () => {
      getState().setSchema(mockSchema);
      getState().addModule('src1', 'TestSource');
      getState().addModule('xform1', 'TestTransform');
      getState().addConnection('src1.output', 'xform1.input');

      getState().renameModule('src1', 'mySource');

      expect(getState().config.connections[0].from).toBe('mySource.output');
      expect(getState().config.connections[0].to).toBe('xform1.input');
    });

    it('handles no-op rename', () => {
      getState().setSchema(mockSchema);
      getState().addModule('src1', 'TestSource');
      getState().markClean();

      getState().renameModule('src1', 'src1');

      // Should not mark dirty for no-op rename
      expect(getState().isDirty).toBe(false);
    });
  });

  describe('updateModuleProperty', () => {
    it('updates a property value', () => {
      getState().setSchema(mockSchema);
      getState().addModule('src1', 'TestSource');

      getState().updateModuleProperty('src1', 'width', 1920);

      expect(getState().config.modules['src1'].properties.width).toBe(1920);
    });

    it('preserves other properties', () => {
      getState().setSchema(mockSchema);
      getState().addModule('src1', 'TestSource');

      getState().updateModuleProperty('src1', 'width', 1920);

      expect(getState().config.modules['src1'].properties.height).toBe(480);
    });
  });

  describe('addConnection', () => {
    it('adds a connection', () => {
      getState().setSchema(mockSchema);
      getState().addModule('src1', 'TestSource');
      getState().addModule('xform1', 'TestTransform');

      getState().addConnection('src1.output', 'xform1.input');

      expect(getState().config.connections).toHaveLength(1);
      expect(getState().config.connections[0]).toEqual({
        from: 'src1.output',
        to: 'xform1.input',
      });
    });

    it('prevents duplicate connections', () => {
      getState().setSchema(mockSchema);
      getState().addModule('src1', 'TestSource');
      getState().addModule('xform1', 'TestTransform');

      getState().addConnection('src1.output', 'xform1.input');
      getState().addConnection('src1.output', 'xform1.input');

      expect(getState().config.connections).toHaveLength(1);
    });
  });

  describe('removeConnection', () => {
    it('removes a connection by index', () => {
      getState().setSchema(mockSchema);
      getState().addModule('src1', 'TestSource');
      getState().addModule('xform1', 'TestTransform');
      getState().addConnection('src1.output', 'xform1.input');

      getState().removeConnection(0);

      expect(getState().config.connections).toHaveLength(0);
    });
  });

  describe('toJSON / fromJSON', () => {
    it('serializes to JSON', () => {
      getState().setSchema(mockSchema);
      getState().addModule('src1', 'TestSource');
      getState().addModule('xform1', 'TestTransform');
      getState().addConnection('src1.output', 'xform1.input');

      const json = getState().toJSON();
      const parsed = JSON.parse(json);

      expect(parsed.modules.src1.type).toBe('TestSource');
      expect(parsed.modules.xform1.type).toBe('TestTransform');
      expect(parsed.connections).toHaveLength(1);
    });

    it('deserializes from JSON', () => {
      const json = JSON.stringify({
        modules: {
          src1: { type: 'TestSource', properties: { width: 1920 } },
        },
        connections: [],
      });

      getState().fromJSON(json);

      expect(getState().config.modules.src1.type).toBe('TestSource');
      expect(getState().config.modules.src1.properties.width).toBe(1920);
      expect(getState().isDirty).toBe(false);
    });
  });

  describe('reset', () => {
    it('resets to initial state', () => {
      getState().setSchema(mockSchema);
      getState().addModule('src1', 'TestSource');

      getState().reset();

      expect(getState().config.modules).toEqual({});
      expect(getState().config.connections).toEqual([]);
      expect(getState().schema).toEqual({});
      expect(getState().isDirty).toBe(false);
    });
  });

  describe('markClean', () => {
    it('marks state as clean', () => {
      getState().setSchema(mockSchema);
      getState().addModule('src1', 'TestSource');
      expect(getState().isDirty).toBe(true);

      getState().markClean();
      expect(getState().isDirty).toBe(false);
    });
  });
});
