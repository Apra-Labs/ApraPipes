import { describe, it, expect, beforeEach } from 'vitest';
import { schemaLoader } from '../services/SchemaLoader.js';

describe('SchemaLoader', () => {
  beforeEach(async () => {
    // Force reload to reset cache
    await schemaLoader.reload();
  });

  it('loads schema successfully', async () => {
    const schema = await schemaLoader.load();

    expect(schema).toBeDefined();
    expect(schema.modules).toBeDefined();
    expect(typeof schema.modules).toBe('object');
  });

  it('returns schema with modules and frameTypes', async () => {
    const schema = await schemaLoader.load();

    // Schema should have both modules and frameTypes
    expect(schema.modules).toBeDefined();
    expect(schema.frameTypes).toBeDefined();
    expect(Object.keys(schema.modules).length).toBeGreaterThan(0);
    expect(Object.keys(schema.frameTypes).length).toBeGreaterThan(0);
  });

  it('loads from addon or returns mock data', async () => {
    const schema = await schemaLoader.load();

    // Either addon or mock - should have TestSignalGenerator
    expect(schema.modules).toHaveProperty('TestSignalGenerator');
    expect(schema.modules.TestSignalGenerator.category).toBe('source');
  });

  it('caches schema after first load', async () => {
    const schema1 = await schemaLoader.load();
    const schema2 = await schemaLoader.load();

    // Should return same instance
    expect(schema1).toBe(schema2);
  });

  it('returns module schema with correct structure', async () => {
    const schema = await schemaLoader.load();
    const testModule = schema.modules.TestSignalGenerator;

    expect(testModule).toBeDefined();
    expect(testModule.category).toBe('source');
    expect(testModule.description).toBeDefined();
    expect(Array.isArray(testModule.inputs)).toBe(true);
    expect(Array.isArray(testModule.outputs)).toBe(true);
    expect(typeof testModule.properties).toBe('object');
  });

  it('getModule returns specific module', async () => {
    const module = await schemaLoader.getModule('TestSignalGenerator');

    expect(module).toBeDefined();
    expect(module?.category).toBe('source');
  });

  it('getModule returns undefined for non-existent module', async () => {
    const module = await schemaLoader.getModule('NonExistentModule');

    expect(module).toBeUndefined();
  });

  it('getFrameTypes returns frame type hierarchy', async () => {
    const frameTypes = await schemaLoader.getFrameTypes();

    expect(frameTypes).toBeDefined();
    expect(Object.keys(frameTypes).length).toBeGreaterThan(0);

    // Check Frame is the base type
    if (frameTypes.Frame) {
      expect(frameTypes.Frame.parent).toBe('');
      expect(frameTypes.Frame.subtypes).toBeDefined();
    }
  });

  it('areFrameTypesCompatible returns true for exact match', async () => {
    const compatible = await schemaLoader.areFrameTypesCompatible('RawImage', 'RawImage');
    expect(compatible).toBe(true);
  });

  it('areFrameTypesCompatible returns true when input accepts Frame (any)', async () => {
    const compatible = await schemaLoader.areFrameTypesCompatible('RawImage', 'Frame');
    expect(compatible).toBe(true);
  });

  it('areFrameTypesCompatible returns true for subtype to parent', async () => {
    // RawImagePlanar is a subtype of RawImage, so it should be compatible
    const compatible = await schemaLoader.areFrameTypesCompatible('RawImagePlanar', 'RawImage');
    expect(compatible).toBe(true);
  });

  it('isAddonLoaded returns boolean', () => {
    const loaded = schemaLoader.isAddonLoaded();
    expect(typeof loaded).toBe('boolean');
  });
});
