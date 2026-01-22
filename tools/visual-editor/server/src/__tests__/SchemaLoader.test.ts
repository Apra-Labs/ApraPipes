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

  it('returns mock data when modules.json not found', async () => {
    const schema = await schemaLoader.load();

    // Mock schema should have at least these modules
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
});
