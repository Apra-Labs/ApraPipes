import { describe, it, expect } from 'vitest';
import { generateNodeId, generateId, getModuleTypeFromId } from '../utils/id';

describe('id utilities', () => {
  describe('generateNodeId', () => {
    it('generates ID with module type prefix', () => {
      const id = generateNodeId('TestSignalGenerator');
      expect(id).toMatch(/^TestSignalGenerator_[a-zA-Z0-9_-]+$/);
    });

    it('generates unique IDs', () => {
      const id1 = generateNodeId('TestModule');
      const id2 = generateNodeId('TestModule');
      expect(id1).not.toBe(id2);
    });
  });

  describe('generateId', () => {
    it('generates a string ID', () => {
      const id = generateId();
      expect(typeof id).toBe('string');
      expect(id.length).toBe(10);
    });

    it('generates unique IDs', () => {
      const ids = new Set<string>();
      for (let i = 0; i < 100; i++) {
        ids.add(generateId());
      }
      expect(ids.size).toBe(100);
    });
  });

  describe('getModuleTypeFromId', () => {
    it('extracts module type from valid ID', () => {
      const type = getModuleTypeFromId('TestSignalGenerator_abc123');
      expect(type).toBe('TestSignalGenerator');
    });

    it('handles module types with underscores', () => {
      const type = getModuleTypeFromId('Some_Module_Name_abc123');
      expect(type).toBe('Some_Module_Name');
    });

    it('returns null for invalid ID', () => {
      const type = getModuleTypeFromId('nounderscores');
      expect(type).toBeNull();
    });
  });
});
