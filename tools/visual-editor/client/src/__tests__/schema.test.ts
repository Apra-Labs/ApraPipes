import { describe, it, expect } from 'vitest';
import { getCategoryColor } from '../types/schema';

describe('schema utilities', () => {
  describe('getCategoryColor', () => {
    it('returns correct color for source category', () => {
      expect(getCategoryColor('source')).toBe('bg-source text-white');
    });

    it('returns correct color for transform category', () => {
      expect(getCategoryColor('transform')).toBe('bg-transform text-white');
    });

    it('returns correct color for sink category', () => {
      expect(getCategoryColor('sink')).toBe('bg-sink text-white');
    });

    it('returns correct color for cuda category', () => {
      expect(getCategoryColor('cuda')).toBe('bg-purple-500 text-white');
    });

    it('returns gray for unknown category', () => {
      expect(getCategoryColor('unknown')).toBe('bg-gray-500 text-white');
    });

    it('is case insensitive', () => {
      expect(getCategoryColor('SOURCE')).toBe('bg-source text-white');
      expect(getCategoryColor('Transform')).toBe('bg-transform text-white');
    });
  });
});
