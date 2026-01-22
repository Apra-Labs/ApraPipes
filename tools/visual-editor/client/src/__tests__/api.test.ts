import { describe, it, expect, vi, beforeEach } from 'vitest';
import { api } from '../services/api';

describe('api', () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  describe('getSchema', () => {
    it('fetches schema from API', async () => {
      const mockSchema = {
        modules: {
          TestModule: {
            category: 'source',
            description: 'Test module',
            inputs: [],
            outputs: [{ name: 'output', frame_types: ['RAW_IMAGE'] }],
            properties: {},
          },
        },
      };

      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: async () => mockSchema,
      });

      const result = await api.getSchema();

      expect(global.fetch).toHaveBeenCalledWith('/api/schema');
      expect(result).toEqual(mockSchema);
    });

    it('throws ApiError on failure', async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
        text: async () => 'Server error',
      });

      await expect(api.getSchema()).rejects.toThrow('Server error');
    });
  });

  describe('validatePipeline', () => {
    it('sends pipeline config for validation', async () => {
      const mockResult = { valid: true, issues: [] };
      const config = {
        modules: { source: { type: 'TestModule' } },
        connections: [],
      };

      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResult,
      });

      const result = await api.validatePipeline(config);

      expect(global.fetch).toHaveBeenCalledWith('/api/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config }),
      });
      expect(result).toEqual(mockResult);
    });
  });
});
