/**
 * API client for communicating with the backend server
 */

import type { SchemaResponse } from '../types/schema';

const API_BASE = '/api';

class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public statusText: string
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const message = await response.text().catch(() => response.statusText);
    throw new ApiError(message, response.status, response.statusText);
  }
  return response.json();
}

/**
 * API client singleton
 */
export const api = {
  /**
   * Fetch module schema from the server
   */
  async getSchema(): Promise<SchemaResponse> {
    const response = await fetch(`${API_BASE}/schema`);
    return handleResponse<SchemaResponse>(response);
  },

  /**
   * Validate a pipeline configuration
   */
  async validatePipeline(config: unknown): Promise<{
    valid: boolean;
    issues: Array<{
      level: 'error' | 'warning' | 'info';
      code: string;
      message: string;
      location: string;
      suggestion?: string;
    }>;
  }> {
    const response = await fetch(`${API_BASE}/validate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ config }),
    });
    return handleResponse(response);
  },

  /**
   * Get schema status including addon loading status
   */
  async getSchemaStatus(): Promise<{
    addonLoaded: boolean;
    moduleCount: number;
    frameTypeCount: number;
  }> {
    const response = await fetch(`${API_BASE}/schema/status`);
    return handleResponse(response);
  },
};
