import '@testing-library/jest-dom';
import { vi, beforeEach } from 'vitest';

// Mock fetch for API calls
global.fetch = vi.fn();

// Reset mocks between tests
beforeEach(() => {
  vi.resetAllMocks();
});
