import { nanoid } from 'nanoid';

/**
 * Generate a unique node ID for the canvas
 * Format: <moduleType>_<shortId> (e.g., "TestSignalGenerator_abc123")
 */
export function generateNodeId(moduleType: string): string {
  const shortId = nanoid(6);
  return `${moduleType}_${shortId}`;
}

/**
 * Generate a unique ID (no prefix)
 */
export function generateId(): string {
  return nanoid(10);
}

/**
 * Extract the module type from a node ID
 */
export function getModuleTypeFromId(nodeId: string): string | null {
  const parts = nodeId.split('_');
  if (parts.length >= 2) {
    return parts.slice(0, -1).join('_');
  }
  return null;
}
