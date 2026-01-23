/**
 * Error messages and codes for pipeline validation
 *
 * Error code format:
 * - E1xx: Module errors
 * - E2xx: Connection errors
 * - E3xx: Property errors
 * - W1xx: Module warnings
 * - W2xx: Connection warnings
 * - W3xx: Property warnings
 * - I1xx: Module info
 * - I2xx: Connection info
 */

import type { IssueSeverity } from '../types/validation.js';

export interface ErrorDefinition {
  level: IssueSeverity;
  message: string;
  suggestion?: string;
}

/**
 * Map of error codes to their definitions
 */
export const errorDefinitions: Record<string, ErrorDefinition> = {
  // Module errors (E1xx)
  E101: {
    level: 'error',
    message: 'Unknown module type',
    suggestion: 'Check the module type name or ensure the module is registered in the schema',
  },
  E102: {
    level: 'error',
    message: 'Duplicate module ID',
    suggestion: 'Each module must have a unique identifier',
  },
  E103: {
    level: 'error',
    message: 'Module ID is empty or invalid',
    suggestion: 'Provide a valid identifier for the module',
  },

  // Connection errors (E2xx)
  E201: {
    level: 'error',
    message: 'Connection references non-existent source module',
    suggestion: 'Ensure the source module exists in the pipeline',
  },
  E202: {
    level: 'error',
    message: 'Connection references non-existent target module',
    suggestion: 'Ensure the target module exists in the pipeline',
  },
  E203: {
    level: 'error',
    message: 'Connection references non-existent output pin',
    suggestion: 'Check the output pin name matches one defined in the module schema',
  },
  E204: {
    level: 'error',
    message: 'Connection references non-existent input pin',
    suggestion: 'Check the input pin name matches one defined in the module schema',
  },
  E205: {
    level: 'error',
    message: 'Incompatible frame types between connected pins',
    suggestion: 'Ensure the output pin frame types are compatible with the input pin',
  },
  E206: {
    level: 'error',
    message: 'Duplicate connection',
    suggestion: 'Remove the duplicate connection between these pins',
  },
  E207: {
    level: 'error',
    message: 'Self-connection detected',
    suggestion: 'A module cannot connect to itself',
  },

  // Property errors (E3xx)
  E301: {
    level: 'error',
    message: 'Missing required property',
    suggestion: 'Provide a value for this required property',
  },
  E302: {
    level: 'error',
    message: 'Invalid property type',
    suggestion: 'Ensure the property value matches the expected type',
  },
  E303: {
    level: 'error',
    message: 'Property value out of range',
    suggestion: 'Adjust the value to be within the allowed range',
  },
  E304: {
    level: 'error',
    message: 'Invalid enum value',
    suggestion: 'Use one of the allowed values for this property',
  },

  // Module warnings (W1xx)
  W101: {
    level: 'warning',
    message: 'Module has no connections',
    suggestion: 'Consider connecting this module or removing it if unused',
  },
  W102: {
    level: 'warning',
    message: 'Source module has no outgoing connections',
    suggestion: 'Connect the source to downstream modules to process its output',
  },
  W103: {
    level: 'warning',
    message: 'Sink module has no incoming connections',
    suggestion: 'Connect upstream modules to this sink',
  },

  // Connection warnings (W2xx)
  W201: {
    level: 'warning',
    message: 'Potential frame type mismatch',
    suggestion: 'Review the connection to ensure frame types are compatible',
  },

  // Property warnings (W3xx)
  W301: {
    level: 'warning',
    message: 'Using default value for optional property',
    suggestion: 'Consider explicitly setting this property',
  },
  W302: {
    level: 'warning',
    message: 'Property value may cause performance issues',
    suggestion: 'Review the value for optimal performance',
  },

  // Info messages (I1xx)
  I101: {
    level: 'info',
    message: 'Empty pipeline',
    suggestion: 'Add modules to create a pipeline',
  },
  I102: {
    level: 'info',
    message: 'Pipeline has multiple disconnected subgraphs',
    suggestion: 'This may be intentional, but verify the pipeline structure',
  },
};

/**
 * Get error definition by code
 */
export function getErrorDefinition(code: string): ErrorDefinition | undefined {
  return errorDefinitions[code];
}

/**
 * Create a formatted error message with location
 */
export function formatErrorMessage(code: string, location: string, details?: string): string {
  const def = errorDefinitions[code];
  if (!def) {
    return details || 'Unknown error';
  }
  return details ? `${def.message}: ${details}` : def.message;
}
