/**
 * Validation types for the client
 */

/**
 * Severity level of a validation issue
 */
export type IssueSeverity = 'error' | 'warning' | 'info';

/**
 * A single validation issue
 */
export interface ValidationIssue {
  /** Severity level */
  level: IssueSeverity;
  /** Error code (e.g., E101, W201) */
  code: string;
  /** User-friendly error message */
  message: string;
  /** JSONPath to the issue location (e.g., "modules.source.properties.width") */
  location: string;
  /** Optional actionable suggestion for fixing the issue */
  suggestion?: string;
}

/**
 * Result of pipeline validation
 */
export interface ValidationResult {
  /** Whether the pipeline is valid (no errors) */
  valid: boolean;
  /** List of validation issues found */
  issues: ValidationIssue[];
}

/**
 * Parse a location string to extract module ID
 * @param location JSONPath-like location string
 * @returns Module ID if found, null otherwise
 */
export function parseLocationModuleId(location: string): string | null {
  // Match patterns like "modules.myModule" or "modules.myModule.properties.width"
  const match = location.match(/^modules\.([^.]+)/);
  return match ? match[1] : null;
}

/**
 * Get severity icon for display
 */
export function getSeverityIcon(level: IssueSeverity): string {
  switch (level) {
    case 'error':
      return '❌';
    case 'warning':
      return '⚠️';
    case 'info':
      return 'ℹ️';
    default:
      return '●';
  }
}

/**
 * Get severity color class for display
 */
export function getSeverityColor(level: IssueSeverity): string {
  switch (level) {
    case 'error':
      return 'text-red-600';
    case 'warning':
      return 'text-yellow-600';
    case 'info':
      return 'text-blue-600';
    default:
      return 'text-gray-600';
  }
}

/**
 * Get severity background color class
 */
export function getSeverityBgColor(level: IssueSeverity): string {
  switch (level) {
    case 'error':
      return 'bg-red-50 border-red-200';
    case 'warning':
      return 'bg-yellow-50 border-yellow-200';
    case 'info':
      return 'bg-blue-50 border-blue-200';
    default:
      return 'bg-gray-50 border-gray-200';
  }
}
