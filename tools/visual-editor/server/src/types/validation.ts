/**
 * Validation types for pipeline validation
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
 * Pipeline configuration format for validation
 */
export interface PipelineConfig {
  modules: Record<string, ModuleConfig>;
  connections: ConnectionConfig[];
}

/**
 * Module configuration
 */
export interface ModuleConfig {
  type: string;
  properties?: Record<string, unknown>;
}

/**
 * Connection configuration
 */
export interface ConnectionConfig {
  from: string;
  to: string;
}
