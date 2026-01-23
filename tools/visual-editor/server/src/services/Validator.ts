/**
 * Pipeline Validator Service
 *
 * Validates pipeline configurations against the module schema.
 * Uses aprapipes.node when available, falls back to schema-based validation.
 */

import type {
  ValidationResult,
  ValidationIssue,
  PipelineConfig,
  ModuleConfig,
} from '../types/validation.js';
import { getErrorDefinition } from './errorMessages.js';
import { SchemaLoader, type ModuleSchema } from './SchemaLoader.js';
import { logger } from '../utils/logger.js';

// Type for the native addon validation result
interface NativeValidationResult {
  valid: boolean;
  issues: Array<{
    level: 'error' | 'warning' | 'info';
    code: string;
    message: string;
    location: string;
    suggestion?: string;
  }>;
}

/**
 * Validator class for pipeline configurations
 */
export class Validator {
  private schemaLoader: SchemaLoader;
  private useNativeValidation: boolean;

  constructor(schemaLoader: SchemaLoader) {
    this.schemaLoader = schemaLoader;
    this.useNativeValidation = this.schemaLoader.isAddonLoaded();

    // Log validation mode
    if (this.useNativeValidation) {
      logger.info('Validator using native aprapipes.node validation');
    } else {
      logger.info('Validator using schema-based validation (fallback)');
    }
  }

  /**
   * Validate a pipeline configuration
   */
  async validate(config: PipelineConfig): Promise<ValidationResult> {
    // Try native validation first if addon is available
    if (this.useNativeValidation) {
      try {
        const nativeResult = await this.validateNative(config);
        if (nativeResult) {
          return nativeResult;
        }
      } catch (error) {
        logger.warn('Native validation failed, falling back to schema-based:', error);
      }
    }

    // Fall back to schema-based validation
    return this.validateWithSchema(config);
  }

  /**
   * Validate using native aprapipes.node addon
   */
  private async validateNative(config: PipelineConfig): Promise<ValidationResult | null> {
    const addon = this.schemaLoader.getAddon();
    if (!addon || typeof addon.validatePipeline !== 'function') {
      return null;
    }

    // Convert config to the format expected by the addon
    const pipelineJson = this.convertToPipelineJson(config);

    try {
      const result: NativeValidationResult = addon.validatePipeline(pipelineJson);

      // Map native issues to our ValidationIssue format
      const issues: ValidationIssue[] = result.issues.map((issue) => ({
        level: issue.level,
        code: issue.code,
        message: issue.message,
        location: issue.location,
        suggestion: issue.suggestion,
      }));

      logger.debug(`Native validation complete: ${issues.length} issues`);

      return {
        valid: result.valid,
        issues,
      };
    } catch (error) {
      logger.error('Native validation threw error:', error);
      throw error;
    }
  }

  /**
   * Convert our PipelineConfig format to the JSON format expected by aprapipes.node
   */
  private convertToPipelineJson(config: PipelineConfig): string {
    // The addon expects the standard ApraPipes JSON format:
    // { "modules": { "id": { "type": "...", "props": {...} } }, "connections": [...] }
    const pipelineObj = {
      modules: {} as Record<string, { type: string; props?: Record<string, unknown> }>,
      connections: config.connections.map((conn) => ({
        from: conn.from,
        to: conn.to,
      })),
    };

    for (const [moduleId, moduleConfig] of Object.entries(config.modules)) {
      pipelineObj.modules[moduleId] = {
        type: moduleConfig.type,
        ...(moduleConfig.properties && Object.keys(moduleConfig.properties).length > 0
          ? { props: moduleConfig.properties }
          : {}),
      };
    }

    return JSON.stringify(pipelineObj);
  }

  /**
   * Validate using schema (fallback when addon not available)
   */
  private async validateWithSchema(config: PipelineConfig): Promise<ValidationResult> {
    const issues: ValidationIssue[] = [];
    const schema = await this.schemaLoader.getSchema();

    // Check for empty pipeline
    if (Object.keys(config.modules).length === 0) {
      issues.push(this.createIssue('I101', ''));
    }

    // Validate modules
    for (const [moduleId, moduleConfig] of Object.entries(config.modules)) {
      this.validateModule(moduleId, moduleConfig, schema, issues);
    }

    // Check for duplicate module IDs (handled by object keys, but check for empty/invalid IDs)
    for (const moduleId of Object.keys(config.modules)) {
      if (!moduleId || moduleId.trim() === '') {
        issues.push(this.createIssue('E103', `modules.${moduleId}`));
      }
    }

    // Validate connections
    await this.validateConnections(config, schema, issues);

    // Check for disconnected modules
    this.checkDisconnectedModules(config, schema, issues);

    // Determine validity (no errors)
    const valid = !issues.some((issue) => issue.level === 'error');

    return { valid, issues };
  }

  /**
   * Check if using native validation
   */
  isUsingNativeValidation(): boolean {
    return this.useNativeValidation;
  }

  /**
   * Validate a single module
   */
  private validateModule(
    moduleId: string,
    moduleConfig: ModuleConfig,
    schema: Record<string, ModuleSchema>,
    issues: ValidationIssue[]
  ): void {
    const location = `modules.${moduleId}`;

    // Check if module type exists in schema
    const moduleSchema = schema[moduleConfig.type];
    if (!moduleSchema) {
      issues.push(
        this.createIssue('E101', location, `Unknown module type: ${moduleConfig.type}`)
      );
      return;
    }

    // Validate properties
    this.validateProperties(moduleId, moduleConfig, moduleSchema, issues);
  }

  /**
   * Validate module properties
   */
  private validateProperties(
    moduleId: string,
    moduleConfig: ModuleConfig,
    moduleSchema: ModuleSchema,
    issues: ValidationIssue[]
  ): void {
    const properties = moduleConfig.properties || {};
    const schemaProps = moduleSchema.properties || {};

    // Check for properties with no default (effectively required)
    for (const [propName, propDef] of Object.entries(schemaProps)) {
      // If a property has no default value and is not provided, it's a warning
      if (propDef.default === undefined && !(propName in properties)) {
        issues.push(
          this.createIssue(
            'W301',
            `modules.${moduleId}.properties.${propName}`,
            `Property ${propName} has no default and is not set`
          )
        );
      }
    }

    // Validate property types and values
    for (const [propName, propValue] of Object.entries(properties)) {
      const propDef = schemaProps[propName];
      if (!propDef) {
        // Unknown property - could be a warning, but we'll allow it for flexibility
        continue;
      }

      const propLocation = `modules.${moduleId}.properties.${propName}`;

      // Type validation
      if (!this.validatePropertyType(propValue, propDef.type)) {
        issues.push(
          this.createIssue(
            'E302',
            propLocation,
            `Expected ${propDef.type}, got ${typeof propValue}`
          )
        );
        continue;
      }

      // Range validation for numbers
      if ((propDef.type === 'int' || propDef.type === 'float') && typeof propValue === 'number') {
        const minValue = propDef.min !== undefined ? parseFloat(propDef.min) : undefined;
        const maxValue = propDef.max !== undefined ? parseFloat(propDef.max) : undefined;

        if (minValue !== undefined && !isNaN(minValue) && propValue < minValue) {
          issues.push(
            this.createIssue(
              'E303',
              propLocation,
              `Value ${propValue} is below minimum ${minValue}`
            )
          );
        }
        if (maxValue !== undefined && !isNaN(maxValue) && propValue > maxValue) {
          issues.push(
            this.createIssue(
              'E303',
              propLocation,
              `Value ${propValue} exceeds maximum ${maxValue}`
            )
          );
        }
      }

      // Enum validation
      if (propDef.type === 'enum' && propDef.enum_values) {
        if (!propDef.enum_values.includes(String(propValue))) {
          issues.push(
            this.createIssue(
              'E304',
              propLocation,
              `Invalid value "${propValue}". Allowed: ${propDef.enum_values.join(', ')}`
            )
          );
        }
      }
    }
  }

  /**
   * Validate property type
   */
  private validatePropertyType(value: unknown, expectedType: string): boolean {
    switch (expectedType) {
      case 'int':
        return typeof value === 'number' && Number.isInteger(value);
      case 'float':
        return typeof value === 'number';
      case 'bool':
        return typeof value === 'boolean';
      case 'string':
        return typeof value === 'string';
      case 'enum':
        return typeof value === 'string';
      case 'json':
        return typeof value === 'object' && value !== null;
      default:
        return true; // Unknown type, allow
    }
  }

  /**
   * Validate connections
   */
  private async validateConnections(
    config: PipelineConfig,
    schema: Record<string, ModuleSchema>,
    issues: ValidationIssue[]
  ): Promise<void> {
    const seenConnections = new Set<string>();

    for (let i = 0; i < config.connections.length; i++) {
      const conn = config.connections[i];
      const location = `connections[${i}]`;

      // Parse connection endpoints
      const [sourceModuleId, sourcePin] = conn.from.split('.');
      const [targetModuleId, targetPin] = conn.to.split('.');

      // Check for self-connection
      if (sourceModuleId === targetModuleId) {
        issues.push(this.createIssue('E207', location, `Self-connection on module ${sourceModuleId}`));
        continue;
      }

      // Check for duplicate connections
      const connKey = `${conn.from}->${conn.to}`;
      if (seenConnections.has(connKey)) {
        issues.push(this.createIssue('E206', location, `Duplicate connection: ${connKey}`));
        continue;
      }
      seenConnections.add(connKey);

      // Check source module exists
      const sourceModule = config.modules[sourceModuleId];
      if (!sourceModule) {
        issues.push(
          this.createIssue('E201', location, `Source module "${sourceModuleId}" not found`)
        );
        continue;
      }

      // Check target module exists
      const targetModule = config.modules[targetModuleId];
      if (!targetModule) {
        issues.push(
          this.createIssue('E202', location, `Target module "${targetModuleId}" not found`)
        );
        continue;
      }

      // Check source pin exists
      const sourceSchema = schema[sourceModule.type];
      if (sourceSchema) {
        const hasOutputPin = sourceSchema.outputs?.some((o) => o.name === sourcePin);
        if (!hasOutputPin) {
          issues.push(
            this.createIssue(
              'E203',
              location,
              `Output pin "${sourcePin}" not found on module "${sourceModuleId}"`
            )
          );
        }
      }

      // Check target pin exists
      const targetSchema = schema[targetModule.type];
      if (targetSchema) {
        const hasInputPin = targetSchema.inputs?.some((i) => i.name === targetPin);
        if (!hasInputPin) {
          issues.push(
            this.createIssue(
              'E204',
              location,
              `Input pin "${targetPin}" not found on module "${targetModuleId}"`
            )
          );
        }
      }

      // Check frame type compatibility
      if (sourceSchema && targetSchema) {
        const sourceOutput = sourceSchema.outputs?.find((o) => o.name === sourcePin);
        const targetInput = targetSchema.inputs?.find((i) => i.name === targetPin);

        if (sourceOutput && targetInput) {
          const compatible = await this.checkFrameTypeCompatibility(
            sourceOutput.frame_types,
            targetInput.frame_types
          );
          if (!compatible) {
            issues.push(
              this.createIssue(
                'E205',
                location,
                `Incompatible frame types: ${sourceOutput.frame_types.join(',')} -> ${targetInput.frame_types.join(',')}`
              )
            );
          }
        }
      }
    }
  }

  /**
   * Check frame type compatibility between pins using hierarchy
   */
  private async checkFrameTypeCompatibility(sourceTypes: string[], targetTypes: string[]): Promise<boolean> {
    // If either side accepts any type, it's compatible
    if (sourceTypes.length === 0 || targetTypes.length === 0) {
      return true;
    }

    // Check if any source type is compatible with any target type
    for (const sourceType of sourceTypes) {
      for (const targetType of targetTypes) {
        // Use the schema loader's hierarchy-aware compatibility check
        const compatible = await this.schemaLoader.areFrameTypesCompatible(sourceType, targetType);
        if (compatible) {
          return true;
        }
      }
    }

    return false;
  }

  /**
   * Check for disconnected modules
   */
  private checkDisconnectedModules(
    config: PipelineConfig,
    schema: Record<string, ModuleSchema>,
    issues: ValidationIssue[]
  ): void {
    const connectedModules = new Set<string>();

    // Build set of connected modules
    for (const conn of config.connections) {
      const [sourceId] = conn.from.split('.');
      const [targetId] = conn.to.split('.');
      connectedModules.add(sourceId);
      connectedModules.add(targetId);
    }

    // Check each module
    for (const [moduleId, moduleConfig] of Object.entries(config.modules)) {
      const moduleSchema = schema[moduleConfig.type];
      if (!moduleSchema) continue;

      const hasInputs = (moduleSchema.inputs?.length || 0) > 0;
      const hasOutputs = (moduleSchema.outputs?.length || 0) > 0;

      // Check if module has any connections
      const hasIncoming = config.connections.some((c) => c.to.startsWith(`${moduleId}.`));
      const hasOutgoing = config.connections.some((c) => c.from.startsWith(`${moduleId}.`));

      // Source modules should have outgoing connections
      if (!hasInputs && hasOutputs && !hasOutgoing) {
        issues.push(
          this.createIssue('W102', `modules.${moduleId}`, `Source module "${moduleId}" has no outgoing connections`)
        );
      }

      // Sink modules should have incoming connections
      if (hasInputs && !hasOutputs && !hasIncoming) {
        issues.push(
          this.createIssue('W103', `modules.${moduleId}`, `Sink module "${moduleId}" has no incoming connections`)
        );
      }

      // Modules with both inputs and outputs should be connected
      if (hasInputs && hasOutputs && !hasIncoming && !hasOutgoing) {
        issues.push(
          this.createIssue('W101', `modules.${moduleId}`, `Module "${moduleId}" has no connections`)
        );
      }
    }
  }

  /**
   * Create a validation issue
   */
  private createIssue(code: string, location: string, details?: string): ValidationIssue {
    const def = getErrorDefinition(code);
    if (!def) {
      return {
        level: 'error',
        code,
        message: details || 'Unknown error',
        location,
      };
    }

    return {
      level: def.level,
      code,
      message: details || def.message,
      location,
      suggestion: def.suggestion,
    };
  }
}

/**
 * Singleton validator instance
 */
let validatorInstance: Validator | null = null;

/**
 * Get the validator instance
 */
export function getValidator(schemaLoader: SchemaLoader): Validator {
  if (!validatorInstance) {
    validatorInstance = new Validator(schemaLoader);
  }
  return validatorInstance;
}
