/**
 * Server configuration
 */

export const config = {
  port: parseInt(process.env.PORT || '3000', 10),
  host: process.env.HOST || 'localhost',

  /** Path to modules.json from schema_generator */
  schemaPath: process.env.APRAPIPES_SCHEMA_PATH,

  /** Enable mock mode when aprapipes.node is unavailable */
  mockMode: process.env.MOCK_MODE === 'true',
};
