/**
 * Smoke test for @apralabs/aprapipes Node.js addon
 *
 * Run with: npm test
 * Prerequisites: Build the addon with: npm run configure && npm run build
 */

const assert = require('assert');
const path = require('path');

// Load the native addon
let aprapipes;
try {
  aprapipes = require(path.join(__dirname, '..', 'aprapipes.node'));
} catch (err) {
  console.error('Failed to load native addon.');
  console.error('Make sure you have built the addon:');
  console.error('  npm run configure');
  console.error('  npm run build');
  console.error('');
  console.error('Error:', err.message);
  process.exit(1);
}

console.log('Testing @apralabs/aprapipes Node.js addon...\n');

// ============================================================
// Test: getVersion()
// ============================================================
console.log('Test: getVersion()');
const version = aprapipes.getVersion();
assert(typeof version === 'string', 'getVersion() should return a string');
assert(version.length > 0, 'Version string should not be empty');
console.log(`  Version: ${version}`);
console.log('  ✓ Passed\n');

// ============================================================
// Test: listModules()
// ============================================================
console.log('Test: listModules()');
const modules = aprapipes.listModules();
assert(Array.isArray(modules), 'listModules() should return an array');
assert(modules.length > 0, 'At least one module should be registered');
console.log(`  Found ${modules.length} modules:`);
modules.slice(0, 5).forEach(m => console.log(`    - ${m}`));
if (modules.length > 5) {
  console.log(`    ... and ${modules.length - 5} more`);
}
console.log('  ✓ Passed\n');

// ============================================================
// Test: describeModule()
// ============================================================
console.log('Test: describeModule()');
if (modules.length > 0) {
  const moduleName = modules[0];
  const info = aprapipes.describeModule(moduleName);
  assert(typeof info === 'object', 'describeModule() should return an object');
  assert(info.name === moduleName, 'Module name should match');
  assert(['source', 'sink', 'transform', 'analytics', 'utility', 'unknown'].includes(info.category),
    'Category should be valid');
  assert(Array.isArray(info.tags), 'Tags should be an array');
  assert(Array.isArray(info.properties), 'Properties should be an array');
  assert(Array.isArray(info.inputs), 'Inputs should be an array');
  assert(Array.isArray(info.outputs), 'Outputs should be an array');
  console.log(`  Module: ${info.name}`);
  console.log(`  Category: ${info.category}`);
  console.log(`  Description: ${info.description || '(none)'}`);
  console.log(`  Properties: ${info.properties.length}`);
  console.log(`  Inputs: ${info.inputs.length}`);
  console.log(`  Outputs: ${info.outputs.length}`);
  console.log('  ✓ Passed\n');
} else {
  console.log('  ⚠ Skipped (no modules registered)\n');
}

// ============================================================
// Test: describeModule() with invalid module
// ============================================================
console.log('Test: describeModule() with invalid module');
try {
  aprapipes.describeModule('NonExistentModule12345');
  assert.fail('Should throw error for non-existent module');
} catch (err) {
  assert(err.message.includes('not found'), 'Error should mention module not found');
  console.log(`  Correctly threw error: ${err.message}`);
  console.log('  ✓ Passed\n');
}

// ============================================================
// Test: validatePipeline() with valid config
// ============================================================
console.log('Test: validatePipeline() with valid config');
const validConfig = {
  pipeline: {
    name: 'test_pipeline'
  },
  modules: {
    source: {
      type: 'TestSignalGenerator',
      props: {
        width: 640,
        height: 480
      }
    },
    sink: {
      type: 'StatSink'
    }
  },
  connections: [
    { from: 'source', to: 'sink' }
  ]
};

const validResult = aprapipes.validatePipeline(validConfig);
assert(typeof validResult === 'object', 'validatePipeline() should return an object');
assert(typeof validResult.valid === 'boolean', 'Result should have valid boolean');
assert(Array.isArray(validResult.issues), 'Result should have issues array');
console.log(`  Valid: ${validResult.valid}`);
console.log(`  Issues: ${validResult.issues.length}`);
if (validResult.issues.length > 0) {
  validResult.issues.forEach(issue => {
    console.log(`    [${issue.level}] ${issue.code}: ${issue.message}`);
  });
}
console.log('  ✓ Passed\n');

// ============================================================
// Test: validatePipeline() with string config
// ============================================================
console.log('Test: validatePipeline() with string config');
const jsonString = JSON.stringify(validConfig);
const stringResult = aprapipes.validatePipeline(jsonString);
assert(typeof stringResult === 'object', 'validatePipeline(string) should return an object');
assert(typeof stringResult.valid === 'boolean', 'Result should have valid boolean');
console.log(`  Valid: ${stringResult.valid}`);
console.log('  ✓ Passed\n');

// ============================================================
// Test: validatePipeline() with invalid JSON
// ============================================================
console.log('Test: validatePipeline() with invalid JSON');
const invalidJsonResult = aprapipes.validatePipeline('{ invalid json }');
assert(invalidJsonResult.valid === false, 'Invalid JSON should fail validation');
assert(invalidJsonResult.issues.length > 0, 'Should have at least one issue');
console.log(`  Valid: ${invalidJsonResult.valid}`);
console.log(`  Issues: ${invalidJsonResult.issues.length}`);
console.log(`  Error: ${invalidJsonResult.issues[0].message}`);
console.log('  ✓ Passed\n');

// ============================================================
// Test: validatePipeline() with missing module type
// ============================================================
console.log('Test: validatePipeline() with unknown module');
const unknownModuleConfig = {
  modules: {
    source: { type: 'UnknownModule12345' }
  },
  connections: []
};
const unknownResult = aprapipes.validatePipeline(unknownModuleConfig);
assert(unknownResult.valid === false, 'Unknown module should fail validation');
assert(unknownResult.issues.some(i => i.level === 'error'), 'Should have error issue');
console.log(`  Valid: ${unknownResult.valid}`);
console.log(`  Issues: ${unknownResult.issues.length}`);
unknownResult.issues.forEach(issue => {
  console.log(`    [${issue.level}] ${issue.code}: ${issue.message}`);
});
console.log('  ✓ Passed\n');

// ============================================================
// Summary
// ============================================================
console.log('='.repeat(50));
console.log('All smoke tests passed!');
console.log('='.repeat(50));
