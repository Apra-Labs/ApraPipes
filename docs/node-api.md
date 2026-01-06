# ApraPipes Node.js API Reference

This document describes the Node.js addon API for ApraPipes declarative pipelines.

## Installation

The Node.js addon is built as part of the ApraPipes build process when `ENABLE_NODEJS=ON`:

```bash
cmake -B build -S . -DENABLE_NODEJS=ON
cmake --build build --parallel
```

The addon file `aprapipes.node` will be copied to the project root.

## Loading the Addon

```javascript
const ap = require('./aprapipes.node');
```

## API Reference

### `ap.createPipeline(config)`

Creates a new pipeline from a JSON configuration object.

**Parameters:**
- `config` (Object): Pipeline configuration

**Returns:** `Pipeline` object

**Throws:** Error if validation or build fails

**Example:**
```javascript
const config = {
    name: "MyPipeline",
    modules: {
        source: {
            type: "TestSignalGenerator",
            props: { width: 640, height: 480 }
        },
        sink: {
            type: "StatSink"
        }
    },
    connections: [
        { from: "source", to: "sink" }
    ]
};

const pipeline = ap.createPipeline(config);
```

### Configuration Schema

```javascript
{
    // Optional pipeline name
    name: "string",

    // Required: module definitions
    modules: {
        "<instance_id>": {
            type: "ModuleTypeName",  // Required
            props: {                  // Optional
                "<prop_name>": <value>
            }
        }
    },

    // Required: connection definitions
    connections: [
        {
            from: "source_instance",      // Or "source_instance.pin_name"
            to: "destination_instance"    // Or "destination_instance.pin_name"
        }
    ]
}
```

---

## Pipeline Class

### `pipeline.getModule(instanceId)`

Gets a handle to a module instance.

**Parameters:**
- `instanceId` (string): The module's instance ID as defined in the config

**Returns:** `ModuleHandle` object

**Example:**
```javascript
const source = pipeline.getModule('source');
console.log(source.type);  // "TestSignalGenerator"
```

### `pipeline.on(event, callback)`

Registers an event listener.

**Parameters:**
- `event` (string): Event name (`"health"` or `"error"`)
- `callback` (Function): Event handler

**Returns:** `this` (for chaining)

**Events:**

| Event | Callback Signature | Description |
|-------|-------------------|-------------|
| `health` | `(event) => void` | Module health status updates |
| `error` | `(event) => void` | Module error notifications |

**Event Object Properties:**
- `moduleId` (string): Module instance ID
- `timestamp` (number): Event timestamp (ms since epoch)
- `message` (string): Human-readable message
- `errorCode` (number): Error code (for error events)

**Example:**
```javascript
pipeline
    .on('health', (e) => {
        console.log(`Health: ${e.moduleId} - ${e.message}`);
    })
    .on('error', (e) => {
        console.error(`Error: ${e.moduleId} - ${e.message}`);
    });
```

### `pipeline.off(event, callback)`

Removes a specific event listener.

**Parameters:**
- `event` (string): Event name
- `callback` (Function): The exact callback to remove

**Returns:** `this` (for chaining)

### `pipeline.removeAllListeners(event)`

Removes all listeners for an event.

**Parameters:**
- `event` (string): Event name

**Returns:** `this` (for chaining)

---

## ModuleHandle Class

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | string | Module instance ID |
| `type` | string | Module type name |

### `module.getProps()`

Gets the module's current properties.

**Returns:** Object with properties `fps`, `qlen`, `logHealth`

### `module.isRunning()`

Checks if the module is currently running.

**Returns:** boolean

### `module.isInputQueFull()`

Checks if the module's input queue is full.

**Returns:** boolean

---

## Dynamic Properties

Some modules support runtime property changes. Use these methods to read and modify properties while the pipeline is running.

### `module.hasDynamicProperties()`

Checks if the module supports dynamic property changes.

**Returns:** boolean

### `module.getDynamicPropertyNames()`

Gets the list of property names that can be changed at runtime.

**Returns:** Array of strings

### `module.getProperty(name)`

Reads a dynamic property value.

**Parameters:**
- `name` (string): Property name

**Returns:** Property value (number, boolean, or string)

**Throws:** Error if property doesn't exist or module doesn't support dynamic properties

### `module.setProperty(name, value)`

Sets a dynamic property value.

**Parameters:**
- `name` (string): Property name
- `value` (any): New value (number, boolean, or string)

**Returns:** boolean (true if successful)

**Throws:** Error if property doesn't exist or module doesn't support dynamic properties

**Example:**
```javascript
const ptz = pipeline.getModule('ptz');

if (ptz.hasDynamicProperties()) {
    // Get available properties
    const props = ptz.getDynamicPropertyNames();
    // ["roiX", "roiY", "roiWidth", "roiHeight"]

    // Read property
    const x = ptz.getProperty('roiX');

    // Write property
    ptz.setProperty('roiX', 0.5);
}
```

---

## Modules with Dynamic Properties

The following modules support runtime property changes:

### VirtualPTZ

| Property | Type | Range | Description |
|----------|------|-------|-------------|
| `roiX` | float | 0.0 - 1.0 | X position of region of interest (0 = left) |
| `roiY` | float | 0.0 - 1.0 | Y position of region of interest (0 = top) |
| `roiWidth` | float | 0.0 - 1.0 | Width of region (1 = full width) |
| `roiHeight` | float | 0.0 - 1.0 | Height of region (1 = full height) |

**Example: PTZ Control**
```javascript
const ptz = pipeline.getModule('ptz');

// Zoom to 2x (50% of frame, centered)
ptz.setProperty('roiWidth', 0.5);
ptz.setProperty('roiHeight', 0.5);
ptz.setProperty('roiX', 0.25);
ptz.setProperty('roiY', 0.25);

// Pan right
ptz.setProperty('roiX', 0.5);

// Reset to full frame
ptz.setProperty('roiX', 0);
ptz.setProperty('roiY', 0);
ptz.setProperty('roiWidth', 1);
ptz.setProperty('roiHeight', 1);
```

### BrightnessContrastControl

| Property | Type | Description |
|----------|------|-------------|
| `brightness` | float | Brightness offset (0.0 = no change) |
| `contrast` | float | Contrast multiplier (1.0 = no change) |

---

## TypeScript Support

TypeScript definitions are provided in `types/aprapipes.d.ts`:

```typescript
import * as ap from './aprapipes.node';

const config: ap.PipelineConfig = {
    modules: {
        source: { type: 'TestSignalGenerator', props: { width: 640, height: 480 } },
        sink: { type: 'StatSink' }
    },
    connections: [{ from: 'source', to: 'sink' }]
};

const pipeline: ap.Pipeline = ap.createPipeline(config);
const module: ap.ModuleHandle = pipeline.getModule('source');
```

---

## Error Handling

### Validation Errors

Pipeline configuration is validated before building. Errors include:
- Unknown module types
- Missing required properties
- Type mismatches (e.g., string where int expected)
- Invalid connections

```javascript
try {
    const pipeline = ap.createPipeline(config);
} catch (e) {
    console.error('Pipeline creation failed:', e.message);
    // "Validation failed: [E201] Type mismatch: expected float, got int"
}
```

### Build Errors

Even after validation passes, build errors can occur:
- Module initialization failures
- Connection incompatibilities (frame type mismatches)

### Runtime Errors

Use the error event handler to catch runtime errors:

```javascript
pipeline.on('error', (e) => {
    console.error(`Module ${e.moduleId} failed: ${e.message} (code: ${e.errorCode})`);
});
```

---

## Best Practices

1. **Register event handlers before starting** - Set up health and error handlers before calling `start()` to avoid missing early events.

2. **Use non-integer floats in config** - JavaScript JSON serialization converts `10.0` to `10`. Use values like `10.5` or `10.1` for float properties.

3. **Check dynamic property support** - Always call `hasDynamicProperties()` before trying to read/write properties.

4. **Clean up listeners** - Call `removeAllListeners()` when done to prevent memory leaks.

---

## Available Modules

Use the CLI to list all available modules:

```bash
./build/aprapipes_cli list-modules
```

Get detailed information about a specific module:

```bash
./build/aprapipes_cli describe VirtualPTZ
```

---

## See Also

- [Examples](../examples/node/README.md) - Working example applications
- [CLI Reference](declarative-pipeline/README.md) - Command-line tool documentation
