# ApraPipes Node.js Examples

This directory contains example applications demonstrating the ApraPipes Node.js addon API.

## Prerequisites

1. Build ApraPipes with Node.js addon support:
   ```bash
   cmake -B build -S . -DENABLE_NODEJS=ON
   cmake --build build --parallel
   ```

2. The addon file `aprapipes.node` should be in the project root.

## Examples

### 1. Basic Pipeline (`basic_pipeline.js`)

A simple "Hello World" example showing how to:
- Load the addon
- Create a pipeline from JSON configuration
- Get module handles
- Set up event handlers

```bash
node examples/node/basic_pipeline.js
```

### 2. PTZ Control (`ptz_control.js`)

Demonstrates runtime property control using the VirtualPTZ module:
- Check for dynamic property support
- Read and write properties at runtime
- Simulate pan/tilt/zoom operations

```bash
node examples/node/ptz_control.js
```

### 3. Event Handling (`event_handling.js`)

Shows how to use the event system:
- Register health and error event handlers
- Method chaining for event registration
- Remove event handlers
- Event data structure

```bash
node examples/node/event_handling.js
```

### 4. Image Processing (`image_processing.js`)

A more complex pipeline with multiple processing stages:
- Color conversion
- Image resizing
- Brightness/contrast adjustment
- Rotation

```bash
node examples/node/image_processing.js
```

## API Quick Reference

### Creating Pipelines

```javascript
const ap = require('./aprapipes.node');

const config = {
    name: "MyPipeline",
    modules: {
        source: { type: "TestSignalGenerator", props: { width: 640, height: 480 } },
        sink: { type: "StatSink" }
    },
    connections: [
        { from: "source", to: "sink" }
    ]
};

const pipeline = ap.createPipeline(config);
```

### Getting Module Handles

```javascript
const module = pipeline.getModule('source');
console.log(module.type);  // "TestSignalGenerator"
console.log(module.id);    // Unique instance ID
```

### Event Handling

```javascript
pipeline
    .on('health', (event) => {
        console.log(`Health: ${event.moduleId} - ${event.message}`);
    })
    .on('error', (event) => {
        console.error(`Error: ${event.moduleId} - ${event.message}`);
    });

// Remove specific handler
pipeline.off('health', myHandler);

// Remove all handlers
pipeline.removeAllListeners('health');
```

### Dynamic Properties (Runtime Control)

```javascript
const ptz = pipeline.getModule('ptz');

// Check if module supports dynamic properties
if (ptz.hasDynamicProperties()) {
    // Get available property names
    const props = ptz.getDynamicPropertyNames();
    // ['roiX', 'roiY', 'roiWidth', 'roiHeight']

    // Read property
    const x = ptz.getProperty('roiX');

    // Write property
    ptz.setProperty('roiX', 0.5);
}
```

## Available Modules

Run the CLI to see all registered modules:

```bash
./build/aprapipes_cli list-modules
```

Or get details about a specific module:

```bash
./build/aprapipes_cli describe VirtualPTZ
```
