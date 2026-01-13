/**
 * Basic Pipeline Example
 *
 * Demonstrates how to create a simple pipeline that generates test frames
 * and saves them as JPEG images to files.
 *
 * Usage: node examples/node/basic_pipeline.js
 *
 * Output: Creates frame_0001.jpg, frame_0002.jpg, etc. in ./output/
 */

const path = require('path');
const fs = require('fs');

// Load the addon from the project root
const addonPath = path.join(__dirname, '../../bin/aprapipes.node');
let ap;
try {
    ap = require(addonPath);
    console.log('ApraPipes addon loaded successfully');
} catch (e) {
    console.error('Failed to load addon:', e.message);
    console.error('Make sure you have built the project first.');
    process.exit(1);
}

// Create output directory
const outputDir = path.join(__dirname, 'output');
if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
}

// Define a pipeline that outputs JPEG images
const pipelineConfig = {
    name: "BasicPipeline",
    modules: {
        // Test signal generator - creates synthetic video frames with CHECKERBOARD pattern
        source: {
            type: "TestSignalGenerator",
            props: {
                width: 640,
                height: 480,
                pattern: "CHECKERBOARD"
            }
        },
        // Convert YUV to RGB for JPEG encoding
        colorConvert: {
            type: "ColorConversion",
            props: {
                conversionType: "YUV420PLANAR_TO_RGB"
            }
        },
        // Encode as JPEG
        encoder: {
            type: "ImageEncoderCV"
        },
        // Write to files
        writer: {
            type: "FileWriterModule",
            props: {
                strFullFileNameWithPattern: path.join(outputDir, "frame_????.jpg")
            }
        }
    },
    connections: [
        { from: "source", to: "colorConvert" },
        { from: "colorConvert", to: "encoder" },
        { from: "encoder", to: "writer" }
    ]
};

async function main() {
    console.log('\n=== Basic Pipeline Example ===\n');
    console.log('Pipeline configuration:');
    console.log(JSON.stringify(pipelineConfig, null, 2));

    // Create the pipeline
    console.log('\nCreating pipeline...');
    const pipeline = ap.createPipeline(pipelineConfig);

    // Get module handles
    const source = pipeline.getModule('source');
    const encoder = pipeline.getModule('encoder');
    const writer = pipeline.getModule('writer');

    console.log('\nModules created:');
    console.log(`  - source: ${source.type} (${source.id})`);
    console.log(`  - encoder: ${encoder.type} (${encoder.id})`);
    console.log(`  - writer: ${writer.type} (${writer.id})`);

    // Check initial properties
    console.log('\nSource properties:');
    const props = source.getProps();
    console.log(`  - fps: ${props.fps}`);
    console.log(`  - qlen: ${props.qlen}`);

    // Set up event handlers
    pipeline
        .on('health', (event) => {
            console.log(`[Health] ${event.moduleId}: ${event.message}`);
        })
        .on('error', (event) => {
            console.error(`[Error] ${event.moduleId}: ${event.message}`);
        });

    console.log('\nEvent handlers registered.');
    console.log(`Output will be written to: ${outputDir}/`);

    // Initialize the pipeline
    console.log('\nInitializing pipeline...');
    await pipeline.init();
    console.log('Pipeline initialized.');

    // Run the pipeline
    console.log('Starting pipeline...');
    pipeline.run();

    // Let it run for 2 seconds to generate some frames
    console.log('Running for 2 seconds...\n');
    await new Promise(resolve => setTimeout(resolve, 2000));

    // Stop the pipeline
    console.log('\nStopping pipeline...');
    await pipeline.stop();
    console.log('Pipeline stopped.');

    // Check output files
    const files = fs.readdirSync(outputDir).filter(f => f.startsWith('frame_') && f.endsWith('.jpg'));
    console.log(`\nGenerated ${files.length} JPEG files in ${outputDir}/`);
    if (files.length > 0) {
        console.log('Files:', files.slice(0, 5).join(', ') + (files.length > 5 ? '...' : ''));
    }

    console.log('\n=== Example Complete ===\n');
}

main().catch(err => {
    console.error('Error:', err);
    process.exit(1);
});
