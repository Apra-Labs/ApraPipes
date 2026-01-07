/**
 * Audio Capture Demo
 *
 * Demonstrates AudioCaptureSrc module which captures audio from the microphone
 * and saves it to a WAV file.
 *
 * OBSERVABLE OUTPUT:
 * - Creates recorded_audio.wav in the output directory
 * - Play the file to hear what was recorded
 *
 * Run with: node audio_capture_demo.js [duration_seconds]
 */

const path = require('path');
const fs = require('fs');

// Load the addon
const addonPath = path.join(__dirname, '../../aprapipes.node');
let ap;
try {
    ap = require(addonPath);
    console.log('ApraPipes addon loaded successfully');
} catch (e) {
    console.error('Failed to load addon:', e.message);
    console.error('Make sure you have built the project first.');
    process.exit(1);
}

// Get duration from command line (default 5 seconds)
const duration = parseInt(process.argv[2]) || 5;

// Paths
const outputDir = path.join(__dirname, 'output');
const outputFile = path.join(outputDir, 'recorded_audio.wav');

// Create output directory
if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
}

// Remove old recording
if (fs.existsSync(outputFile)) {
    fs.unlinkSync(outputFile);
}

console.log('=== Audio Capture Demo ===\n');
console.log(`Duration: ${duration} seconds`);
console.log(`Output: ${outputFile}`);
console.log('');

// Simple pipeline: AudioCaptureSrc -> FileWriterModule
const pipelineConfig = {
    name: "AudioCapturePipeline",
    modules: {
        // Capture audio from microphone
        microphone: {
            type: "AudioCaptureSrc",
            props: {
                sampleRate: 48000,       // 48 kHz
                channels: 2,              // Stereo
                audioInputDeviceIndex: 0, // Default device
                processingIntervalMS: 100 // 100ms chunks
            }
        },

        // Write to WAV file (append mode for continuous recording)
        writer: {
            type: "FileWriterModule",
            props: {
                strFullFileNameWithPattern: outputFile,
                append: true
            }
        }
    },
    connections: [
        { from: "microphone", to: "writer" }
    ]
};

async function main() {
    console.log('Pipeline structure:');
    console.log('  [AudioCaptureSrc] <- Microphone Input');
    console.log('         |');
    console.log('         v');
    console.log('  [FileWriterModule] -> recorded_audio.wav');
    console.log('');

    // Create pipeline
    console.log('Creating pipeline...');
    const pipeline = ap.createPipeline(pipelineConfig);

    // Get module handles
    const mic = pipeline.getModule('microphone');
    console.log('Microphone settings:');
    console.log(`  Sample rate: ${mic.getProps().sampleRate || 48000} Hz`);
    console.log(`  Channels: ${mic.getProps().channels || 2}`);
    console.log(`  Device index: ${mic.getProps().audioInputDeviceIndex || 0}`);
    console.log('');

    // Set up event handlers
    pipeline
        .on('health', (event) => {
            console.log(`[Health] ${event.moduleId}: ${event.message}`);
        })
        .on('error', (event) => {
            console.error(`[Error] ${event.moduleId}: ${event.message}`);
        });

    // Initialize
    console.log('Initializing pipeline...');
    await pipeline.init();

    // Start recording
    console.log(`\nRECORDING for ${duration} seconds...`);
    console.log('Speak into your microphone!\n');
    pipeline.run();

    // Record for specified duration with countdown
    for (let remaining = duration; remaining > 0; remaining--) {
        process.stdout.write(`\r  Recording: ${remaining}s remaining...   `);
        await new Promise(resolve => setTimeout(resolve, 1000));
    }
    console.log('\n');

    // Stop
    console.log('Stopping recording...');
    await pipeline.stop();

    // Check output
    if (fs.existsSync(outputFile)) {
        const stats = fs.statSync(outputFile);
        const expectedSize = 48000 * 2 * 2 * duration; // sampleRate * channels * bytesPerSample * seconds

        console.log('\n=== SUCCESS ===');
        console.log(`Created: ${outputFile}`);
        console.log(`File size: ${(stats.size / 1024).toFixed(1)} KB`);
        console.log(`Expected: ~${(expectedSize / 1024).toFixed(1)} KB for ${duration}s stereo 48kHz`);
        console.log('');
        console.log('To play the recording:');
        console.log(`  afplay ${outputFile}`);
        console.log('  OR open in Audacity/VLC');
    } else {
        console.log('\nNo output file generated. Check for errors above.');
        console.log('Make sure your microphone is connected and permissions are granted.');
    }

    console.log('\n=== Demo Complete ===\n');
}

main().catch(err => {
    console.error('Error:', err);
    process.exit(1);
});
