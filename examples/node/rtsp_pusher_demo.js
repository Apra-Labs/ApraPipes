/**
 * RTSP Pusher Demo
 *
 * Demonstrates RTSPPusher module which streams H264 video to an RTSP server.
 * Reads H264 encoded video from an MP4 file and pushes it to RTSP.
 *
 * PREREQUISITE:
 * - An RTSP server must be running to receive the stream
 * - Default URL: rtsp://192.168.1.11:5544/unit_test
 * - You can use mediamtx: brew install mediamtx && mediamtx
 *
 * OBSERVABLE OUTPUT:
 * - Open the RTSP stream in VLC or ffplay to view the live video
 * - Example: ffplay rtsp://192.168.1.11:5544/unit_test
 *
 * Run with: node rtsp_pusher_demo.js [mp4_file] [rtsp://server:port/path]
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

// Get MP4 file and RTSP URL from command line or use defaults
const defaultMp4 = path.join(__dirname, '../../data/Mp4_videos/h264_video/apraH264.mp4');
const mp4File = process.argv[2] || defaultMp4;
const rtspUrl = process.argv[3] || 'rtsp://192.168.1.11:5544/unit_test';

// Verify MP4 file exists
if (!fs.existsSync(mp4File)) {
    console.error(`Error: MP4 file not found: ${mp4File}`);
    console.error('Usage: node rtsp_pusher_demo.js [mp4_file] [rtsp_url]');
    process.exit(1);
}

console.log('=== RTSP Pusher Demo ===\n');
console.log(`Source: ${mp4File}`);
console.log(`Streaming to: ${rtspUrl}`);
console.log('');
console.log('To view the stream, open in VLC or run:');
console.log(`  ffplay ${rtspUrl}`);
console.log('');

// Pipeline: Mp4ReaderSource -> RTSPPusher
// Mp4ReaderSource outputs H264Data which RTSPPusher accepts directly
const pipelineConfig = {
    name: "RTSPStreamingPipeline",
    modules: {
        // Read H264 encoded frames from MP4 file
        source: {
            type: "Mp4ReaderSource",
            props: {
                videoPath: mp4File,
                parseFS: false,
                readLoop: true,  // Loop the video for continuous streaming
                giveLiveTS: true, // Use live timestamps for smooth playback
                outputFormat: "h264"  // Required for declarative pipelines
            }
        },

        // Push to RTSP server
        pusher: {
            type: "RTSPPusher",
            props: {
                url: rtspUrl,
                title: "ApraPipes RTSP Stream",
                isTCP: true,
                encoderTargetKbps: 2048
            }
        }
    },
    connections: [
        { from: "source", to: "pusher" }
    ]
};

async function main() {
    console.log('Pipeline structure:');
    console.log(`  [Mp4ReaderSource] ${path.basename(mp4File)}`);
    console.log('         |');
    console.log('         | (H264 frames)');
    console.log('         v');
    console.log('  [RTSPPusher] -> RTSP Server');
    console.log('');

    // Create pipeline
    console.log('Creating pipeline...');
    const pipeline = ap.createPipeline(pipelineConfig);

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

    // Run
    console.log('Starting stream...');
    console.log('Press Ctrl+C to stop\n');
    pipeline.run();

    // Stream for 60 seconds or until interrupted
    let running = true;
    let elapsed = 0;
    const duration = 60;

    process.on('SIGINT', () => {
        console.log('\nInterrupted by user');
        running = false;
    });

    while (running && elapsed < duration) {
        await new Promise(resolve => setTimeout(resolve, 1000));
        elapsed++;
        process.stdout.write(`\rStreaming... ${elapsed}/${duration}s   `);
    }

    console.log('\n\nStopping pipeline...');
    await pipeline.stop();

    console.log('Stream ended.');
    console.log('\n=== Demo Complete ===\n');
}

main().catch(err => {
    console.error('Error:', err);
    process.exit(1);
});
