/**
 * Face Detection Demo - Sieve Passthrough Example
 *
 * Demonstrates FacialLandmarkCV module which detects faces in images
 * and draws green rectangles around them.
 *
 * Pipeline:
 *   faces.jpg -> Decode -> FaceDetect -> Encode -> Write to file
 *                                    \-> StatSink (for landmarks data)
 *
 * KEY CONCEPT: Sieve Passthrough
 * FacialLandmarkCV declares only FaceLandmarksInfo as output, but with sieve=false
 * (the default), the input RawImage frame passes through automatically.
 * This allows connecting to ImageEncoderCV which expects RawImage input.
 *
 * OBSERVABLE OUTPUT:
 * - Creates face_detected_output.jpg with green rectangles around detected faces
 *
 * Run with: node face_detection_demo.js
 */

const path = require('path');
const fs = require('fs');

// Load the addon
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

// Paths
const inputImage = path.join(__dirname, '../../data/faces.jpg');
const outputImage = path.join(__dirname, '../../data/face_detected_output.jpg');

// Verify input exists
if (!fs.existsSync(inputImage)) {
    console.error(`Input file not found: ${inputImage}`);
    process.exit(1);
}

console.log('=== Face Detection Demo (Sieve Passthrough) ===\n');
console.log(`Input:  ${inputImage}`);
console.log(`Output: ${outputImage}`);
console.log('');

// Pipeline configuration demonstrating sieve passthrough
// FacialLandmarkCV:
// - Declared output: FaceLandmarksInfo (landmark points)
// - With sieve=false (default): RawImage also passes through
// This allows connecting to ImageEncoderCV which expects RawImage
const pipelineConfig = {
    name: "FaceDetectionPipeline",
    modules: {
        // Read the JPEG image file
        reader: {
            type: "FileReaderModule",
            props: {
                strFullFileNameWithPattern: inputImage,
                outputFrameType: "EncodedImage"
            }
        },

        // Decode JPEG to raw RGB image
        decoder: {
            type: "ImageDecoderCV"
        },

        // Detect faces and draw green rectangles
        // Uses SSD (Single Shot Detector) model by default
        // Declared output: FaceLandmarksInfo
        // With sieve=false: also passes through RawImage
        faceDetector: {
            type: "FacialLandmarkCV",
            props: {
                modelType: "SSD",
                faceDetectionConfig: path.join(__dirname, '../../data/assets/deploy.prototxt'),
                faceDetectionWeights: path.join(__dirname, '../../data/assets/res10_300x300_ssd_iter_140000_fp16.caffemodel'),
                landmarksModel: path.join(__dirname, '../../data/assets/face_landmark_model.dat')
            }
        },

        // Encode the passed-through RawImage to JPEG
        // This works because of sieve passthrough - RawImage flows through faceDetector
        encoder: {
            type: "ImageEncoderCV"
        },

        // Write the JPEG to file
        writer: {
            type: "FileWriterModule",
            props: {
                strFullFileNameWithPattern: outputImage
            }
        },

        // StatSink accepts any frames (for the landmarks data)
        landmarksSink: {
            type: "StatSink"
        }
    },
    connections: [
        { from: "reader", to: "decoder" },
        { from: "decoder", to: "faceDetector" },
        // sieve=false (default) - RawImage passes through to encoder
        { from: "faceDetector", to: "encoder" },
        { from: "encoder", to: "writer" },
        // Also send landmarks data to StatSink
        // (Would need pin-specific connection for landmarks only)
        { from: "faceDetector", to: "landmarksSink" }
    ]
};

async function main() {
    console.log('Pipeline structure:');
    console.log('  [FileReaderModule] faces.jpg');
    console.log('         |');
    console.log('         v');
    console.log('  [ImageDecoderCV] JPEG -> RawImage');
    console.log('         |');
    console.log('         v');
    console.log('  [FacialLandmarkCV] <- Draws green rectangles');
    console.log('         | (sieve=false: RawImage passes through)');
    console.log('         |');
    console.log('         +---- RawImage (passthrough) ---> [ImageEncoderCV] -> [FileWriterModule]');
    console.log('         |');
    console.log('         +---- FaceLandmarksInfo --------> [StatSink]');
    console.log('');

    // Create and configure pipeline
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

    // Run for enough time to process the image
    console.log('Processing image (face detection with SSD model)...');
    pipeline.run();

    // Wait for processing
    await new Promise(resolve => setTimeout(resolve, 3000));

    // Stop
    console.log('Stopping pipeline...');
    await pipeline.stop();

    // Check if output was created
    if (fs.existsSync(outputImage)) {
        const stats = fs.statSync(outputImage);
        console.log(`\n=== SUCCESS ===`);
        console.log(`Output file created: ${outputImage} (${stats.size} bytes)`);
        console.log('The output image contains green rectangles around detected faces.');
    } else {
        console.log('\nNote: Output file was not created (may need model files)');
    }

    console.log('\n=== Key Takeaway ===');
    console.log('With sieve=false (default), transform modules pass through their input types.');
    console.log('FacialLandmarkCV declares FaceLandmarksInfo output, but RawImage passes through.');
    console.log('This allows connecting to ImageEncoderCV which expects RawImage input.');
    console.log('To disable passthrough, set sieve: true on the connection.');

    console.log('\n=== Demo Complete ===\n');
}

main().catch(err => {
    console.error('Error:', err);
    process.exit(1);
});
