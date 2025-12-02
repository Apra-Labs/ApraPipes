/**
 * @file thumbnail_generator main.cpp
 * @brief Generate thumbnail images from MP4 video files
 *
 * This sample demonstrates how to extract a single frame from an MP4 video
 * and save it as a JPEG thumbnail image. This is useful for:
 * - Video previews in media libraries
 * - Creating poster images for video players
 * - Quick visual identification of video content
 * - Generating thumbnails for video galleries
 *
 * Pipeline Structure:
 *   [Mp4ReaderSource] → [H264Decoder] → [ValveModule] → [CudaMemCopy] →
 *   [JPEGEncoderNVJPEG] → [FileWriterModule]
 *
 * Key Concept: ValveModule
 * The ValveModule acts as a gate that controls how many frames pass through.
 * By setting it to allow only 1 frame, we extract exactly one thumbnail
 * from the video, regardless of video length.
 *
 * Features Demonstrated:
 * - MP4 video reading
 * - H264 video decoding
 * - Frame filtering with ValveModule
 * - CUDA-accelerated JPEG encoding
 * - File writing
 *
 * Usage:
 *   thumbnail_generator.exe <video_path> <output_path>
 *
 * Example:
 *   thumbnail_generator.exe input.mp4 thumbnail_????.jpg
 *
 * The output path uses ???? as a placeholder for frame numbering.
 * The first frame will be saved as thumbnail_0000.jpg
 */

#include <iostream>
#include <chrono>
#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>

// ApraPipes core
#include "PipeLine.h"
#include "Logger.h"
#include "AIPExceptions.h"

// Source and sink modules
#include "Mp4ReaderSource.h"
#include "FileWriterModule.h"

// Processing modules
#include "H264Decoder.h"
#include "ValveModule.h"
#include "CudaMemCopy.h"
#include "JPEGEncoderNVJPEG.h"

// Metadata
#include "FrameMetadata.h"
#include "H264Metadata.h"
#include "Mp4VideoMetadata.h"

/**
 * @class ThumbnailGenerator
 * @brief Pipeline for extracting a thumbnail image from an MP4 video
 *
 * This class sets up a pipeline that:
 * 1. Reads an MP4 video file
 * 2. Decodes H264 compressed frames
 * 3. Uses a valve to allow exactly 1 frame through
 * 4. Copies frame to GPU memory
 * 5. Encodes as JPEG using NVIDIA JPEG encoder
 * 6. Writes JPEG file to disk
 *
 * The ValveModule is the key component - it acts as a programmable gate
 * that can be configured to allow a specific number of frames to pass.
 * After the specified count, it blocks all subsequent frames.
 */
class ThumbnailGenerator {
public:
    /**
     * @brief Constructor - initializes the pipeline with a name
     */
    ThumbnailGenerator() : pipeline("ThumbnailGeneratorPipeline") {}

    /**
     * @brief Sets up the thumbnail generation pipeline
     *
     * @param videoPath Path to the input MP4 video file
     * @param outFolderPath Output path for the thumbnail (e.g., "thumb_????.jpg")
     * @return true if setup successful, false otherwise
     *
     * Pipeline flow:
     * 1. Mp4ReaderSource: Reads H264 frames from MP4 container
     * 2. H264Decoder: Decodes H264 to raw YUV420 frames
     * 3. ValveModule: Configured to pass only 1 frame (initially set to 0)
     * 4. CudaMemCopy: Transfers frame from host to device memory
     * 5. JPEGEncoderNVJPEG: Encodes frame to JPEG on GPU
     * 6. FileWriterModule: Writes JPEG to disk
     */
    bool setupPipeline(const std::string &videoPath, const std::string &outFolderPath) {
        std::cout << "\n╔══════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║     ApraPipes Sample: Thumbnail Generator                   ║" << std::endl;
        std::cout << "╚══════════════════════════════════════════════════════════════╝\n" << std::endl;

        std::cout << "Setting up thumbnail generation pipeline..." << std::endl;
        std::cout << "  Input video: " << videoPath << std::endl;
        std::cout << "  Output path: " << outFolderPath << std::endl;

        try {
            // 1. Setup MP4 reader source
            std::cout << "\n[1/6] Setting up MP4 reader..." << std::endl;
            bool parseFS = false;  // Don't parse filesystem for frame timestamps
            auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));  // Auto-detect dimensions
            auto frameType = FrameMetadata::FrameType::H264_DATA;

            auto mp4ReaderProps = Mp4ReaderSourceProps(
                videoPath,  // file path
                parseFS,    // parse filesystem
                0,          // start frame (0 = beginning)
                true,       // read loop (doesn't matter for single frame)
                false,      // rewind on loop
                false       // direction (false = forward)
            );

            mp4Reader = boost::shared_ptr<Mp4ReaderSource>(
                new Mp4ReaderSource(mp4ReaderProps)
            );
            mp4Reader->addOutPutPin(h264ImageMetadata);

            // Add MP4 metadata pin
            auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
            mp4Reader->addOutPutPin(mp4Metadata);

            // Get the H264 data pin for connection
            std::vector<std::string> mImagePin;
            mImagePin = mp4Reader->getAllOutputPinsByType(frameType);
            std::cout << "  ✓ MP4 reader configured" << std::endl;

            // 2. Setup H264 decoder
            std::cout << "[2/6] Setting up H264 decoder..." << std::endl;
            decoder = boost::shared_ptr<H264Decoder>(
                new H264Decoder(H264DecoderProps())
            );
            // Connect MP4 reader's H264 pin to decoder
            mp4Reader->setNext(decoder, mImagePin);
            std::cout << "  ✓ H264 decoder configured" << std::endl;

            // 3. Setup valve module (initially closed - allows 0 frames)
            std::cout << "[3/6] Setting up valve module..." << std::endl;
            // ValveModule(0) means initially no frames are allowed through
            // We'll open it to allow 1 frame after pipeline starts
            valve = boost::shared_ptr<ValveModule>(
                new ValveModule(ValveModuleProps(0))
            );
            decoder->setNext(valve);
            std::cout << "  ✓ Valve module configured (initially closed)" << std::endl;

            // 4. Setup CUDA memory copy (Host to Device)
            std::cout << "[4/6] Setting up CUDA memory copy..." << std::endl;
            auto stream = cudastream_sp(new ApraCudaStream);
            cudaCopy = boost::shared_ptr<CudaMemCopy>(
                new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, stream))
            );
            valve->setNext(cudaCopy);
            std::cout << "  ✓ CUDA memory copy configured" << std::endl;

            // 5. Setup JPEG encoder (NVIDIA GPU accelerated)
            std::cout << "[5/6] Setting up JPEG encoder..." << std::endl;
            jpegEncoder = boost::shared_ptr<JPEGEncoderNVJPEG>(
                new JPEGEncoderNVJPEG(JPEGEncoderNVJPEGProps(stream))
            );
            cudaCopy->setNext(jpegEncoder);
            std::cout << "  ✓ JPEG encoder configured" << std::endl;

            // 6. Setup file writer (saves JPEG to disk)
            std::cout << "[6/6] Setting up file writer..." << std::endl;
            fileWriter = boost::shared_ptr<FileWriterModule>(
                new FileWriterModule(FileWriterModuleProps(outFolderPath))
            );
            jpegEncoder->setNext(fileWriter);
            std::cout << "  ✓ File writer configured" << std::endl;

            std::cout << "\n✓ Pipeline setup completed successfully!" << std::endl;
            std::cout << "\nPipeline structure:" << std::endl;
            std::cout << "  [Mp4Reader] → [H264Decoder] → [ValveModule] → [CudaMemCopy] →" << std::endl;
            std::cout << "  [JPEGEncoder] → [FileWriter]" << std::endl;

            return true;

        } catch (const std::exception &e) {
            std::cerr << "\n✗ Error during pipeline setup: " << e.what() << std::endl;
            return false;
        } catch (...) {
            std::cerr << "\n✗ Unknown error during pipeline setup" << std::endl;
            return false;
        }
    }

    /**
     * @brief Starts the pipeline and generates thumbnail
     *
     * @return true if started successfully, false otherwise
     *
     * Process:
     * 1. Add MP4 reader to pipeline
     * 2. Initialize all modules
     * 3. Start pipeline in threaded mode
     * 4. Open valve to allow exactly 1 frame through
     * 5. Wait for processing to complete
     */
    bool startPipeline() {
        try {
            std::cout << "\nInitializing and starting pipeline..." << std::endl;

            // Add the source module to the pipeline
            pipeline.appendModule(mp4Reader);

            // Initialize the pipeline (calls init() on all modules)
            if (!pipeline.init()) {
                throw AIPException(
                    AIP_FATAL,
                    "Pipeline initialization failed. Check logs for details."
                );
                return false;
            }

            // Run pipeline in threaded mode (each module in separate thread)
            pipeline.run_all_threaded();

            std::cout << "✓ Pipeline started successfully!" << std::endl;

            // Open the valve to allow exactly 1 frame to pass through
            std::cout << "\nOpening valve to capture 1 frame..." << std::endl;
            valve->allowFrames(1);
            std::cout << "✓ Valve opened (1 frame allowed)" << std::endl;

            return true;

        } catch (const std::exception &e) {
            std::cerr << "✗ Error starting pipeline: " << e.what() << std::endl;
            return false;
        } catch (...) {
            std::cerr << "✗ Unknown error starting pipeline" << std::endl;
            return false;
        }
    }

    /**
     * @brief Stops the pipeline and cleans up resources
     *
     * @return true if stopped successfully, false otherwise
     */
    bool stopPipeline() {
        try {
            std::cout << "\nStopping pipeline..." << std::endl;

            pipeline.stop();
            pipeline.term();
            pipeline.wait_for_all();

            std::cout << "✓ Pipeline stopped successfully!" << std::endl;

            return true;

        } catch (const std::exception &e) {
            std::cerr << "✗ Error stopping pipeline: " << e.what() << std::endl;
            return false;
        } catch (...) {
            std::cerr << "✗ Unknown error stopping pipeline" << std::endl;
            return false;
        }
    }

private:
    PipeLine pipeline;
    boost::shared_ptr<Mp4ReaderSource> mp4Reader;
    boost::shared_ptr<H264Decoder> decoder;
    boost::shared_ptr<ValveModule> valve;
    boost::shared_ptr<CudaMemCopy> cudaCopy;
    boost::shared_ptr<JPEGEncoderNVJPEG> jpegEncoder;
    boost::shared_ptr<FileWriterModule> fileWriter;
};

/**
 * @brief Main entry point
 *
 * Demonstrates thumbnail generation from an MP4 video file.
 * The first frame is extracted and saved as a JPEG image.
 */
int main(int argc, char *argv[]) {
    // Setup logger
    LoggerProps loggerProps;
    loggerProps.logLevel = boost::log::trivial::severity_level::info;
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
    Logger::initLogger(loggerProps);

    // Validate command line arguments
    if (argc < 3) {
        std::cerr << "Error: Missing required arguments\n" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <video_path> <output_path>" << std::endl;
        std::cerr << "\nExample:" << std::endl;
        std::cerr << "  " << argv[0] << " video.mp4 thumbnail_????.jpg" << std::endl;
        std::cerr << "\nArguments:" << std::endl;
        std::cerr << "  video_path   - Path to input MP4 video file" << std::endl;
        std::cerr << "  output_path  - Output path for thumbnail (use ???? for frame number)" << std::endl;
        std::cerr << "\nNotes:" << std::endl;
        std::cerr << "  - The output path should include ???? which will be replaced with 0000" << std::endl;
        std::cerr << "  - Only the first frame of the video will be saved" << std::endl;
        std::cerr << "  - Output format is JPEG" << std::endl;
        return 1;
    }

    std::string videoPath = argv[1];
    std::string outFolderPath = argv[2];

    // Create and setup thumbnail generator
    ThumbnailGenerator thumbnailGenerator;

    if (!thumbnailGenerator.setupPipeline(videoPath, outFolderPath)) {
        std::cerr << "\nFailed to setup pipeline. Exiting..." << std::endl;
        return 1;
    }

    if (!thumbnailGenerator.startPipeline()) {
        std::cerr << "\nFailed to start pipeline. Exiting..." << std::endl;
        return 1;
    }

    // Wait for thumbnail generation to complete
    // 5 seconds should be enough for reading one frame and encoding it
    std::cout << "\nWaiting for thumbnail generation..." << std::endl;
    boost::this_thread::sleep_for(boost::chrono::seconds(5));

    // Stop the pipeline
    if (!thumbnailGenerator.stopPipeline()) {
        std::cerr << "\nFailed to stop pipeline cleanly." << std::endl;
        return 1;
    }

    std::cout << "\n╔══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║     Thumbnail Generated Successfully!                        ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << "\nOutput saved to: " << outFolderPath << std::endl;
    std::cout << "(Replace ???? with 0000 to see the actual filename)\n" << std::endl;

    return 0;
}
