/**
 * @file file_reader main.cpp
 * @brief Play MP4 video files with seek functionality
 *
 * This sample demonstrates basic MP4 video playback with seeking capability.
 * It shows how to:
 * - Read and play MP4 video files
 * - Decode H264 video streams
 * - Display video frames in a window
 * - Seek to specific timestamps in the video
 * - Handle pipeline flush operations
 *
 * Pipeline Structure:
 *   [Mp4ReaderSource] → [H264Decoder] → [ColorConversion] → [ImageViewerModule]
 *
 * Key Concepts:
 * - MP4 file reading and playback
 * - Video seeking with timestamp-based navigation
 * - Pipeline queue flushing for clean seeks
 * - Frame rate control for smooth playback
 *
 * Features Demonstrated:
 * - MP4 video file reading
 * - H264 video decoding
 * - Color space conversion (YUV420 to RGB)
 * - Video display in window
 * - Seek functionality (jump to timestamp)
 *
 * Usage:
 *   file_reader.exe <video_path>
 *
 * Example:
 *   file_reader.exe video.mp4
 *
 * The sample will:
 * 1. Play the video for 3 seconds
 * 2. Seek to a specific timestamp
 * 3. Continue playing for 5 more seconds
 * 4. Stop playback
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
#include "ImageViewerModule.h"

// Processing modules
#include "H264Decoder.h"
#include "ColorConversionXForm.h"

// Metadata
#include "FrameMetadata.h"
#include "H264Metadata.h"
#include "Mp4VideoMetadata.h"
#include "RawImagePlanarMetadata.h"

/**
 * @class Mp4FileReader
 * @brief Pipeline for playing MP4 video files with seek capability
 *
 * This class sets up a simple but complete video playback pipeline that:
 * 1. Reads H264-encoded frames from an MP4 file
 * 2. Decodes H264 frames to raw YUV420 format
 * 3. Converts YUV420 to RGB for display
 * 4. Displays frames in an OpenCV window
 *
 * The pipeline supports seeking to any timestamp in the video by:
 * 1. Flushing all queues to clear buffered frames
 * 2. Using randomSeek() to jump to the desired timestamp
 * 3. Resuming playback from the new position
 */
class Mp4FileReader {
public:
    /**
     * @brief Constructor - initializes the pipeline with a name
     */
    Mp4FileReader() : pipeline("Mp4FileReaderPipeline") {}

    /**
     * @brief Sets up the video playback pipeline
     *
     * @param videoPath Path to the input MP4 video file
     * @return true if setup successful, false otherwise
     *
     * Pipeline flow:
     * 1. Mp4ReaderSource: Reads H264 frames from MP4 container
     * 2. H264Decoder: Decodes H264 to raw YUV420 frames
     * 3. ColorConversion: Converts YUV420 to RGB for display
     * 4. ImageViewerModule: Displays frames in OpenCV window
     */
    bool setupPipeline(const std::string &videoPath) {
        std::cout << "\n╔══════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║     ApraPipes Sample: MP4 File Reader                       ║" << std::endl;
        std::cout << "╚══════════════════════════════════════════════════════════════╝\n" << std::endl;

        std::cout << "Setting up MP4 playback pipeline..." << std::endl;
        std::cout << "  Input video: " << videoPath << std::endl;

        try {
            // 1. Setup MP4 reader source
            std::cout << "\n[1/4] Setting up MP4 reader..." << std::endl;
            bool parseFS = false;  // Don't parse filesystem for frame timestamps
            auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));  // Auto-detect dimensions
            auto frameType = FrameMetadata::FrameType::H264_DATA;

            auto mp4ReaderProps = Mp4ReaderSourceProps(
                videoPath,  // file path
                parseFS,    // parse filesystem
                0,          // start frame (0 = beginning)
                true,       // read loop
                true,       // rewind on loop
                false       // direction (false = forward)
            );
            mp4ReaderProps.fps = 24;  // Playback at 24 fps

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
            std::cout << "  ✓ MP4 reader configured (fps=24)" << std::endl;

            // 2. Setup H264 decoder
            std::cout << "[2/4] Setting up H264 decoder..." << std::endl;
            decoder = boost::shared_ptr<H264Decoder>(
                new H264Decoder(H264DecoderProps())
            );
            // Connect MP4 reader's H264 pin to decoder
            mp4Reader->setNext(decoder, mImagePin);
            std::cout << "  ✓ H264 decoder configured" << std::endl;

            // 3. Setup color conversion (YUV420 PLANAR to RGB)
            std::cout << "[3/4] Setting up color conversion..." << std::endl;
            auto conversionType = ColorConversionProps::ConversionType::YUV420PLANAR_TO_RGB;

            colorConversion = boost::shared_ptr<ColorConversion>(
                new ColorConversion(ColorConversionProps(conversionType))
            );
            decoder->setNext(colorConversion);
            std::cout << "  ✓ Color conversion configured (YUV420→RGB)" << std::endl;

            // 4. Setup image viewer (display window)
            std::cout << "[4/4] Setting up image viewer..." << std::endl;
            imageViewer = boost::shared_ptr<ImageViewerModule>(
                new ImageViewerModule(ImageViewerModuleProps("MP4 File Reader"))
            );
            colorConversion->setNext(imageViewer);
            std::cout << "  ✓ Image viewer configured" << std::endl;

            std::cout << "\n✓ Pipeline setup completed successfully!" << std::endl;
            std::cout << "\nPipeline structure:" << std::endl;
            std::cout << "  [Mp4Reader] → [H264Decoder] → [ColorConversion] → [ImageViewer]" << std::endl;

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
     * @brief Starts the pipeline execution
     *
     * @return true if started successfully, false otherwise
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
            std::cout << "\nVideo playback started..." << std::endl;

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

    /**
     * @brief Flush queues and seek to a specific timestamp
     *
     * @param timestamp The timestamp to seek to (in milliseconds)
     * @return true if seek successful, false otherwise
     *
     * This operation:
     * 1. Flushes all pipeline queues to clear buffered frames
     * 2. Seeks to the specified timestamp in the video
     * 3. Resumes playback from the new position
     *
     * Note: Flushing is important to ensure clean seeks. Without flushing,
     * old buffered frames would still be in the pipeline and displayed first.
     */
    bool flushAndSeek(uint64_t timestamp) {
        try {
            std::cout << "\nPerforming seek operation..." << std::endl;
            std::cout << "  Target timestamp: " << timestamp << " ms" << std::endl;

            // Flush all pipeline queues
            std::cout << "  Flushing pipeline queues..." << std::endl;
            pipeline.flushAllQueues();
            std::cout << "  ✓ Queues flushed" << std::endl;

            // Seek to timestamp
            std::cout << "  Seeking to timestamp..." << std::endl;
            mp4Reader->randomSeek(timestamp, false);  // false = forward seek
            std::cout << "  ✓ Seek completed" << std::endl;

            std::cout << "✓ Seek operation successful!" << std::endl;
            std::cout << "Resuming playback from new position..." << std::endl;

            return true;

        } catch (const std::exception &e) {
            std::cerr << "✗ Error during seek: " << e.what() << std::endl;
            return false;
        } catch (...) {
            std::cerr << "✗ Unknown error during seek" << std::endl;
            return false;
        }
    }

    // Public member access for testing (like in reference code)
    boost::shared_ptr<Mp4ReaderSource> mp4Reader;
    boost::shared_ptr<H264Decoder> decoder;
    boost::shared_ptr<ColorConversion> colorConversion;
    boost::shared_ptr<ImageViewerModule> imageViewer;

private:
    PipeLine pipeline;
};

/**
 * @brief Main entry point
 *
 * Demonstrates MP4 video playback with seeking capability.
 * Plays video, seeks to a specific timestamp, then continues playing.
 */
int main(int argc, char *argv[]) {
    // Setup logger
    LoggerProps loggerProps;
    loggerProps.logLevel = boost::log::trivial::severity_level::info;
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
    Logger::initLogger(loggerProps);

    // Validate command line arguments
    if (argc < 2) {
        std::cerr << "Error: Missing required argument\n" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <video_path>" << std::endl;
        std::cerr << "\nExample:" << std::endl;
        std::cerr << "  " << argv[0] << " video.mp4" << std::endl;
        std::cerr << "\nArguments:" << std::endl;
        std::cerr << "  video_path - Path to input MP4 video file" << std::endl;
        std::cerr << "\nNotes:" << std::endl;
        std::cerr << "  - Input must be H264-encoded MP4 video" << std::endl;
        std::cerr << "  - Video will play for 3 seconds, seek, then play 5 more seconds" << std::endl;
        std::cerr << "  - Close the video window or press Ctrl+C to stop early" << std::endl;
        return 1;
    }

    std::string videoPath = argv[1];

    // Create and setup file reader
    Mp4FileReader fileReader;

    if (!fileReader.setupPipeline(videoPath)) {
        std::cerr << "\nFailed to setup pipeline. Exiting..." << std::endl;
        return 1;
    }

    if (!fileReader.startPipeline()) {
        std::cerr << "\nFailed to start pipeline. Exiting..." << std::endl;
        return 1;
    }

    // Play for 3 seconds
    std::cout << "\n[Phase 1] Playing video from beginning..." << std::endl;
    boost::this_thread::sleep_for(boost::chrono::seconds(3));

    // Demonstrate seek functionality
    // Note: This timestamp should be adjusted based on your actual video
    // Using a sample timestamp from the reference code
    uint64_t seekTimestamp = 1686723796848;  // Example timestamp in milliseconds

    std::cout << "\n[Phase 2] Demonstrating seek functionality..." << std::endl;
    if (!fileReader.flushAndSeek(seekTimestamp)) {
        std::cerr << "\nSeek operation failed. Continuing with current playback..." << std::endl;
    }

    // Continue playing for 5 more seconds
    std::cout << "\n[Phase 3] Continuing playback after seek..." << std::endl;
    boost::this_thread::sleep_for(boost::chrono::seconds(5));

    // Stop the pipeline
    if (!fileReader.stopPipeline()) {
        std::cerr << "\nFailed to stop pipeline cleanly." << std::endl;
        return 1;
    }

    std::cout << "\n╔══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║     File Reader Sample Completed Successfully!              ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << "\nKey Concepts Demonstrated:" << std::endl;
    std::cout << "  ✓ MP4 file reading and playback" << std::endl;
    std::cout << "  ✓ H264 video decoding" << std::endl;
    std::cout << "  ✓ Color space conversion" << std::endl;
    std::cout << "  ✓ Video display" << std::endl;
    std::cout << "  ✓ Pipeline queue flushing" << std::endl;
    std::cout << "  ✓ Timestamp-based seeking\n" << std::endl;

    return 0;
}
