/**
 * @file relay.cpp
 * @brief Relay sample demonstrating dynamic source switching
 *
 * This sample demonstrates the "relay" pattern in ApraPipes, which allows
 * dynamically switching between multiple input sources without stopping the pipeline.
 *
 * Pipeline Structure:
 *   [RTSPClientSrc]  ─┐
 *                     ├─> [H264Decoder] → [ColorConversion] → [ImageViewer]
 *   [Mp4ReaderSource]─┘
 *
 * Features Demonstrated:
 * - RTSP streaming from network cameras
 * - MP4 file reading
 * - H264 video decoding
 * - Dynamic source switching using relay()
 * - Interactive keyboard control
 *
 * Usage:
 *   relay.exe <rtsp_url> <mp4_file_path>
 *
 * Keyboard Controls:
 *   'r' - Switch to RTSP source
 *   'm' - Switch to MP4 source
 *   's' - Stop and exit
 *
 * Requirements:
 * - RTSP camera URL (e.g., rtsp://server:port/stream)
 * - MP4 video file
 */

#include <iostream>
#include <chrono>
#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>

// ApraPipes modules
#include "PipeLine.h"
#include "Logger.h"
#include "RTSPClientSrc.h"
#include "Mp4ReaderSource.h"
#include "H264Decoder.h"
#include "ColorConversionXForm.h"
#include "ImageViewerModule.h"
#include "H264Metadata.h"
#include "Mp4VideoMetadata.h"
#include "FrameMetadata.h"

// Keyboard control constants
constexpr int KEY_RTSP = 'r';      // 114
constexpr int KEY_MP4 = 'm';       // 109
constexpr int KEY_STOP = 's';      // 115

/**
 * @class RelayPipeline
 * @brief Manages a pipeline that can switch between RTSP and MP4 sources
 *
 * This class demonstrates the relay pattern, where a processing module (H264Decoder)
 * can receive input from multiple sources, but only one source is active at a time.
 * The relay() method allows switching between sources dynamically.
 *
 * Key Concepts:
 * - Multiple sources can be connected to a single module
 * - Only one source is active at a time
 * - relay(module, true) enables a source
 * - relay(module, false) disables a source
 * - Switching is done at runtime without stopping the pipeline
 */
class RelayPipeline {
public:
    /**
     * @brief Constructor - initializes the pipeline with a name
     */
    RelayPipeline() : pipeline("RelaySample") {}

    /**
     * @brief Sets up the pipeline modules and connections
     *
     * @param rtspUrl URL of the RTSP camera stream
     * @param mp4VideoPath Path to the MP4 video file
     * @return true if setup successful, false otherwise
     *
     * Pipeline setup:
     * 1. RTSP source (network camera) → H264 decoder
     * 2. MP4 source (file) → H264 decoder
     * 3. H264 decoder → Color conversion → Image viewer
     *
     * Both sources output H264 compressed video, which is decoded and displayed.
     */
    bool setupPipeline(const std::string &rtspUrl, const std::string &mp4VideoPath) {
        std::cout << "\n╔══════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║     ApraPipes Sample: Relay (Source Switching)              ║" << std::endl;
        std::cout << "╚══════════════════════════════════════════════════════════════╝\n" << std::endl;

        std::cout << "Setting up relay pipeline..." << std::endl;
        std::cout << "  RTSP URL: " << rtspUrl << std::endl;
        std::cout << "  MP4 Path: " << mp4VideoPath << std::endl;

        try {
            // 1. Setup RTSP source (network camera)
            std::cout << "\n[1/5] Setting up RTSP source..." << std::endl;
            rtspSource = boost::shared_ptr<RTSPClientSrc>(
                new RTSPClientSrc(RTSPClientSrcProps(rtspUrl, "", ""))
            );
            auto rtspMetaData = framemetadata_sp(new H264Metadata(1280, 720));
            rtspSource->addOutputPin(rtspMetaData);
            std::cout << "  ✓ RTSP source configured" << std::endl;

            // 2. Setup MP4 source (file reader)
            std::cout << "[2/5] Setting up MP4 source..." << std::endl;
            bool parseFS = false;
            auto h264ImageMetadata = framemetadata_sp(new H264Metadata(1280, 720));
            auto frameType = FrameMetadata::FrameType::H264_DATA;

            auto mp4ReaderProps = Mp4ReaderSourceProps(
                mp4VideoPath,  // file path
                parseFS,       // parse filesystem
                0,             // start frame
                true,          // read loop
                true,          // rewind on loop
                false          // direction (forward)
            );
            mp4ReaderProps.fps = 9;  // Playback at 9 fps

            mp4ReaderSource = boost::shared_ptr<Mp4ReaderSource>(
                new Mp4ReaderSource(mp4ReaderProps)
            );
            mp4ReaderSource->addOutputPin(h264ImageMetadata);  // Fixed: was addOutPutPin

            auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
            mp4ReaderSource->addOutputPin(mp4Metadata);

            std::vector<std::string> mImagePin;
            mImagePin = mp4ReaderSource->getAllOutputPinsByType(frameType);
            std::cout << "  ✓ MP4 source configured" << std::endl;

            // 3. Setup H264 decoder (receives from both sources)
            std::cout << "[3/5] Setting up H264 decoder..." << std::endl;
            h264Decoder = boost::shared_ptr<H264Decoder>(
                new H264Decoder(H264DecoderProps())
            );

            // Connect both sources to the decoder
            rtspSource->setNext(h264Decoder);
            mp4ReaderSource->setNext(h264Decoder, mImagePin);
            std::cout << "  ✓ H264 decoder configured with dual inputs" << std::endl;

            // 4. Setup color conversion (YUV420 to RGB)
            std::cout << "[4/5] Setting up color conversion..." << std::endl;
            colorConversion = boost::shared_ptr<ColorConversion>(
                new ColorConversion(
                    ColorConversionProps(ColorConversionProps::YUV420PLANAR_TO_RGB)
                )
            );
            h264Decoder->setNext(colorConversion);
            std::cout << "  ✓ Color conversion configured" << std::endl;

            // 5. Setup image viewer (display window)
            std::cout << "[5/5] Setting up image viewer..." << std::endl;
            imageViewer = boost::shared_ptr<ImageViewerModule>(
                new ImageViewerModule(ImageViewerModuleProps("Relay Sample"))
            );
            colorConversion->setNext(imageViewer);
            std::cout << "  ✓ Image viewer configured" << std::endl;

            std::cout << "\n✓ Pipeline setup completed successfully!" << std::endl;
            std::cout << "\nPipeline structure:" << std::endl;
            std::cout << "  [RTSPClientSrc]  ─┐" << std::endl;
            std::cout << "                     ├─> [H264Decoder] → [ColorConversion] → [ImageViewer]" << std::endl;
            std::cout << "  [Mp4ReaderSource]─┘" << std::endl;

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
     *
     * By default, the pipeline starts with MP4 source active (RTSP disabled).
     * Use keyboard controls to switch between sources.
     */
    bool startPipeline() {
        try {
            std::cout << "\nInitializing and starting pipeline..." << std::endl;

            pipeline.appendModule(rtspSource);
            pipeline.appendModule(mp4ReaderSource);
            pipeline.init();
            pipeline.run_all_threaded();

            // Start with MP4 source active, RTSP disabled
            addRelayToMp4(false);  // Disable MP4 relay (RTSP active by default)

            std::cout << "✓ Pipeline started successfully!" << std::endl;
            std::cout << "\nDefault source: RTSP" << std::endl;
            std::cout << "\nKeyboard Controls:" << std::endl;
            std::cout << "  'r' - Switch to RTSP source" << std::endl;
            std::cout << "  'm' - Switch to MP4 source" << std::endl;
            std::cout << "  's' - Stop and exit\n" << std::endl;

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
     * @brief Enable or disable RTSP source
     *
     * @param open true to enable RTSP source, false to disable
     *
     * The relay() method controls whether this source feeds data to the decoder.
     * Only one source should be enabled at a time.
     */
    void addRelayToRtsp(bool open) {
        rtspSource->relay(h264Decoder, open);
        if (open) {
            std::cout << "→ Switched to RTSP source" << std::endl;
        }
    }

    /**
     * @brief Enable or disable MP4 source
     *
     * @param open true to enable MP4 source, false to disable
     *
     * The relay() method controls whether this source feeds data to the decoder.
     * Only one source should be enabled at a time.
     */
    void addRelayToMp4(bool open) {
        mp4ReaderSource->relay(h264Decoder, open);
        if (open) {
            std::cout << "→ Switched to MP4 source" << std::endl;
        }
    }

private:
    PipeLine pipeline;
    boost::shared_ptr<RTSPClientSrc> rtspSource;
    boost::shared_ptr<Mp4ReaderSource> mp4ReaderSource;
    boost::shared_ptr<H264Decoder> h264Decoder;
    boost::shared_ptr<ColorConversion> colorConversion;
    boost::shared_ptr<ImageViewerModule> imageViewer;
};

/**
 * @brief Main entry point
 *
 * Demonstrates dynamic source switching between RTSP camera and MP4 file.
 * The user can interactively switch sources using keyboard controls.
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
        std::cerr << "Usage: " << argv[0] << " <rtsp_url> <mp4_file_path>" << std::endl;
        std::cerr << "\nExample:" << std::endl;
        std::cerr << "  " << argv[0] << " rtsp://192.168.1.100:554/stream video.mp4" << std::endl;
        std::cerr << "\nArguments:" << std::endl;
        std::cerr << "  rtsp_url        - URL of RTSP camera stream" << std::endl;
        std::cerr << "  mp4_file_path   - Path to MP4 video file" << std::endl;
        return 1;
    }

    std::string rtspUrl = argv[1];
    std::string mp4VideoPath = argv[2];

    // Create and setup pipeline
    RelayPipeline pipelineInstance;

    if (!pipelineInstance.setupPipeline(rtspUrl, mp4VideoPath)) {
        std::cerr << "\nFailed to setup pipeline. Exiting..." << std::endl;
        return 1;
    }

    if (!pipelineInstance.startPipeline()) {
        std::cerr << "\nFailed to start pipeline. Exiting..." << std::endl;
        return 1;
    }

    // Interactive keyboard control loop
    std::cout << "Waiting for keyboard input..." << std::endl;
    while (true) {
        int key = getchar();

        switch (key) {
            case KEY_RTSP:
                // Switch to RTSP source
                pipelineInstance.addRelayToMp4(false);   // Disable MP4
                pipelineInstance.addRelayToRtsp(true);   // Enable RTSP
                break;

            case KEY_MP4:
                // Switch to MP4 source
                pipelineInstance.addRelayToRtsp(false);  // Disable RTSP
                pipelineInstance.addRelayToMp4(true);    // Enable MP4
                break;

            case KEY_STOP:
                // Stop and exit
                std::cout << "\nStop command received" << std::endl;
                if (!pipelineInstance.stopPipeline()) {
                    std::cerr << "\nFailed to stop pipeline cleanly." << std::endl;
                    return 1;
                }
                goto exit_loop;

            case '\n':
            case '\r':
                // Ignore newlines
                break;

            default:
                if (key >= 32 && key <= 126) {  // Printable ASCII
                    std::cout << "Unknown key: '" << static_cast<char>(key)
                              << "' (press 'r', 'm', or 's')" << std::endl;
                }
                break;
        }
    }

exit_loop:
    std::cout << "\n╔══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║     Relay Sample Completed Successfully!                    ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n" << std::endl;

    return 0;
}
