/**
 * @file timelapse main.cpp
 * @brief Generate timelapse/summary videos by extracting frames with significant motion
 *
 * This sample demonstrates how to create a timelapse video from a longer input video
 * by intelligently selecting only frames that contain significant motion. This is useful for:
 * - Compressing hours of surveillance footage into minutes
 * - Creating time-lapse videos from long recordings
 * - Generating video summaries that skip static scenes
 * - Reducing storage requirements while preserving important events
 *
 * Pipeline Structure:
 *   [Mp4ReaderSource] → [MotionVectorExtractor] → [ColorConversion] →
 *   [ColorConversion] → [CudaMemCopy] → [CudaStreamSync] → [H264Encoder] →
 *   [Mp4WriterSink]
 *
 * Key Concept: MotionVectorExtractor
 * The MotionVectorExtractor analyzes motion vectors in H264 compressed video
 * and outputs only frames that exceed a motion threshold. This allows creating
 * a summary video without decoding/analyzing every single frame.
 *
 * Features Demonstrated:
 * - MP4 video reading
 * - Motion-based frame extraction
 * - Multiple color space conversions (BGR → RGB → YUV420)
 * - CUDA memory operations
 * - Hardware-accelerated H264 encoding
 * - MP4 video writing
 *
 * Usage:
 *   timelapse.exe <input_video_path> <output_folder_path>
 *
 * Example:
 *   timelapse.exe surveillance_8hours.mp4 timelapse_output/
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
#include "Mp4WriterSink.h"

// Processing modules
#include "MotionVectorExtractor.h"
#include "ColorConversionXForm.h"
#include "CudaMemCopy.h"
#include "CudaStreamSynchronize.h"
#include "H264EncoderNVCodec.h"

// Metadata
#include "FrameMetadata.h"
#include "H264Metadata.h"
#include "Mp4VideoMetadata.h"
#include "CudaCommon.h"

/**
 * @class TimelapsePipeline
 * @brief Pipeline for generating timelapse videos with motion-based frame selection
 *
 * This class sets up a sophisticated pipeline that:
 * 1. Reads H264 video from MP4 file
 * 2. Extracts frames with significant motion using MotionVectorExtractor
 * 3. Converts color spaces for H264 encoding (BGR → RGB → YUV420)
 * 4. Transfers to GPU and encodes to H264
 * 5. Writes output to new MP4 file
 *
 * The MotionVectorExtractor is the intelligence of this pipeline - it analyzes
 * motion vectors embedded in H264 streams and filters out static frames,
 * dramatically reducing output video length while preserving action.
 */
class TimelapsePipeline {
public:
    /**
     * @brief Constructor - initializes pipeline and CUDA resources
     */
    TimelapsePipeline()
        : pipeline("TimelapseSamplePipeline"),
          cudaStream(new ApraCudaStream()),
          cuContext(new ApraCUcontext()),
          h264ImageMetadata(new H264Metadata(0, 0)) {}

    /**
     * @brief Sets up the timelapse generation pipeline
     *
     * @param videoPath Path to input MP4 video file
     * @param outFolderPath Output folder for generated timelapse video
     * @return true if setup successful, false otherwise
     *
     * Pipeline flow:
     * 1. Mp4ReaderSource: Reads H264 frames from input MP4
     * 2. MotionVectorExtractor: Analyzes motion and outputs frames above threshold
     * 3. ColorConversion (BGR→RGB): First color space conversion
     * 4. ColorConversion (RGB→YUV420): Prepare for H264 encoding
     * 5. CudaMemCopy: Transfer to GPU memory
     * 6. CudaStreamSynchronize: Ensure CUDA operations complete
     * 7. H264EncoderNVCodec: Encode to H264 on GPU
     * 8. Mp4WriterSink: Write H264 stream to output MP4 file
     */
    bool setupPipeline(const std::string &videoPath, const std::string &outFolderPath) {
        std::cout << "\n╔══════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║     ApraPipes Sample: Timelapse Generator                   ║" << std::endl;
        std::cout << "╚══════════════════════════════════════════════════════════════╝\n" << std::endl;

        std::cout << "Setting up timelapse generation pipeline..." << std::endl;
        std::cout << "  Input video: " << videoPath << std::endl;
        std::cout << "  Output folder: " << outFolderPath << std::endl;

        try {
            // H264 encoder configuration
            uint32_t gopLength = 25;           // Group of Pictures size
            uint32_t bitRateKbps = 1000;       // Output bitrate (1 Mbps)
            uint32_t frameRate = 30;           // Output frame rate
            H264EncoderNVCodecProps::H264CodecProfile profile = H264EncoderNVCodecProps::MAIN;
            bool enableBFrames = false;        // Don't use B-frames for simplicity
            bool sendDecodedFrames = true;     // MotionExtractor outputs decoded frames

            // 1. Setup MP4 reader source
            std::cout << "\n[1/8] Setting up MP4 reader..." << std::endl;
            auto mp4ReaderProps = Mp4ReaderSourceProps(
                videoPath,  // file path
                false,      // parseFS - parse filesystem
                0,          // start frame
                true,       // read loop
                false,      // rewind on loop
                false       // direction (forward)
            );
            mp4ReaderProps.parseFS = true;    // Parse filesystem for proper timestamps
            mp4ReaderProps.readLoop = false;  // Don't loop - process once

            mp4Reader = boost::shared_ptr<Mp4ReaderSource>(
                new Mp4ReaderSource(mp4ReaderProps)
            );
            mp4Reader->addOutPutPin(h264ImageMetadata);

            auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
            mp4Reader->addOutPutPin(mp4Metadata);
            std::cout << "  ✓ MP4 reader configured" << std::endl;

            // 2. Setup Motion Vector Extractor
            std::cout << "[2/8] Setting up motion vector extractor..." << std::endl;
            auto motionExtractorProps = MotionVectorExtractorProps(
                MotionVectorExtractorProps::MVExtractMethod::OPENH264,  // Use OpenH264 for extraction
                sendDecodedFrames,  // Send decoded frames (not just motion vectors)
                2                   // Motion threshold (frames with motion > 2 are selected)
            );

            motionExtractor = boost::shared_ptr<MotionVectorExtractor>(
                new MotionVectorExtractor(motionExtractorProps)
            );

            std::vector<std::string> mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::H264_DATA);
            mp4Reader->setNext(motionExtractor, mImagePin);
            std::cout << "  ✓ Motion vector extractor configured (threshold=2)" << std::endl;

            // 3. Setup first color conversion (BGR to RGB)
            std::cout << "[3/8] Setting up color conversion (BGR→RGB)..." << std::endl;
            colorChange1 = boost::shared_ptr<ColorConversion>(
                new ColorConversion(ColorConversionProps(ColorConversionProps::BGR_TO_RGB))
            );

            std::vector<std::string> mDecodedPin = motionExtractor->getAllOutputPinsByType(FrameMetadata::RAW_IMAGE);
            motionExtractor->setNext(colorChange1, mDecodedPin);
            std::cout << "  ✓ First color conversion configured" << std::endl;

            // 4. Setup second color conversion (RGB to YUV420 PLANAR)
            std::cout << "[4/8] Setting up color conversion (RGB→YUV420)..." << std::endl;
            colorChange2 = boost::shared_ptr<ColorConversion>(
                new ColorConversion(ColorConversionProps(ColorConversionProps::RGB_TO_YUV420PLANAR))
            );
            colorChange1->setNext(colorChange2);
            std::cout << "  ✓ Second color conversion configured" << std::endl;

            // 5. Setup CUDA memory copy (Host to Device)
            std::cout << "[5/8] Setting up CUDA memory copy..." << std::endl;
            cudaCopy = boost::shared_ptr<Module>(
                new CudaMemCopy(CudaMemCopyProps(cudaMemcpyHostToDevice, cudaStream))
            );
            colorChange2->setNext(cudaCopy);
            std::cout << "  ✓ CUDA memory copy configured" << std::endl;

            // 6. Setup CUDA stream synchronization
            std::cout << "[6/8] Setting up CUDA stream synchronization..." << std::endl;
            sync = boost::shared_ptr<Module>(
                new CudaStreamSynchronize(CudaStreamSynchronizeProps(cudaStream))
            );
            cudaCopy->setNext(sync);
            std::cout << "  ✓ CUDA stream synchronization configured" << std::endl;

            // 7. Setup H264 encoder
            std::cout << "[7/8] Setting up H264 encoder..." << std::endl;
            encoder = boost::shared_ptr<Module>(
                new H264EncoderNVCodec(
                    H264EncoderNVCodecProps(bitRateKbps, cuContext, gopLength, frameRate, profile, enableBFrames)
                )
            );
            sync->setNext(encoder);
            std::cout << "  ✓ H264 encoder configured (bitrate=" << bitRateKbps << " kbps, fps=" << frameRate << ")" << std::endl;

            // 8. Setup MP4 writer sink
            std::cout << "[8/8] Setting up MP4 writer..." << std::endl;
            auto mp4WriterSinkProps = Mp4WriterSinkProps(
                UINT32_MAX,         // chunkTimeInMins (unlimited)
                10,                 // syncTimeInSecs
                24,                 // fps
                outFolderPath,      // base folder path
                true                // enableLiveMode
            );
            mp4WriterSinkProps.recordedTSBasedDTS = false;  // Use frame-based timestamps

            mp4WriterSink = boost::shared_ptr<Module>(
                new Mp4WriterSink(mp4WriterSinkProps)
            );
            encoder->setNext(mp4WriterSink);
            std::cout << "  ✓ MP4 writer configured" << std::endl;

            std::cout << "\n✓ Pipeline setup completed successfully!" << std::endl;
            std::cout << "\nPipeline structure:" << std::endl;
            std::cout << "  [Mp4Reader] → [MotionExtractor] → [BGR→RGB] → [RGB→YUV420] →" << std::endl;
            std::cout << "  [CudaCopy] → [CudaSync] → [H264Encoder] → [Mp4Writer]" << std::endl;

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

            pipeline.appendModule(mp4Reader);

            if (!pipeline.init()) {
                throw AIPException(
                    AIP_FATAL,
                    "Pipeline initialization failed. Check logs for details."
                );
                return false;
            }

            pipeline.run_all_threaded();

            std::cout << "✓ Pipeline started successfully!" << std::endl;
            std::cout << "\nProcessing video..." << std::endl;
            std::cout << "  - Analyzing motion in each frame" << std::endl;
            std::cout << "  - Extracting frames with significant motion" << std::endl;
            std::cout << "  - Encoding and writing output video" << std::endl;
            std::cout << "\nThis may take a while for long videos..." << std::endl;

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
    cudastream_sp cudaStream;
    apracucontext_sp cuContext;
    framemetadata_sp h264ImageMetadata;

    boost::shared_ptr<Mp4ReaderSource> mp4Reader;
    boost::shared_ptr<MotionVectorExtractor> motionExtractor;
    boost::shared_ptr<ColorConversion> colorChange1;
    boost::shared_ptr<ColorConversion> colorChange2;
    boost::shared_ptr<Module> cudaCopy;
    boost::shared_ptr<Module> sync;
    boost::shared_ptr<Module> encoder;
    boost::shared_ptr<Module> mp4WriterSink;
};

/**
 * @brief Main entry point
 *
 * Demonstrates timelapse/summary video generation from an input video.
 * Only frames with significant motion are included in the output.
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
        std::cerr << "Usage: " << argv[0] << " <input_video_path> <output_folder_path>" << std::endl;
        std::cerr << "\nExample:" << std::endl;
        std::cerr << "  " << argv[0] << " surveillance.mp4 timelapse_output/" << std::endl;
        std::cerr << "\nArguments:" << std::endl;
        std::cerr << "  input_video_path   - Path to input MP4 video file" << std::endl;
        std::cerr << "  output_folder_path - Folder where timelapse video will be saved" << std::endl;
        std::cerr << "\nNotes:" << std::endl;
        std::cerr << "  - Input must be H264-encoded MP4 video" << std::endl;
        std::cerr << "  - Output video will contain only frames with significant motion" << std::endl;
        std::cerr << "  - Motion threshold is set to 2 (configurable in code)" << std::endl;
        std::cerr << "  - Processing time depends on input video length" << std::endl;
        return 1;
    }

    std::string videoPath = argv[1];
    std::string outFolderPath = argv[2];

    // Create and setup timelapse generator
    TimelapsePipeline timelapsePipeline;

    if (!timelapsePipeline.setupPipeline(videoPath, outFolderPath)) {
        std::cerr << "\nFailed to setup pipeline. Exiting..." << std::endl;
        return 1;
    }

    if (!timelapsePipeline.startPipeline()) {
        std::cerr << "\nFailed to start pipeline. Exiting..." << std::endl;
        return 1;
    }

    // Wait for timelapse generation to complete
    // For long videos, this can take several minutes
    // The pipeline will automatically stop when input video ends
    std::cout << "\nProcessing... (wait time depends on video length)" << std::endl;
    std::cout << "Typical: 1 minute of processing per 10 minutes of input video" << std::endl;

    // Wait for 2 minutes (adjust based on expected video length)
    boost::this_thread::sleep_for(boost::chrono::minutes(2));

    // Stop the pipeline
    if (!timelapsePipeline.stopPipeline()) {
        std::cerr << "\nFailed to stop pipeline cleanly." << std::endl;
        return 1;
    }

    std::cout << "\n╔══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║     Timelapse Video Generated Successfully!                 ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << "\nOutput saved to: " << outFolderPath << std::endl;
    std::cout << "Check the folder for the generated timelapse MP4 file\n" << std::endl;

    return 0;
}
