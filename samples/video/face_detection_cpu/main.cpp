/**
 * @file face_detection_cpu.cpp
 * @brief CPU-based face detection sample using webcam input
 *
 * This sample demonstrates:
 * - Capturing video from a webcam
 * - Performing CPU-based face detection using Caffe DNN model
 * - Drawing bounding boxes around detected faces
 * - Displaying the result in a window
 *
 * Pipeline Structure:
 *   [WebCam] → [FaceDetector] → [Overlay] → [ColorConversion] → [ImageViewer]
 *
 * Requirements:
 * - Webcam connected to the system
 * - Caffe model files in ./data/assets/ directory:
 *   - deploy.prototxt (model architecture)
 *   - res10_300x300_ssd_iter_140000_fp16.caffemodel (trained weights)
 *
 * Usage:
 *   face_detection_cpu.exe
 *
 * The sample will run for 50 seconds and then automatically stop.
 * Press Ctrl+C to stop early (if running in console).
 */

#include <iostream>
#include <chrono>
#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>

// ApraPipes modules
#include "PipeLine.h"
#include "WebCamSource.h"
#include "ImageViewerModule.h"
#include "OverlayModule.h"
#include "FaceDetectorXform.h"
#include "ColorConversionXForm.h"

// Cross-platform sleep macro
#ifdef _WIN32
    #include <windows.h>
    #define SLEEP_SECONDS(x) Sleep((x) * 1000)
#else
    #include <unistd.h>
    #define SLEEP_SECONDS(x) sleep(x)
#endif

/**
 * @class FaceDetectionCPU
 * @brief Encapsulates a face detection pipeline using CPU-based processing
 *
 * This class manages the lifecycle of a face detection pipeline that:
 * 1. Captures video frames from a webcam
 * 2. Detects faces using a Caffe-based DNN model
 * 3. Overlays bounding boxes on detected faces
 * 4. Displays the processed frames in a window
 */
class FaceDetectionCPU {
public:
    /**
     * @brief Constructor - initializes the pipeline
     */
    FaceDetectionCPU() : faceDetectionCPUSamplePipeline("faceDetectionCPUSamplePipeline") {}

    /**
     * @brief Sets up the pipeline modules and connections
     *
     * @param cameraId Camera device ID (usually 0 for default webcam)
     * @param scaleFactor Scaling factor for input images (1.0 = no scaling)
     * @param threshold Confidence threshold for face detection (0.0-1.0)
     * @return true if setup successful, false otherwise
     *
     * @note Model paths are hardcoded in FaceDetectorXform implementation:
     *       - Config: ./data/assets/deploy.prototxt
     *       - Weights: ./data/assets/res10_300x300_ssd_iter_140000_fp16.caffemodel
     */
    bool setupPipeline(const int &cameraId,
                      const double &scaleFactor,
                      const double &threshold)
    {
        std::cout << "============================================================\n" << std::endl;
        std::cout << "     ApraPipes Sample: Face Detection CPU                    " << std::endl;
        std::cout << "=============================================================\n" << std::endl;

        std::cout << "Setting up face detection pipeline..." << std::endl;
        std::cout << "  Camera ID: " << cameraId << std::endl;
        std::cout << "  Scale Factor: " << scaleFactor << std::endl;
        std::cout << "  Detection Threshold: " << threshold << std::endl;
        std::cout << "  Model paths (hardcoded in FaceDetectorXform):" << std::endl;
        std::cout << "    Config: ./data/assets/deploy.prototxt" << std::endl;
        std::cout << "    Weights: ./data/assets/res10_300x300_ssd_iter_140000_fp16.caffemodel" << std::endl;

        try {
            // 1. Create webcam source
            WebCamSourceProps webCamSourceprops(cameraId);
            mSource = boost::shared_ptr<WebCamSource>(new WebCamSource(webCamSourceprops));

            // 2. Create face detector with Caffe model
            // Note: Model paths are hardcoded in FaceDetectorXform implementation
            FaceDetectorXformProps faceDetectorProps(scaleFactor, static_cast<float>(threshold));
            mFaceDetector = boost::shared_ptr<FaceDetectorXform>(new FaceDetectorXform(faceDetectorProps));
            mSource->setNext(mFaceDetector);

            // 3. Create overlay module for drawing bounding boxes
            mOverlay = boost::shared_ptr<OverlayModule>(new OverlayModule(OverlayModuleProps()));
            mFaceDetector->setNext(mOverlay);

            // 4. Create color conversion (RGB to BGR for OpenCV display)
            mColorConversion = boost::shared_ptr<ColorConversion>(
                new ColorConversion(ColorConversionProps(ColorConversionProps::RGB_TO_BGR))
            );
            mOverlay->setNext(mColorConversion);

            // 5. Create image viewer sink
            mImageViewerSink = boost::shared_ptr<ImageViewerModule>(
                new ImageViewerModule(ImageViewerModuleProps("Face Detection - CPU"))
            );
            mColorConversion->setNext(mImageViewerSink);

            std::cout << "\nPipeline setup completed successfully!" << std::endl;
            std::cout << "\nPipeline structure:" << std::endl;
            std::cout << "  [WebCam] → [FaceDetector] → [Overlay] → [ColorConversion] → [ImageViewer]" << std::endl;

            return true;

        } catch (const std::exception &e) {
            std::cerr << "\nError during pipeline setup: " << e.what() << std::endl;
            return false;
        } catch (...) {
            std::cerr << "\nUnknown error during pipeline setup" << std::endl;
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

            faceDetectionCPUSamplePipeline.appendModule(mSource);
            faceDetectionCPUSamplePipeline.init();
            faceDetectionCPUSamplePipeline.run_all_threaded();

            std::cout << "Pipeline started successfully!" << std::endl;
            std::cout << "\nProcessing webcam feed with face detection..." << std::endl;
            std::cout << "Press Ctrl+C to stop early, or wait for automatic shutdown.\n" << std::endl;

            return true;

        } catch (const std::exception &e) {
            std::cerr << "Error starting pipeline : " << e.what() << std::endl;
            return false;
        } catch (...) {
            std::cerr << "Unknown error starting pipeline" << std::endl;
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

            faceDetectionCPUSamplePipeline.stop();
            faceDetectionCPUSamplePipeline.term();
            faceDetectionCPUSamplePipeline.wait_for_all();

            std::cout << " Pipeline stopped successfully!" << std::endl;

            return true;

        } catch (const std::exception &e) {
            std::cerr << " Error stopping pipeline: " << e.what() << std::endl;
            return false;
        } catch (...) {
            std::cerr << " Unknown error stopping pipeline" << std::endl;
            return false;
        }
    }

private:
    PipeLine faceDetectionCPUSamplePipeline;
    boost::shared_ptr<WebCamSource> mSource;
    boost::shared_ptr<FaceDetectorXform> mFaceDetector;
    boost::shared_ptr<OverlayModule> mOverlay;
    boost::shared_ptr<ColorConversion> mColorConversion;
    boost::shared_ptr<ImageViewerModule> mImageViewerSink;
};

/**
 * @brief Main entry point
 *
 * Demonstrates face detection on webcam feed using CPU-based processing.
 * The pipeline runs for 50 seconds then automatically stops.
 */
int main() {
    // Pipeline configuration
    const int cameraId = 0;                    // Default webcam
    const double scaleFactor = 1.0;            // No scaling
    const double confidenceThreshold = 0.7;    // 70% confidence minimum
    const int runDurationSeconds = 50;         // Run for 50 seconds

    // Note: Caffe model paths are hardcoded in FaceDetectorXform:
    //   - Config: ./data/assets/deploy.prototxt
    //   - Weights: ./data/assets/res10_300x300_ssd_iter_140000_fp16.caffemodel

    // Create pipeline
    FaceDetectionCPU faceDetectionCPUSamplePipeline;

    // Setup pipeline
    if (!faceDetectionCPUSamplePipeline.setupPipeline(
            cameraId, scaleFactor, confidenceThreshold)) {
        std::cerr << "\nFailed to setup pipeline. Exiting..." << std::endl;
        return 1;  // Exit with error code
    }

    // Start pipeline
    if (!faceDetectionCPUSamplePipeline.startPipeline()) {
        std::cerr << "\nFailed to start pipeline. Exiting..." << std::endl;
        return 1;  // Exit with error code
    }

    // Run for specified duration
    std::cout << "Running for " << runDurationSeconds << " seconds..." << std::endl;
    boost::this_thread::sleep_for(boost::chrono::seconds(runDurationSeconds));

    // Stop pipeline
    if (!faceDetectionCPUSamplePipeline.stopPipeline()) {
        std::cerr << "\nFailed to stop pipeline cleanly." << std::endl;
        return 1;  // Exit with error code
    }

    std::cout << "\n╔══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║     Face Detection Sample Completed Successfully!           ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n" << std::endl;

    return 0;  // Success
}
