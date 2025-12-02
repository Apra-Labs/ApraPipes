/**
 * @file main.cpp
 * @brief Hello Pipeline - A minimal ApraPipes example
 *
 * This sample demonstrates the most basic usage of ApraPipes:
 * - Creating a simple SOURCE -> TRANSFORM -> SINK pipeline
 * - Adding metadata and connecting modules
 * - Processing frames through the pipeline
 * - Proper initialization and cleanup
 *
 * This is the "Hello World" of ApraPipes - the simplest possible working pipeline.
 */

#include <iostream>
#include <boost/shared_ptr.hpp>

// Core ApraPipes headers
#include "PipeLine.h"
#include "ExternalSourceModule.h"
#include "ExternalSinkModule.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "Logger.h"

// Simple transform module for demonstration
#include "Split.h"

// Cross-platform sleep
#ifdef _WIN32
#include <windows.h>
#define SLEEP_MS(x) Sleep(x)
#else
#include <unistd.h>
#define SLEEP_MS(x) usleep((x) * 1000)
#endif

/**
 * Print a formatted banner to the console
 */
void printBanner()
{
    std::cout << "\n";
    std::cout << "=====================================" << std::endl;
    std::cout << "  ApraPipes Hello Pipeline Sample   " << std::endl;
    std::cout << "=====================================" << std::endl;
    std::cout << "\n";
    std::cout << "This sample demonstrates:" << std::endl;
    std::cout << "  1. Creating modules (Source -> Transform -> Sink)" << std::endl;
    std::cout << "  2. Connecting modules in a pipeline" << std::endl;
    std::cout << "  3. Initializing and processing frames" << std::endl;
    std::cout << "  4. Proper cleanup and termination" << std::endl;
    std::cout << "\n";
}

/**
 * Print pipeline status information
 */
void printPipelineInfo(const std::string& stage, bool success)
{
    std::cout << "[" << stage << "] ";
    if (success) {
        std::cout << "✓ Success" << std::endl;
    } else {
        std::cout << "✗ Failed" << std::endl;
    }
}

/**
 * Main function - demonstrates a basic ApraPipes pipeline
 */
int main(int argc, char* argv[])
{
    // Initialize logger with info level
    Logger::setLogLevel(boost::log::trivial::severity_level::info);

    printBanner();

    try {
        std::cout << "=== Step 1: Creating Modules ===" << std::endl;

        // 1. Create Source Module
        std::cout << "Creating ExternalSourceModule..." << std::endl;
        auto source = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());

        // Define frame metadata - what kind of data flows through the pipeline
        auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::GENERAL));
        auto source_pin = source->addOutputPin(metadata);
        printPipelineInfo("Source created", true);

        // 2. Create Transform Module (Split - splits one input into multiple outputs)
        std::cout << "Creating Split module (1 input -> 2 outputs)..." << std::endl;
        SplitProps splitProps;
        splitProps.number = 2;  // Split into 2 streams
        auto split = boost::shared_ptr<Split>(new Split(splitProps));
        printPipelineInfo("Transform created", true);

        // 3. Create Sink Module
        std::cout << "Creating ExternalSinkModule..." << std::endl;
        auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
        printPipelineInfo("Sink created", true);

        std::cout << "\n=== Step 2: Connecting Modules ===" << std::endl;

        // Connect: Source -> Split -> Sink
        source->setNext(split);
        split->setNext(sink);
        std::cout << "Pipeline topology: [Source] -> [Split] -> [Sink]" << std::endl;
        printPipelineInfo("Modules connected", true);

        std::cout << "\n=== Step 3: Initializing Pipeline ===" << std::endl;

        // Initialize all modules
        bool initSuccess = true;

        std::cout << "Initializing source..." << std::endl;
        initSuccess &= source->init();
        printPipelineInfo("Source init", initSuccess);

        std::cout << "Initializing split..." << std::endl;
        initSuccess &= split->init();
        printPipelineInfo("Split init", initSuccess);

        std::cout << "Initializing sink..." << std::endl;
        initSuccess &= sink->init();
        printPipelineInfo("Sink init", initSuccess);

        if (!initSuccess) {
            std::cerr << "\n✗ Pipeline initialization failed!" << std::endl;
            return 1;
        }

        std::cout << "\n=== Step 4: Processing Frames ===" << std::endl;

        // Get split output pin IDs
        auto splitPinIds = split->getAllOutputPinsByType(FrameMetadata::GENERAL);
        std::cout << "Split module has " << splitPinIds.size() << " output pins" << std::endl;

        // Process frames through the pipeline
        const int NUM_FRAMES = 5;
        const size_t FRAME_SIZE = 1024;  // 1KB frames

        std::cout << "\nProcessing " << NUM_FRAMES << " frames (each " << FRAME_SIZE << " bytes)..." << std::endl;

        for (int i = 0; i < NUM_FRAMES; i++) {
            std::cout << "\n  Frame " << (i + 1) << "/" << NUM_FRAMES << ":" << std::endl;

            // Create a frame
            auto frame = source->makeFrame(FRAME_SIZE, source_pin);
            frame_container frames;
            frames.insert(std::make_pair(source_pin, frame));

            // Send frame through source
            std::cout << "    - Sending from source..." << std::endl;
            bool sent = source->send(frames);
            if (!sent) {
                std::cerr << "    ✗ Failed to send frame!" << std::endl;
                continue;
            }

            // Process through split
            std::cout << "    - Processing through split..." << std::endl;
            split->step();

            // Receive at sink
            std::cout << "    - Receiving at sink..." << std::endl;
            auto receivedFrames = sink->pop();

            if (!receivedFrames.empty()) {
                auto receivedPinId = receivedFrames.begin()->first;
                auto receivedFrame = receivedFrames.begin()->second;

                // Determine which output pin this came from
                int outputIndex = -1;
                for (size_t j = 0; j < splitPinIds.size(); j++) {
                    if (splitPinIds[j] == receivedPinId) {
                        outputIndex = j;
                        break;
                    }
                }

                std::cout << "    ✓ Received frame on split output #" << outputIndex
                          << " (size: " << receivedFrame->size() << " bytes)" << std::endl;
            } else {
                std::cout << "    ! No frame received" << std::endl;
            }

            // Small delay for readability
            SLEEP_MS(100);
        }

        std::cout << "\n=== Step 5: Pipeline Termination ===" << std::endl;

        // Terminate all modules in reverse order
        std::cout << "Terminating sink..." << std::endl;
        sink->term();
        printPipelineInfo("Sink terminated", true);

        std::cout << "Terminating split..." << std::endl;
        split->term();
        printPipelineInfo("Split terminated", true);

        std::cout << "Terminating source..." << std::endl;
        source->term();
        printPipelineInfo("Source terminated", true);

        std::cout << "\n";
        std::cout << "=====================================" << std::endl;
        std::cout << "  ✓ Pipeline completed successfully!" << std::endl;
        std::cout << "=====================================" << std::endl;
        std::cout << "\n";

        std::cout << "Next Steps:" << std::endl;
        std::cout << "  - Explore samples/video/ for video processing" << std::endl;
        std::cout << "  - Explore samples/image/ for image transforms" << std::endl;
        std::cout << "  - Explore samples/advanced/ for complex pipelines" << std::endl;
        std::cout << "  - Read samples/README.md for more information" << std::endl;
        std::cout << "\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n";
        std::cerr << "=====================================" << std::endl;
        std::cerr << "  ✗ Pipeline Error!" << std::endl;
        std::cerr << "=====================================" << std::endl;
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << "\n";
        return 1;
    } catch (...) {
        std::cerr << "\n";
        std::cerr << "=====================================" << std::endl;
        std::cerr << "  ✗ Unknown Error!" << std::endl;
        std::cerr << "=====================================" << std::endl;
        std::cerr << "\n";
        return 1;
    }
}
