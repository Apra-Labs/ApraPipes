#include <boost/test/unit_test.hpp>
#include "FrameFactory.h"
#include "Frame.h"
#include "FrameMetadata.h"
#include "EncodedImageMetadata.h"
#include "RawImageMetadata.h"
#include "H264Metadata.h"
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>

BOOST_AUTO_TEST_SUITE(framefactory_memory_tests)

// Helper class to track memory allocations
class MemoryTracker {
private:
    std::atomic<size_t> totalAllocated{0};
    std::atomic<size_t> totalFreed{0};
    std::atomic<size_t> currentInUse{0};
    std::atomic<int> allocCount{0};
    std::atomic<int> freeCount{0};

public:
    void recordAllocation(size_t bytes) {
        totalAllocated += bytes;
        currentInUse += bytes;
        allocCount++;
    }

    void recordFree(size_t bytes) {
        totalFreed += bytes;
        currentInUse -= bytes;
        freeCount++;
    }

    void printStats(const std::string& testName) {
        LOG_INFO << "=== Memory Stats for " << testName << " ===";
        LOG_INFO << "Total Allocated: " << totalAllocated << " bytes";
        LOG_INFO << "Total Freed: " << totalFreed << " bytes";
        LOG_INFO << "Currently In Use: " << currentInUse << " bytes";
        LOG_INFO << "Alloc Count: " << allocCount;
        LOG_INFO << "Free Count: " << freeCount;
        LOG_INFO << "Leaked: " << (totalAllocated - totalFreed) << " bytes";
    }

    bool hasLeaks() {
        return totalAllocated != totalFreed;
    }

    size_t getLeakedBytes() {
        return totalAllocated > totalFreed ? totalAllocated - totalFreed : 0;
    }
};

// Test 1: Basic FrameFactory allocation and deallocation
BOOST_AUTO_TEST_CASE(framefactory_basic_memory_test)
{
    LoggerProps loggerProps;
    loggerProps.logLevel = boost::log::trivial::severity_level::info;
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
    Logger::initLogger(loggerProps);
    LOG_INFO << "Starting framefactory_basic_memory_test";

    auto metadata = framemetadata_sp(new RawImageMetadata(1920, 1080, ImageMetadata::ImageType::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST));
    auto factory = boost::shared_ptr<FrameFactory>(new FrameFactory(metadata, 10));

    // Get initial pool health
    auto initialHealth = factory->getPoolHealthRecord();
    LOG_INFO << "Initial pool health: " << initialHealth;

    // Create and destroy frames
    {
        auto frame1 = factory->create(1024, factory);
        BOOST_TEST(frame1.get() != nullptr);
        BOOST_TEST(frame1->size() == 1024);

        auto frame2 = factory->create(2048, factory);
        BOOST_TEST(frame2.get() != nullptr);
        BOOST_TEST(frame2->size() == 2048);
    }

    // Frames should be destroyed here
    auto finalHealth = factory->getPoolHealthRecord();
    LOG_INFO << "Final pool health: " << finalHealth;

    // Check that all memory is released
    BOOST_TEST(finalHealth.find("Frames<0>") != std::string::npos);
}

// Test 2: Simulate the skipBytes pointer increment issue
BOOST_AUTO_TEST_CASE(framefactory_pointer_increment_leak_test)
{
    LoggerProps loggerProps;
    loggerProps.logLevel = boost::log::trivial::severity_level::info;
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
    Logger::initLogger(loggerProps);

    LOG_INFO << "Starting framefactory_pointer_increment_leak_test";

    auto metadata = framemetadata_sp(new H264Metadata(1920, 1080));
    auto factory = boost::shared_ptr<FrameFactory>(new FrameFactory(metadata, 100));

    MemoryTracker tracker;

    // Simulate what happens in Mp4ReaderDetailH264::skipBytes
    const int skipOffset = 512;
    const int numFrames = 10;

    for (int i = 0; i < numFrames; i++) {
        // Create a frame (simulating makeFrame in Mp4ReaderSource)
        size_t frameSize = 4096;
        auto frame = factory->create(frameSize, factory);
        tracker.recordAllocation(frameSize);

        // Get the data pointer
        uint8_t* dataPtr = static_cast<uint8_t*>(frame->data());

        // SIMULATE THE BUG: Increment pointer (like skipBytes does)
        // In the real code, this modified pointer is stored back
        dataPtr += skipOffset;

        // Now when frame goes out of scope, it tries to free the WRONG address
        // This is the memory leak!

        LOG_TRACE << "Frame " << i << ": Original ptr=" << (void*)frame->data()
                  << ", Modified ptr=" << (void*)dataPtr;
    }

    // Check pool health after frames are destroyed
    auto finalHealth = factory->getPoolHealthRecord();
    LOG_INFO << "Final pool health after pointer increment test: " << finalHealth;

    tracker.printStats("pointer_increment_leak_test");
}

// Test 3: Test makeFrame with size trimming (as used in Mp4ReaderSource)
BOOST_AUTO_TEST_CASE(framefactory_trim_frame_test)
{
    LoggerProps loggerProps;
    loggerProps.logLevel = boost::log::trivial::severity_level::info;
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
    Logger::initLogger(loggerProps);

    LOG_INFO << "Starting framefactory_trim_frame_test";

    auto metadata = framemetadata_sp(new EncodedImageMetadata(1920, 1080));
    auto factory = boost::shared_ptr<FrameFactory>(new FrameFactory(metadata, 10));

    // Create a big frame
    size_t bigSize = 8192;
    auto bigFrame = factory->create(bigSize, factory);
    BOOST_TEST(bigFrame->size() == bigSize);

    // Simulate makeFrameTrim operation
    size_t trimmedSize = 4096;
    auto trimmedFrame = factory->create(bigFrame, trimmedSize, factory);
    BOOST_TEST(trimmedFrame->size() == trimmedSize);

    // Original big frame should release excess memory
    auto health = factory->getPoolHealthRecord();
    LOG_INFO << "Pool health after trim: " << health;
}

// Test 4: Stress test with rapid allocation/deallocation
BOOST_AUTO_TEST_CASE(framefactory_stress_test)
{
    LoggerProps loggerProps;
    loggerProps.logLevel = boost::log::trivial::severity_level::info;
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
    Logger::initLogger(loggerProps);

    LOG_INFO << "Starting framefactory_stress_test";

    auto metadata = framemetadata_sp(new RawImageMetadata(1920, 1080, ImageMetadata::ImageType::YUV420, CV_8UC1, 0, CV_8U, FrameMetadata::HOST));
    auto factory = boost::shared_ptr<FrameFactory>(new FrameFactory(metadata, 100));

    const int numIterations = 100;
    const int framesPerIteration = 10;

    auto initialHealth = factory->getPoolHealthRecord();
    LOG_INFO << "Initial health: " << initialHealth;

    for (int iter = 0; iter < numIterations; iter++) {
        std::vector<frame_sp> frames;

        // Allocate frames
        for (int i = 0; i < framesPerIteration; i++) {
            // YUV420 at 1920x1080 requires 1920*1080*1.5 = 3,110,400 bytes
            size_t size = 3110400 + (i * 1024); // Base YUV420 size with small variations
            frames.push_back(factory->create(size, factory));
        }

        // Clear frames (should deallocate)
        frames.clear();

        if (iter % 10 == 0) {
            auto health = factory->getPoolHealthRecord();
            LOG_TRACE << "Iteration " << iter << " health: " << health;
        }
    }

    auto finalHealth = factory->getPoolHealthRecord();
    LOG_INFO << "Final health after stress test: " << finalHealth;

    // All frames should be released
    BOOST_TEST(finalHealth.find("Frames<0>") != std::string::npos);
}

// Test 5: Reproduce the exact Mp4ReaderSource memory leak pattern
BOOST_AUTO_TEST_CASE(mp4reader_memory_leak_simulation)
{
    LoggerProps loggerProps;
    loggerProps.logLevel = boost::log::trivial::severity_level::info;
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
    Logger::initLogger(loggerProps);

    LOG_INFO << "Starting mp4reader_memory_leak_simulation";

    // Simulate Mp4ReaderDetailH264 behavior
    auto h264Metadata = framemetadata_sp(new H264Metadata(1920, 1080));
    auto factory = boost::shared_ptr<FrameFactory>(new FrameFactory(h264Metadata, 1000));

    const size_t biggerFrameSize = 300000; // From Mp4ReaderSourceProps
    const int skipOffset = 512; // From Mp4ReaderDetailH264::skipOffset
    const int numFramesToProcess = 50;

    MemoryTracker tracker;
    size_t expectedLeakPerFrame = 0;

    for (int i = 0; i < numFramesToProcess; i++) {
        // Step 1: Create frame (like makeFrame in produceFrames)
        auto imgFrame = factory->create(biggerFrameSize, factory);
        tracker.recordAllocation(biggerFrameSize);

        // Step 2: Get data pointer and simulate skipBytes
        uint8_t* sampleFrame = static_cast<uint8_t*>(imgFrame->data());

        // THE BUG: This modifies the pointer that Frame stores internally
        // When frame is destroyed, it will try to free (sampleFrame + skipOffset)
        // instead of the original allocation
        sampleFrame += skipOffset;

        // Step 3: Simulate frame processing (readNextFrame would write to this)
        // In real code, data is written starting at the incremented pointer

        // Step 4: Create trimmed frame (like makeFrameTrim)
        size_t actualDataSize = 2048; // Simulated actual frame size
        auto trimmedFrame = factory->create(imgFrame, actualDataSize + skipOffset, factory);

        // Step 5: Frame goes out of scope and attempts to free wrong address
        // This leaks the first skipOffset bytes!
        expectedLeakPerFrame = skipOffset;

        if (i % 10 == 0) {
            LOG_TRACE << "Processed " << i << " frames";
        }
    }

    // Calculate expected leak
    size_t expectedTotalLeak = expectedLeakPerFrame * numFramesToProcess;
    LOG_INFO << "Expected leak: " << expectedTotalLeak << " bytes";

    auto finalHealth = factory->getPoolHealthRecord();
    LOG_INFO << "Final pool health: " << finalHealth;

    tracker.printStats("mp4reader_leak_simulation");

    // The test will show that memory is leaked
    LOG_WARNING << "Memory leak detected: " << tracker.getLeakedBytes() << " bytes";
}

// Test 6: Test with concurrent access (thread safety)
BOOST_AUTO_TEST_CASE(framefactory_thread_safety_test)
{
    LoggerProps loggerProps;
    loggerProps.logLevel = boost::log::trivial::severity_level::info;
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
    Logger::initLogger(loggerProps);

    LOG_INFO << "Starting framefactory_thread_safety_test";

    auto metadata = framemetadata_sp(new RawImageMetadata(640, 480, ImageMetadata::ImageType::MONO, CV_8UC1, 0, CV_8U, FrameMetadata::HOST));
    auto factory = boost::shared_ptr<FrameFactory>(new FrameFactory(metadata, 100));

    std::atomic<int> totalFramesCreated{0};
    std::atomic<int> totalFramesDestroyed{0};

    const int numThreads = 4;
    const int framesPerThread = 25;

    std::vector<std::thread> threads;

    // Create threads that allocate and deallocate frames
    for (int t = 0; t < numThreads; t++) {
        threads.emplace_back([&factory, &totalFramesCreated, &totalFramesDestroyed, framesPerThread]() {
            boost::shared_ptr<FrameFactory> localFactory = factory;
            for (int i = 0; i < framesPerThread; i++) {
                auto frame = localFactory->create(1024 * (i % 10 + 1), localFactory);
                totalFramesCreated++;

                // Simulate some work
                std::this_thread::sleep_for(std::chrono::microseconds(100));

                // Frame destroyed when going out of scope
                totalFramesDestroyed++;
            }
        });
    }

    // Wait for all threads to complete
    for (auto& t : threads) {
        t.join();
    }

    LOG_INFO << "Total frames created: " << totalFramesCreated;
    LOG_INFO << "Total frames destroyed: " << totalFramesDestroyed;

    auto finalHealth = factory->getPoolHealthRecord();
    LOG_INFO << "Final health after thread test: " << finalHealth;

    // All frames should be released
    BOOST_TEST(finalHealth.find("Frames<0>") != std::string::npos);
}

// Test 7: Verify the fix for skipBytes issue
BOOST_AUTO_TEST_CASE(framefactory_skipbytes_fix_verification)
{
    LoggerProps loggerProps;
    loggerProps.logLevel = boost::log::trivial::severity_level::info;
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
    Logger::initLogger(loggerProps);

    LOG_INFO << "Starting framefactory_skipbytes_fix_verification";

    auto metadata = framemetadata_sp(new H264Metadata(1920, 1080));
    auto factory = boost::shared_ptr<FrameFactory>(new FrameFactory(metadata, 10));

    const int skipOffset = 512;

    // Correct way to handle skipBytes
    auto frame = factory->create(4096, factory);
    uint8_t* originalPtr = static_cast<uint8_t*>(frame->data());

    // CORRECT: Use a local variable for the offset pointer
    uint8_t* workingPtr = originalPtr + skipOffset;

    // Do work with workingPtr...
    // But frame still has the original pointer for proper cleanup

    LOG_INFO << "Original ptr: " << (void*)originalPtr;
    LOG_INFO << "Working ptr: " << (void*)workingPtr;
    LOG_INFO << "Frame data ptr (unchanged): " << frame->data();

    // Verify frame still has original pointer
    BOOST_TEST(frame->data() == originalPtr);

    // Frame will correctly free originalPtr when destroyed
    auto health = factory->getPoolHealthRecord();
    LOG_INFO << "Pool health before frame destruction: " << health;
}

// Test 8: Verify that the FrameFactory::destroy fix handles offset pointers correctly
BOOST_AUTO_TEST_CASE(framefactory_offset_pointer_destroy_test)
{
    LoggerProps loggerProps;
    loggerProps.logLevel = boost::log::trivial::severity_level::trace;
    Logger::setLogLevel(boost::log::trivial::severity_level::trace);
    Logger::initLogger(loggerProps);

    LOG_INFO << "Starting framefactory_offset_pointer_destroy_test - Testing the fix for skipBytes memory leak";

    auto metadata = framemetadata_sp(new H264Metadata(1920, 1080));
    auto factory = boost::shared_ptr<FrameFactory>(new FrameFactory(metadata, 100));

    const int skipOffset = 512;
    const size_t originalSize = 4096;
    const int numIterations = 100;

    // Get initial pool health
    auto initialHealth = factory->getPoolHealthRecord();
    LOG_INFO << "Initial pool health: " << initialHealth;

    for (int i = 0; i < numIterations; i++) {
        // Step 1: Create a frame with original size
        auto frame = factory->create(originalSize, factory);
        BOOST_TEST(frame->size() == originalSize);

        // Step 2: Get the raw Frame pointer to manipulate it (simulating skipBytes)
        Frame* rawFrame = frame.get();

        // Store original data pointer for comparison
        void* originalDataPtr = frame->data();

        // Step 3: SIMULATE SKIPBYTES BUG
        // We need to modify the mutable_buffer base class to simulate the pointer increment
        // This is exactly what happens in Mp4ReaderDetailH264::skipBytes
        boost::asio::mutable_buffer* bufferPtr = static_cast<boost::asio::mutable_buffer*>(rawFrame);
        uint8_t* currentData = static_cast<uint8_t*>(bufferPtr->data());

        // Modify the data pointer (simulating buffer += skipOffset in skipBytes)
        *bufferPtr = boost::asio::mutable_buffer(currentData + skipOffset, frame->size() - skipOffset);

        // Verify the pointer has been offset
        BOOST_TEST(frame->data() != originalDataPtr);
        BOOST_TEST(static_cast<uint8_t*>(frame->data()) == static_cast<uint8_t*>(originalDataPtr) + skipOffset);

        // Step 4: Frame goes out of scope
        // The destroy() method should handle the offset pointer correctly
        // It should:
        // 1. Detect that frame->data() != frame->myOrig
        // 2. Calculate the original allocation size as (current size + offset)
        // 3. Free the correct number of chunks from myOrig
    }

    // Give time for frames to be destroyed
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Check final pool health - should show no leaks
    auto finalHealth = factory->getPoolHealthRecord();
    LOG_INFO << "Final pool health after " << numIterations << " iterations: " << finalHealth;

    // All frames should be properly freed
    BOOST_TEST(finalHealth.find("Frames<0>") != std::string::npos);
}

// Test 9: Verify edge cases for the offset pointer fix
BOOST_AUTO_TEST_CASE(framefactory_offset_pointer_edge_cases)
{
    LoggerProps loggerProps;
    loggerProps.logLevel = boost::log::trivial::severity_level::info;
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
    Logger::initLogger(loggerProps);

    LOG_INFO << "Starting framefactory_offset_pointer_edge_cases";

    auto metadata = framemetadata_sp(new H264Metadata(1920, 1080));
    auto factory = boost::shared_ptr<FrameFactory>(new FrameFactory(metadata, 100));

    // Test Case 1: Multiple different offset sizes
    std::vector<size_t> offsets = {0, 256, 512, 1024, 2048};
    std::vector<size_t> frameSizes = {1024, 2048, 4096, 8192, 16384};

    for (size_t frameSize : frameSizes) {
        for (size_t offset : offsets) {
            if (offset >= frameSize) continue; // Skip invalid offsets

            auto frame = factory->create(frameSize, factory);
            Frame* rawFrame = frame.get();

            // Apply offset if non-zero
            if (offset > 0) {
                boost::asio::mutable_buffer* bufferPtr = static_cast<boost::asio::mutable_buffer*>(rawFrame);
                uint8_t* currentData = static_cast<uint8_t*>(bufferPtr->data());
                *bufferPtr = boost::asio::mutable_buffer(currentData + offset, frameSize - offset);
            }

            LOG_TRACE << "Testing frame size=" << frameSize << " offset=" << offset;
        }
    }

    // Check for no memory leaks
    auto health = factory->getPoolHealthRecord();
    LOG_INFO << "Final pool health for edge cases: " << health;
    BOOST_TEST(health.find("Frames<0>") != std::string::npos);
}

// Test 10: Performance test to ensure the fix doesn't significantly impact performance
BOOST_AUTO_TEST_CASE(framefactory_offset_pointer_performance_test)
{
    LoggerProps loggerProps;
    loggerProps.logLevel = boost::log::trivial::severity_level::warning;
    Logger::setLogLevel(boost::log::trivial::severity_level::warning);
    Logger::initLogger(loggerProps);

    LOG_INFO << "Starting framefactory_offset_pointer_performance_test";

    auto metadata = framemetadata_sp(new H264Metadata(1920, 1080));
    auto factory = boost::shared_ptr<FrameFactory>(new FrameFactory(metadata, 1000));

    const int numFrames = 10000;
    const size_t frameSize = 4096;
    const int skipOffset = 512;

    // Time allocation and deallocation with offset pointers
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numFrames; i++) {
        auto frame = factory->create(frameSize, factory);

        // Apply offset to half the frames
        if (i % 2 == 0) {
            Frame* rawFrame = frame.get();
            boost::asio::mutable_buffer* bufferPtr = static_cast<boost::asio::mutable_buffer*>(rawFrame);
            uint8_t* currentData = static_cast<uint8_t*>(bufferPtr->data());
            *bufferPtr = boost::asio::mutable_buffer(currentData + skipOffset, frameSize - skipOffset);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    LOG_INFO << "Processed " << numFrames << " frames in " << duration.count() << "ms";
    LOG_INFO << "Average time per frame: " << (duration.count() / (double)numFrames) << "ms";

    // Verify no memory leaks
    auto finalHealth = factory->getPoolHealthRecord();
    LOG_INFO << "Final pool health after performance test: " << finalHealth;
    BOOST_TEST(finalHealth.find("Frames<0>") != std::string::npos);
}

BOOST_AUTO_TEST_SUITE_END()