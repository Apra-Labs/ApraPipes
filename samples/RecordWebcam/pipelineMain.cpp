#include "RecordWebcam.h"
#include <chrono>
#include <iostream>
#include <boost/thread/thread.hpp>
 

int main(int argc, char* argv[]) {
    std::cout << "[Main] Program started." << std::endl;
    std::cout << "[Main] Creating RecordWebcam instance." << std::endl;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <videoPath>" << std::endl;
        return -1;
    }

    std::string videoPath = "";
    const int camId = 0;
    RecordWebcam pipelineInstance;

    std::cout << "[Main] Calling setUpPipeLine..." << std::endl;
    if (!pipelineInstance.setUpPipeLine(camId, videoPath)) {
        std::cerr << "Failed to setup pipeline." << std::endl;
        return -1;
    }
    std::cout << "[Main] Finished setUpPipeLine." << std::endl;

    std::cout << "[Main] Calling startPipeline..." << std::endl;
    if (!pipelineInstance.startPipeline()) {
        std::cerr << "Failed to start pipeline." << std::endl;
        return -1;
    }
    std::cout << "[Main] Finished startPipeline. Pipeline is running." << std::endl;
    std::cout << "[Main] Recording for 10 seconds..." << std::endl;

    //  Let the pipeline run for 10 seconds
    boost::this_thread::sleep_for(boost::chrono::seconds(30));
    std::cout << "[Main] Auto-stopping after 10 seconds..." << std::endl;

    if (!pipelineInstance.stopPipeline()) {
        std::cerr << "Failed to stop pipeline." << std::endl;
        return -1;
    }

    return 0;
}