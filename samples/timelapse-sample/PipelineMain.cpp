#include "timelapse_summary.h"
#include <chrono>
#include <iostream>

void main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <videoPath> <outFolderPath>"
              << std::endl;
  }

  std::string videoPath = argv[argc - 2];
  std::string outFolderPath = argv[argc - 1];

  TimelapsePipeline timelapsePipeline;
  if (!timelapsePipeline.setupPipeline(videoPath, outFolderPath)) {
    std::cerr << "Failed to setup pipeline." << std::endl;
  }

  if (!timelapsePipeline.startPipeline()) {
    std::cerr << "Failed to start pipeline." << std::endl;
  }

  // Wait for the pipeline to run for 10 seconds
  boost::this_thread::sleep_for(boost::chrono::minutes(2));

  // Stop the pipeline
  if (!timelapsePipeline.stopPipeline()) {
    std::cerr << "Failed to stop pipeline." << std::endl;
  }
  std::cerr << "Saved Generated TimeLapse in" << outFolderPath << std::endl;
}