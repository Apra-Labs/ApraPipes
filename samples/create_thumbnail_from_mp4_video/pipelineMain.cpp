#include "GenerateThumbnailsPipeline.h"
#include <chrono>
#include <iostream>

void main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <videoPath> <outFolderPath>"
              << std::endl;
  }

  std::string videoPath = argv[argc - 2];
  std::string outFolderPath = argv[argc - 1];

  GenerateThumbnailsPipeline thumbnailPipeline;
  if (!thumbnailPipeline.setUpPipeLine(videoPath, outFolderPath)) {
    std::cerr << "Failed to setup pipeline." << std::endl;
  }

  if (!thumbnailPipeline.startPipeLine()) {
    std::cerr << "Failed to start pipeline." << std::endl;
  }

  // Wait for the pipeline to run for 10 seconds
  boost::this_thread::sleep_for(boost::chrono::seconds(5));

  // Stop the pipeline
  if (!thumbnailPipeline.stopPipeLine()) {
    std::cerr << "Failed to stop pipeline." << std::endl;
  } else {
    std::cerr << "Saved Generated Thumbnail in <" << outFolderPath << ">"; 
  }
}