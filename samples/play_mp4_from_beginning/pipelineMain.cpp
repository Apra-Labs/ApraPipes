#include "PlayMp4VideoFromBeginning.h"
#include <chrono>
#include <iostream>

void main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <videoPath> <outFolderPath>"
              << std::endl;
  }
  std::string videoPath = argv[argc - 1];

  PlayMp4VideoFromBeginning pipelineInstance;

  if (!pipelineInstance.setUpPipeLine(videoPath)) {
    std::cerr << "Failed to setup pipeline." << std::endl;
  }
  if (!pipelineInstance.startPipeLine()) {
    std::cerr << "Failed to start pipeline." << std::endl;
  }
  // Wait for the pipeline to run for 10 seconds
  boost::this_thread::sleep_for(boost::chrono::seconds(3));
  if (!pipelineInstance.flushQueuesAndSeek()) {
    std::cerr << "Failed to flush Queues." << std::endl;
  }
  boost::this_thread::sleep_for(boost::chrono::seconds(5));
  // Stop the pipeline
  if (!pipelineInstance.stopPipeLine()) {
    std::cerr << "Failed to stop pipeline." << std::endl;
  }
}