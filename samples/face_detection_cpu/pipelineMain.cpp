#include "face_detection_cpu.h"
#include <chrono>
#include <iostream>

void main() {
  FaceDetectionCPU faceDetectionCPUSamplePipeline;
  if (!faceDetectionCPUSamplePipeline.setupPipeline()) {
    std::cerr << "Failed to setup pipeline." << std::endl;
  }

  if (!faceDetectionCPUSamplePipeline.startPipeline()) {
    std::cerr << "Failed to start pipeline." << std::endl;
  }

  // Wait for the pipeline to run for 10 seconds
  boost::this_thread::sleep_for(boost::chrono::seconds(50));

  // Stop the pipeline
  if (!faceDetectionCPUSamplePipeline.stopPipeline()) {
    std::cerr << "Failed to stop pipeline." << std::endl;
  }
}