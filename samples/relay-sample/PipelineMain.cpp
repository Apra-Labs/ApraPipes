#include "relay_sample.h"
#include <iostream>
#include <chrono>
#include <boost/test/unit_test.hpp>

int main(int argc, char *argv[]) {

    if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <RtspCameraURL> <Mp4VideoFilePath>" << std::endl;
    return 1;
  }

  std::string rtspUrl = argv[argc - 2];
  std::string mp4VideoPath = argv[argc - 1];

  LoggerProps loggerProps;
  loggerProps.logLevel = boost::log::trivial::severity_level::info;
  Logger::setLogLevel(boost::log::trivial::severity_level::info);
  Logger::initLogger(loggerProps);

  RelayPipeline pipelineInstance;

  if (!pipelineInstance.setupPipeline(rtspUrl, mp4VideoPath)) {
    std::cerr << "Failed to setup pipeline." << std::endl;
    return 1;
  }

  if (!pipelineInstance.startPipeline()) {
      std::cerr << "Failed to start pipeline." << std::endl;
      return 1;
  }

  while (true) {
    int k = getchar();
    if (k == 114) {
      pipelineInstance.addRelayToRtsp(false);
      pipelineInstance.addRelayToMp4(true);
    }
    if (k == 108) {
      pipelineInstance.addRelayToMp4(false);
      pipelineInstance.addRelayToRtsp(true);
    }
    if (k == 115) {
      pipelineInstance.stopPipeline();
      break;
    }
  }

  return 0;
}