#include "WebCamSource.h"
#include "ImageViewerModule.h"
#include "face_detection_cpu.h"
#include "OverlayModule.h"
#include "FacialLandmarksCV.h"
#include "ImageDecoderCV.h"
#include "FaceDetectorXform.h"
#include "ColorConversionXForm.h"



FaceDetectionCPU::FaceDetectionCPU() : pipeline("test") {}

bool FaceDetectionCPU::setupPipeline() {
    WebCamSourceProps webCamSourceprops(0, 640, 480);
    auto source = boost::shared_ptr<Module>(new WebCamSource(webCamSourceprops));

    FaceDetectorXformProps faceDetectorProps(1.0, 0.8, "../../data/assets/deploy.prototxt", "../../data/assets/res10_300x300_ssd_iter_140000_fp16.caffemodel");
    auto faceDetector = boost::shared_ptr<FaceDetectorXform>(new FaceDetectorXform(faceDetectorProps));
    source->setNext(faceDetector);

    auto overlay = boost::shared_ptr<Module>(new OverlayModule(OverlayModuleProps()));
    faceDetector->setNext(overlay);

    auto colorConversion = boost::shared_ptr<ColorConversion>(new ColorConversion(ColorConversionProps(ColorConversionProps::RGB_TO_BGR)));
    overlay->setNext(colorConversion);

    auto sink = boost::shared_ptr<Module>(new ImageViewerModule(ImageViewerModuleProps("imageview")));
    colorConversion->setNext(sink);
    
    pipeline.appendModule(source);
    pipeline.init();
    return true;
}

bool FaceDetectionCPU::startPipeline() {
    pipeline.run_all_threaded();
    return true;
}

bool FaceDetectionCPU::stopPipeline() {
    pipeline.stop();
    pipeline.term();
    pipeline.wait_for_all();
    return true;
}

int main() {
  LoggerProps loggerProps;
  loggerProps.logLevel = boost::log::trivial::severity_level::info;
  Logger::setLogLevel(boost::log::trivial::severity_level::info);
  Logger::initLogger(loggerProps);

  FaceDetectionCPU pipelineInstance;

  // Setup the pipeline
  if (!pipelineInstance.setupPipeline()) {
    std::cerr << "Failed to setup pipeline." << std::endl;
    return 1; // Or any error code indicating failure
  }

  // Start the pipeline
  if (!pipelineInstance.startPipeline()) {
    std::cerr << "Failed to start pipeline." << std::endl;
    return 1; // Or any error code indicating failure
  }

  // Wait for the pipeline to run for 10 seconds
  boost::this_thread::sleep_for(boost::chrono::seconds(50));

  // Stop the pipeline
  if (!pipelineInstance.stopPipeline()) {
    std::cerr << "Failed to stop pipeline." << std::endl;
    return 1; // Or any error code indicating failure
  }

  return 0; // Or any success code
}