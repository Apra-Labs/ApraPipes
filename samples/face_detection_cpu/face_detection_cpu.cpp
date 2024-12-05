#include "WebCamSource.h"
#include "ImageViewerModule.h"
#include "face_detection_cpu.h"
#include "OverlayModule.h"
#include "ImageDecoderCV.h"
#include "FaceDetectorXform.h"
#include "ColorConversionXForm.h"
#include <boost/test/unit_test.hpp>

FaceDetectionCPU::FaceDetectionCPU() : faceDetectionCPUSamplePipeline("faceDetectionCPUSamplePipeline") {}

bool FaceDetectionCPU::setupPipeline(const int &cameraId, const double &scaleFactor, const double &threshold, const std::string &faceDetectionConfiguration, const std::string &faceDetectionWeight) {
    WebCamSourceProps webCamSourceprops(cameraId);
    mSource = boost::shared_ptr<WebCamSource>(new WebCamSource(webCamSourceprops));

    FaceDetectorXformProps faceDetectorProps(scaleFactor, threshold, faceDetectionConfiguration, faceDetectionWeight);
    mFaceDetector = boost::shared_ptr<FaceDetectorXform>(new FaceDetectorXform(faceDetectorProps));
    mSource->setNext(mFaceDetector);

    mOverlay = boost::shared_ptr<OverlayModule>(new OverlayModule(OverlayModuleProps()));
    mFaceDetector->setNext(mOverlay);

    mColorConversion = boost::shared_ptr<ColorConversion>(new ColorConversion(ColorConversionProps(ColorConversionProps::RGB_TO_BGR)));
    mOverlay->setNext(mColorConversion);

    mImageViewerSink = boost::shared_ptr<ImageViewerModule>(new ImageViewerModule(ImageViewerModuleProps("imageview")));
    mColorConversion->setNext(mImageViewerSink);
  
    return true;
}

bool FaceDetectionCPU::startPipeline() {
    faceDetectionCPUSamplePipeline.appendModule(mSource);
    faceDetectionCPUSamplePipeline.init();
    faceDetectionCPUSamplePipeline.run_all_threaded();
    return true;
}

bool FaceDetectionCPU::stopPipeline() {
    faceDetectionCPUSamplePipeline.stop();
    faceDetectionCPUSamplePipeline.term();
    faceDetectionCPUSamplePipeline.wait_for_all();
    return true;
}