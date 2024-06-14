#include "WebCamSource.h"
#include "ImageViewerModule.h"
#include "face_detection_cpu.h"
#include "OverlayModule.h"
#include "FacialLandmarksCV.h"
#include "ImageDecoderCV.h"
#include "FaceDetectorXform.h"
#include "ColorConversionXForm.h"
#include <boost/test/unit_test.hpp>

FaceDetectionCPU::FaceDetectionCPU() : faceDetectionCPUSamplePipeline("faceDetectionCPUSamplePipeline") {}

bool FaceDetectionCPU::setupPipeline() {
    WebCamSourceProps webCamSourceprops(0, 640, 480);
    mSource = boost::shared_ptr<WebCamSource>(new WebCamSource(webCamSourceprops));

    FaceDetectorXformProps faceDetectorProps(1.0, 0.8, "./data/assets/deploy.prototxt", "./data/assets/res10_300x300_ssd_iter_140000_fp16.caffemodel");
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