#include "WebCamSource.h"
#include "ImageViewerModule.h"
#include "face_detection_cpu.h"
#include "OverlayModule.h"
#include "ImageDecoderCV.h"
#include "FaceDetectorXform.h"
#include "ColorConversionXForm.h"
#include <boost/test/unit_test.hpp>
using namespace std;

FaceDetectionCPU::FaceDetectionCPU() : faceDetectionCPUSamplePipeline("faceDetectionCPUSamplePipeline") {}

bool FaceDetectionCPU::setupPipeline(const int& cameraId, const double& scaleFactor, const double& threshold, const std::string& faceDetectionConfiguration, const std::string& faceDetectionWeight) {
    WebCamSourceProps webCamSourceprops(cameraId);
    mSource = boost::shared_ptr<WebCamSource>(new WebCamSource(webCamSourceprops));

    cout << "Check point 1" << endl;

    //FaceDetectorXformProps faceDetectorProps(scaleFactor, threshold, faceDetectionConfiguration, faceDetectionWeight);
    FaceDetectorXformProps faceDetectorProps(scaleFactor, threshold);
    cout << "Check point 2" << endl;

    mFaceDetector = boost::shared_ptr<FaceDetectorXform>(new FaceDetectorXform(faceDetectorProps));
    mSource->setNext(mFaceDetector);
    cout << "Check point 3" << endl;



    mOverlay = boost::shared_ptr<OverlayModule>(new OverlayModule(OverlayModuleProps()));
    mFaceDetector->setNext(mOverlay);
    cout << "Check point 4" << endl;

    mColorConversion = boost::shared_ptr<ColorConversion>(new ColorConversion(ColorConversionProps(ColorConversionProps::RGB_TO_BGR)));
    mOverlay->setNext(mColorConversion);
    cout << "Check point 5" << endl;

    mImageViewerSink = boost::shared_ptr<ImageViewerModule>(new ImageViewerModule(ImageViewerModuleProps("imageview")));
    mColorConversion->setNext(mImageViewerSink);
    cout << "Check point 6" << endl;
    return true;
}


//bool FaceDetectionCPU::setupPipeline(const int& cameraId, const double& scaleFactor, const double& threshold, const std::string& faceDetectionConfiguration, const std::string& faceDetectionWeight)
//{
//    WebCamSourceProps webCamSourceprops(cameraId);
//    mSource = boost::shared_ptr<WebCamSource>(new WebCamSource(webCamSourceprops));
//    std::cout << "Check point 1" << std::endl;
//
//    // Add ImageDecoderCV to convert to RAW_IMAGE
//    mDecoder = boost::shared_ptr<ImageDecoderCV>(new ImageDecoderCV(ImageDecoderCVProps()));
//    mSource->setNext(mDecoder);
//    std::cout << "Check point 2 (decoder added)" << std::endl;
//
//    // Face detector
//    FaceDetectorXformProps faceDetectorProps(scaleFactor, threshold);
//    mFaceDetector = boost::shared_ptr<FaceDetectorXform>(new FaceDetectorXform(faceDetectorProps));
//    mDecoder->setNext(mFaceDetector);
//    std::cout << "Check point 3" << std::endl;
//
//    // Overlay
//    mOverlay = boost::shared_ptr<OverlayModule>(new OverlayModule(OverlayModuleProps()));
//    mFaceDetector->setNext(mOverlay);
//    std::cout << "Check point 4" << std::endl;
//
//    // Color Conversion
//    mColorConversion = boost::shared_ptr<ColorConversion>(new ColorConversion(ColorConversionProps(ColorConversionProps::RGB_TO_BGR)));
//    mOverlay->setNext(mColorConversion);
//    std::cout << "Check point 5" << std::endl;
//
//    // Image Viewer
//    mImageViewerSink = boost::shared_ptr<ImageViewerModule>(new ImageViewerModule(ImageViewerModuleProps("imageview")));
//    mColorConversion->setNext(mImageViewerSink);
//    std::cout << "Check point 6" << std::endl;
//
//    return true;
//}


//bool FaceDetectionCPU::setupPipeline(const int& cameraId, const double& scaleFactor, const double& threshold, const std::string& faceDetectionConfiguration, const std::string& faceDetectionWeight) {
//    WebCamSourceProps webCamSourceprops(cameraId);
//    mSource = boost::shared_ptr<WebCamSource>(new WebCamSource(webCamSourceprops));
//    std::cout << "Check point 1" << std::endl;
//
//    // Decode ENCODED_IMAGE to RAW_IMAGE
//    mDecoder = boost::shared_ptr<ImageDecoderCV>(new ImageDecoderCV());
//    mSource->setNext(mDecoder);
//    std::cout << "Check point 2" << std::endl;
//
//    // Setup face detector
//    FaceDetectorXformProps faceDetectorProps(scaleFactor, threshold);
//    mFaceDetector = boost::shared_ptr<FaceDetectorXform>(new FaceDetectorXform(faceDetectorProps));
//    mDecoder->setNext(mFaceDetector);
//    std::cout << "Check point 3" << std::endl;
//
//    // Overlay module expects RAW_IMAGE input
//    mOverlay = boost::shared_ptr<OverlayModule>(new OverlayModule(OverlayModuleProps()));
//    mFaceDetector->setNext(mOverlay);
//    std::cout << "Check point 4" << std::endl;
//
//    // Convert RGB to BGR for OpenCV ImageViewer
//    mColorConversion = boost::shared_ptr<ColorConversion>(new ColorConversion(ColorConversionProps(ColorConversionProps::RGB_TO_BGR)));
//    mOverlay->setNext(mColorConversion);
//    std::cout << "Check point 5" << std::endl;
//
//    // Image Viewer
//    mImageViewerSink = boost::shared_ptr<ImageViewerModule>(new ImageViewerModule(ImageViewerModuleProps("imageview")));
//    mColorConversion->setNext(mImageViewerSink);
//    std::cout << "Check point 6" << std::endl;
//
//    return true;
//}






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