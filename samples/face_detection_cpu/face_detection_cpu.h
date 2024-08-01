#include <PipeLine.h>
#include "WebCamSource.h"
#include "ImageViewerModule.h"
#include "OverlayModule.h"
#include "FacialLandmarksCV.h"
#include "ImageDecoderCV.h"
#include "FaceDetectorXform.h"
#include "ColorConversionXForm.h"

class FaceDetectionCPU {
public:
    FaceDetectionCPU(); 
    bool setupPipeline(const int &cameraId, const double &scaleFactor, const double &threshold, const std::string &faceDetectionConfiguration, const std::string &faceDetectionWeight);
    bool startPipeline();
    bool stopPipeline();
private:
    PipeLine faceDetectionCPUSamplePipeline;
    boost::shared_ptr<WebCamSource> mSource;
    boost::shared_ptr<FaceDetectorXform> mFaceDetector;
    boost::shared_ptr<OverlayModule> mOverlay;
    boost::shared_ptr<ColorConversion> mColorConversion;
    boost::shared_ptr<ImageViewerModule> mImageViewerSink;
};