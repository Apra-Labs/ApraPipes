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
    bool setupPipeline();
    bool startPipeline();
    bool stopPipeline();
    bool testPipeline();
private:
    PipeLine pipeline;
    boost::shared_ptr<WebCamSource> source;
    boost::shared_ptr<FaceDetectorXform> faceDetector;
    boost::shared_ptr<OverlayModule> overlay;
    boost::shared_ptr<ColorConversion> colorConversion;
    boost::shared_ptr<ImageViewerModule> imageViewerSink;
};