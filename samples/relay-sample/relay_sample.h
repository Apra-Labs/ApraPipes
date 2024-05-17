#include <PipeLine.h>
#include <ColorConversionXForm.h>
#include "KeyboardListener.h"
#include "Mp4VideoMetadata.h"
#include "ImageViewerModule.h"

class ImageViewerModuleExtended : public ImageViewerModule {
public:
    ImageViewerModuleExtended(ImageViewerModuleProps _props) : ImageViewerModule(_props) {}
protected:
    bool validateInputPins() override
    {
        framemetadata_sp metadata = getFirstInputMetadata();
        FrameMetadata::FrameType frameType = metadata->getFrameType();
        FrameMetadata::MemType inputMemType = metadata->getMemType();

#if defined(__arm__) || defined(__aarch64__)
        if (inputMemType != FrameMetadata::MemType::DMABUF) {
            LOG_ERROR << "<" << getId()
                << ">::validateInputPins input memType is expected to be DMABUF. Actual<"
                << inputMemType << ">";
            return false;
        }
        if (frameType != FrameMetadata::RAW_IMAGE &&
            frameType != FrameMetadata::RAW_IMAGE_PLANAR) {
            LOG_ERROR << "<" << getId()
                << ">::validateInputPins input frameType is expected to be "
                "RAW_IMAGE or RAW_IMAGE_PLANAR. Actual<"
                << frameType << ">";
            return false;
        }
#else
        if (frameType != FrameMetadata::RAW_IMAGE) {
            LOG_ERROR << "<" << getId()
                << ">::validateInputPins input frameType is expected to be RAW_IMAGE. Actual<"
                << frameType << ">";
            return false;
        }
#endif
        return true;
    }
};

class RelayPipeline {
public:
	boost::shared_ptr<ColorConversion> rtspColorConversion;
	boost::shared_ptr<ColorConversion> mp4ColorConversion;
	boost::shared_ptr<ImageViewerModuleExtended> sink;
    RelayPipeline();
    bool setupPipeline();
    bool startPipeline();
    bool stopPipeline();

private:
    PipeLine pipeline;
};