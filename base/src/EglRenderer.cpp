#include "Logger.h"
#include "EglRenderer.h"
#include "ApraNvEglRenderer.h"
#include "DMAFDWrapper.h"
#include <drm/drm_fourcc.h>
#include <chrono>
#include <thread>
class EglRenderer::Detail
{

public:
	Detail(uint32_t _x_offset, uint32_t _y_offset, uint32_t _width, uint32_t _height , bool _displayOnTop): x_offset(_x_offset), y_offset(_y_offset), width(_width), height(_height), displayOnTop(_displayOnTop) {}

	~Detail() 
    {
        if(renderer)
        {
            delete renderer;
        }
    }

    bool init(uint32_t _height, uint32_t _width , bool _displayOnTop){
        uint32_t displayHeight, displayWidth;
        NvEglRenderer::getDisplayResolution(displayWidth,displayHeight);
        if(height!=0 && width!=0){
            x_offset += (displayWidth-width)/2;
            y_offset += (displayHeight-height)/2;
            renderer = NvEglRenderer::createEglRenderer(__TIMESTAMP__, width, height, x_offset, y_offset,displayOnTop);
        }else{
            x_offset += (displayWidth-_width)/2;
            y_offset += (displayHeight-_height)/2;
            renderer = NvEglRenderer::createEglRenderer(__TIMESTAMP__, _width, _height, x_offset, y_offset, displayOnTop);
        }
        if (!renderer)
        {
            LOG_ERROR << "Failed to create EGL renderer";
            return false;
        }

        return true;
    }

    bool destroyWindow()
    {
        if(renderer)
        {
            delete renderer;
            renderer = nullptr;
        }
        return true;
    }

    bool shouldTriggerSOS()
    {
        return (!m_isEglWindowCreated) || (!renderer);
    }

	NvEglRenderer *renderer = nullptr;
    uint32_t x_offset,y_offset,width,height;
    bool displayOnTop;
};

EglRenderer::EglRenderer(EglRendererProps props) : Module(SINK, "EglRenderer", props)
{
    mDetail.reset(new Detail(props.x_offset,props.y_offset, props.width, props.height,props.displayOnTop));
}

EglRenderer::~EglRenderer() {}

bool EglRenderer::init(){
    if (!Module::init())
	{
		return false;
	}
    return true;
}

bool EglRenderer::process(frame_container& frames)
{
    auto frame = frames.cbegin()->second;
	if (isFrameEmpty(frame))
	{
		return true;
	}

    mDetail->renderer->render((static_cast<DMAFDWrapper *>(frame->data()))->getFd());
    return true;
}

bool EglRenderer::validateInputPins(){
    if (getNumberOfInputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

    framemetadata_sp metadata = getFirstInputMetadata();
	FrameMetadata::MemType memType = metadata->getMemType();
	if (memType != FrameMetadata::MemType::DMABUF)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is expected to be DMABUF. Actual<" << memType << ">";
		return false;
	}

    return true;
}

bool EglRenderer::term(){
    bool res = Module::term();
    return res;
}

bool EglRenderer::processSOS(frame_sp& frame)
{
    auto inputMetadata = frame->getMetadata();
    auto frameType = inputMetadata->getFrameType();
    int width = 0;
    int height = 0;
    int pitch = 0;
    int fourcc = 0;
    int offset0 = 0, offset1 = 0, offset2 = 0;
    int pitch1_dmabuf = 0, pitch2_dmabuf = 0;

    switch (frameType)
    {
    case FrameMetadata::FrameType::RAW_IMAGE:
    {
        auto metadata = FrameMetadataFactory::downcast<RawImageMetadata>(inputMetadata);
        width = metadata->getWidth();
        height = metadata->getHeight();
        pitch = static_cast<int>(metadata->getStep());
        switch (metadata->getImageType())
        {
        case ImageMetadata::RGBA:
            fourcc = DRM_FORMAT_ABGR8888; // Tegra commonly expects ABGR for RGBA memory
            break;
        case ImageMetadata::BGRA:
            fourcc = DRM_FORMAT_BGRA8888; // Correct mapping for BGRA layouts
            break;
        case ImageMetadata::UYVY:
            fourcc = DRM_FORMAT_UYVY;
            break;
        case ImageMetadata::YUYV:
            fourcc = DRM_FORMAT_YUYV;
            break;
        default:
            fourcc = DRM_FORMAT_RGBA8888;
            break;
        }
    }
    break;
    case FrameMetadata::FrameType::RAW_IMAGE_PLANAR:
    {
        auto metadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(inputMetadata);
        width = metadata->getWidth(0);
        height = metadata->getHeight(0);

        // Use actual dmabuf plane pitches/offsets from producer
        auto dma = static_cast<DMAFDWrapper *>(frame->data());
        auto surf = dma->getNvBufSurface();
        auto &fdParams = surf->surfaceList[0];

        pitch = static_cast<int>(fdParams.planeParams.pitch[0]);
        pitch1_dmabuf = static_cast<int>(fdParams.planeParams.pitch[1]);
        pitch2_dmabuf = static_cast<int>(fdParams.planeParams.pitch[2]);

        offset0 = static_cast<int>(fdParams.planeParams.offset[0]);
        offset1 = static_cast<int>(fdParams.planeParams.offset[1]);
        offset2 = static_cast<int>(fdParams.planeParams.offset[2]);

        if (metadata->getImageType() == ImageMetadata::NV12)
        {
            fourcc = DRM_FORMAT_NV12;
        }
        else if (metadata->getImageType() == ImageMetadata::YUV420)
        {
            fourcc = DRM_FORMAT_YUV420;
        }
    }
    break;
    default:
        throw AIPException(AIP_FATAL, "Unsupported FrameType<" + std::to_string(frameType) + ">");
    }

    mDetail->init(height, width);
    if (mDetail->renderer && fourcc != 0 && pitch != 0)
    {
        if (frameType == FrameMetadata::FrameType::RAW_IMAGE_PLANAR &&
            (fourcc == DRM_FORMAT_NV12 || fourcc == DRM_FORMAT_YUV420))
        {
            int pitch1 = pitch1_dmabuf;
            int pitch2 = (fourcc == DRM_FORMAT_YUV420) ? pitch2_dmabuf : 0;
            int numPlanes = (fourcc == DRM_FORMAT_NV12) ? 2 : 3;
            mDetail->renderer->setImportParamsPlanar(
                fourcc,
                width,
                height,
                pitch,
                offset0,
                pitch1,
                offset1,
                pitch2,
                offset2,
                numPlanes);
        }
        else
        {
            mDetail->renderer->setImportParams(pitch, fourcc, 0, width, height);
        }
    }
    return true;
}

bool EglRenderer::shouldTriggerSOS()
{
	return mDetail->shouldTriggerSOS();
}

bool EglRenderer::handleCommand(Command::CommandType type, frame_sp &frame)
{
    if (type == Command::CommandType::DeleteWindow)
    {
        EglRendererCloseWindow cmd;
        getCommand(cmd, frame);
        mDetail->destroyWindow();
        return true;
    }
    else if (type == Command::CommandType::CreateWindow)
    {
        EglRendererCreateWindow cmd;
        getCommand(cmd, frame);
        if(!mDetail->m_isEglWindowCreated)
        {
            mDetail->init(cmd.height, cmd.width);
        }
        return true;
    }
    return Module::handleCommand(type, frame);
}

bool EglRenderer::closeWindow()
{
    EglRendererCloseWindow cmd;
    return queueCommand(cmd);
}

bool EglRenderer::createWindow(int width, int height)
{
    EglRendererCreateWindow cmd;
    cmd.width = width;
    cmd.height = height;
    return queueCommand(cmd);
}

