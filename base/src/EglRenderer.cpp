#include "Logger.h"
#include "EglRenderer.h"
#include "ApraNvEglRenderer.h"
#include "DMAFDWrapper.h"
#include "Command.h"

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
        }
    }

	bool shouldTriggerSOS()
	{
		return !renderer;
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
    int height =0;
    
    switch (frameType)
    {
    case FrameMetadata::FrameType::RAW_IMAGE:
    {
        auto metadata = FrameMetadataFactory::downcast<RawImageMetadata>(inputMetadata);
        width = metadata->getWidth();
        height = metadata->getHeight();
    }
    break;
    case FrameMetadata::FrameType::RAW_IMAGE_PLANAR:
    {
        auto metadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(inputMetadata);
        width = metadata->getWidth(0);
        height = metadata->getHeight(0);
    }
    break;
    default:
        throw AIPException(AIP_FATAL, "Unsupported FrameType<" + std::to_string(frameType) + ">");
    }

    mDetail->init(height,width,mDetail->displayOnTop);
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
        mDetail->init(cmd.width, cmd.height,mDetail->displayOnTop);
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

