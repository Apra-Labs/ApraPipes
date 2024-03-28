#include "Logger.h"
#include "EglRendererReview.h"
#include "ApraNvEglRenderer.h"
#include "DMAFDWrapper.h"

class EglRendererReview::Detail
{

public:
    Detail(uint32_t _x_offset, uint32_t _y_offset, uint32_t _width, uint32_t _height) : x_offset(_x_offset), y_offset(_y_offset), width(_width), height(_height)
    {
        m_isEglWindowCreated = false;
        m_mediaRunning = false;
    }

    ~Detail()
    {
        if (renderer)
        {
            delete renderer;
            renderer = nullptr;
        }
    }

    bool init(uint32_t _height, uint32_t _width)
    {
        LOG_ERROR << "WILL INITIALIZE NEW WINDOW";
        uint32_t displayHeight, displayWidth;
        NvEglRenderer::getDisplayResolution(displayWidth, displayHeight);
        if (height != 0 && width != 0)
        {
            // x_offset += (displayWidth-width)/2;
            // y_offset += (displayHeight-height)/2;
            LOG_DEBUG << "X_OFFSET" << x_offset << "y_offset" << y_offset;
            renderer = NvEglRenderer::createEglRenderer(__TIMESTAMP__, width, height, x_offset, y_offset);
        }
        else
        {
            x_offset += (displayWidth - _width) / 2;
            y_offset += (displayHeight - _height) / 2;
            LOG_DEBUG << "X_OFFSET" << x_offset << "y_offset" << y_offset;
            renderer = NvEglRenderer::createEglRenderer(__TIMESTAMP__, _width, _height, x_offset, y_offset);
        }
        if (!renderer)
        {
            LOG_ERROR << "Failed to create EGL renderer";
            return false;
        }
        m_isEglWindowCreated = true;
        return true;
    }

    bool destroyWindow()
    {
        LOG_ERROR << "GOING TO DESTROY WINDOW";
        if (renderer)
        {
            LOG_ERROR << "Window Exist";
            m_isEglWindowCreated = false;
            delete renderer;
        }
    }

    bool shouldTriggerSOS()
    {
        return !renderer;
    }

    NvEglRenderer *renderer = nullptr;
    uint32_t x_offset, y_offset, width, height;
    std::chrono::milliseconds m_frameDelay{27};
    bool m_isEglWindowCreated;
    bool m_mediaRunning;
};

EglRendererReview::EglRendererReview(EglRendererReviewProps props) : Module(SINK, "EglRendererReview", props)
{
    mDetail.reset(new Detail(props.x_offset, props.y_offset, props.width, props.height));
}

EglRendererReview::~EglRendererReview() {}

bool EglRendererReview::init()
{
    if (!Module::init())
    {
        return false;
    }
    return true;
}

bool EglRendererReview::process(frame_container &frames)
{
    auto frame = frames.cbegin()->second;
    mDetail->m_mediaRunning = true;
    if (isFrameEmpty(frame))
    {
        return true;
    }
    if (mDetail->renderer)
    {
        // LOG_ERROR << "Egl Frame TimeStamp is " << frame->timestamp; 
        mDetail->renderer->render((static_cast<DMAFDWrapper *>(frame->data()))->getFd());
        waitForNextFrame();
    }
    else
    {
        LOG_ERROR << "renderer not found for rendering frames =============================>>>>>>>>>";
    }
    return true;
}

bool EglRendererReview::validateInputPins()
{
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

bool EglRendererReview::term()
{
    bool res = Module::term();
    return res;
}

bool EglRendererReview::processSOS(frame_sp &frame)
{
    auto inputMetadata = frame->getMetadata();
    auto frameType = inputMetadata->getFrameType();
    int width = 0;
    int height = 0;

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

    mDetail->init(height, width);
    return true;
}

bool EglRendererReview::shouldTriggerSOS()
{
    return mDetail->shouldTriggerSOS();
}

bool EglRendererReview::handleCommand(Command::CommandType type, frame_sp &frame)
{
    if (type == Command::CommandType::DeleteWindow)
    {
        LOG_ERROR << "Got Command TO Destroy Window";
        EglRendererCloseWindow cmd;
        getCommand(cmd, frame);
        if (mDetail->m_isEglWindowCreated)
        {
            mDetail->destroyWindow();
        }
        return true;
    }
    else if (type == Command::CommandType::CreateWindow)
    {
        LOG_ERROR << "GOT CREATE WINDOW COMMAND";
        EglRendererCreateWindow cmd;
        getCommand(cmd, frame);
        if(!mDetail->m_isEglWindowCreated)
        {
            mDetail->init(cmd.width, cmd.height);
        }
        return true;
    }
    else
    {
        LOG_ERROR << " In Else :- Type Of Command is " << type;
        return Module::handleCommand(type, frame);
    }
}

bool EglRendererReview::closeWindow()
{
    EglRendererCloseWindow cmd;
    return queueCommand(cmd, false);
}

bool EglRendererReview::createWindow(int width, int height)
{
    LOG_ERROR << "GOT REQUEST TO CREATE WINDOW";
    EglRendererCreateWindow cmd;
    cmd.width = width;
    cmd.height = height;
    return queueCommand(cmd, false);
}

bool EglRendererReview::processEOS(string &pinId)
{
    mDetail->m_mediaRunning = false;
    if (m_callbackFunction)
    {
        m_callbackFunction();
    }
    LOG_ERROR << "WILL CALL CALLBACK FUNCTIONS WILL CALL CALLBACK FUNCTIONS WILL CALL CALLBACK FUNCTIONS WILL CALL CALLBACK FUNCTIONS WILL CALL CALLBACK FUNCTIONS WILL CALL CALLBACK FUNCTIONSWILL CALL CALLBACK FUNCTIONS WILL CALL CALLBACK FUNCTIONS WILL CALL CALLBACK FUNCTIONS";
    return true;
}

void EglRendererReview::waitForNextFrame()
{
    std::this_thread::sleep_for(mDetail->m_frameDelay);
}

bool EglRendererReview::statusOfEglWindow()
{
    return mDetail->m_isEglWindowCreated;
}

bool EglRendererReview::getCurrentStausOfMedia()
{
    LOG_ERROR << "CUrrent State of Media is " << mDetail->m_mediaRunning;
    return mDetail->m_mediaRunning;
}