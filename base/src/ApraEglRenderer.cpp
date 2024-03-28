#include "Logger.h"
#include "ApraEglRenderer.h"
#include "ApraNvEglRenderer.h"
#include "DMAFDWrapper.h"
#include <queue>
#include <chrono>
#include <mutex>

class ApraEglRenderer::Detail
{
public:
    Detail(uint32_t _x_offset, uint32_t _y_offset, uint32_t _width, uint32_t _height)
        : x_offset(_x_offset), y_offset(_y_offset), width(_width), height(_height) ,m_isEglWindowCreated(false) {}

    ~Detail()
    {
        destroyWindow();
    }

    void pushFrame(frame_sp frame)
    {
        // std::lock_guard<std::mutex> lock(queueMutex);
        frameQueue.push(frame);
    }

    void processQueue()
    {
        // std::lock_guard<std::mutex> lock(queueMutex);
        if (!frameQueue.empty())
        {
            auto currentFrame = frameQueue.front();
            frameQueue.pop();
            auto currentTime = std::chrono::steady_clock::now();
            auto timeDiff = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastFrameTime).count();

            if (timeDiff >= 30)
            {
                if (renderer)
                {
                    renderer->render((static_cast<DMAFDWrapper *>(currentFrame->data()))->getFd());
                    lastFrameTime = currentTime;
                }
                else
                {
                    LOG_ERROR << "Renderer not found for rendering frames";
                }
            }
        }
    }

    bool init(uint32_t _height, uint32_t _width)
    {
        LOG_ERROR << "WILL INITIALIZE NEW WINDOW";
        uint32_t displayHeight, displayWidth;
        NvEglRenderer::getDisplayResolution(displayWidth, displayHeight);
        if (height != 0 && width != 0)
        {
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
        if (renderer)
        {
            m_isEglWindowCreated = false;
            delete renderer;
        }
        return true;
    }

    bool shouldTriggerSOS()
    {
        return !renderer;
    }

    NvEglRenderer *renderer = nullptr;
    uint32_t x_offset, y_offset, width, height;
    std::chrono::milliseconds m_frameDelay{30};
    std::queue<frame_sp> frameQueue;
    std::mutex queueMutex;
    bool m_isEglWindowCreated;
    std::chrono::steady_clock::time_point lastFrameTime = std::chrono::steady_clock::now();
};

ApraEglRenderer::ApraEglRenderer(ApraEglRendererProps props) : Module(SINK, "ApraEglRenderer", props)
{
    mDetail.reset(new Detail(props.x_offset, props.y_offset, props.width, props.height));
}

ApraEglRenderer::~ApraEglRenderer() {}

bool ApraEglRenderer::init()
{
    if (!Module::init())
    {
        return false;
    }
    return true;
}

bool ApraEglRenderer::process(frame_container &frames)
{
    for (const auto &pair : frames)
    {
        auto frame = pair.second;
        if (!isFrameEmpty(frame))
        {
            mDetail->pushFrame(frame);
        }
    }
    mDetail->processQueue();
    return true;
}

bool ApraEglRenderer::validateInputPins()
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

bool ApraEglRenderer::term()
{
    bool res = Module::term();
    return res;
}

bool ApraEglRenderer::processSOS(frame_sp &frame)
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
        break;
    }
    case FrameMetadata::FrameType::RAW_IMAGE_PLANAR:
    {
        auto metadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(inputMetadata);
        width = metadata->getWidth(0);
        height = metadata->getHeight(0);
        break;
    }
    default:
        throw AIPException(AIP_FATAL, "Unsupported FrameType<" + std::to_string(frameType) + ">");
    }

    mDetail->init(height, width);
    return true;
}

bool ApraEglRenderer::shouldTriggerSOS()
{
    return mDetail->shouldTriggerSOS();
}

bool ApraEglRenderer::handleCommand(Command::CommandType type, frame_sp &frame)
{
    if (type == Command::CommandType::DeleteWindow)
    {
        EglRendererCloseWindow cmd;
        getCommand(cmd, frame);
        if (mDetail->m_isEglWindowCreated)
        {
            mDetail->destroyWindow();
            return true;
        }
    }
    else if (type == Command::CommandType::CreateWindow)
    {
        EglRendererCreateWindow cmd;
        getCommand(cmd, frame);
        if (!mDetail->m_isEglWindowCreated)
        {
            mDetail->init(cmd.width, cmd.height);
            return true;
        }
    }
    return Module::handleCommand(type, frame);
}

bool ApraEglRenderer::closeWindow()
{
    EglRendererCloseWindow cmd;
    return queueCommand(cmd, true);
}

bool ApraEglRenderer::createWindow(int width, int height)
{
    EglRendererCreateWindow cmd;
    cmd.width = width;
    cmd.height = height;
    return queueCommand(cmd, true);
}

bool ApraEglRenderer::processEOS(string &pinId)
{
    if (m_callbackFunction)
    {
        m_callbackFunction();
    }
    return true;
}

bool ApraEglRenderer::statusOfEglWindow()
{
    return mDetail->m_isEglWindowCreated;
}
