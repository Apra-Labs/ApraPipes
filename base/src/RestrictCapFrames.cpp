#include "RestrictCapFrames.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "AIPExceptions.h"
#include "DMAFDWrapper.h"
#include "DMAUtils.h"
#include "nvbuf_utils.h"
#include "cudaEGL.h"
#include <Argus/Argus.h>
#include <deque>
#include "npp.h"

class RestrictCapFrames::RestrictCapFramesResetCommand : public Command
{
public:
    RestrictCapFramesResetCommand() : Command(static_cast<Command::CommandType>(Command::CommandType::PipelineReset))
    {
    }
    size_t getSerializeSize()
    {
        return Command::getSerializeSize();
    }

private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive &ar, const unsigned int /* file_version */)
    {
        ar &boost::serialization::base_object<Command>(*this);
    }
};

class RestrictCapFrames::Detail
{
public:
    Detail(RestrictCapFramesProps &_props) : mProps(_props), mFramesSaved(0), enableModule(false)
    {
    }

    ~Detail()
    {
    }

    void resetCurrentFrameSave()
    {
        mFramesSaved = 0;
        enableModule = true;
    }

    void setProps(RestrictCapFramesProps &_props)
    {
        mProps = _props;
    }

public:
    int mFramesSaved;
    bool enableModule;
    FrameMetadata::FrameType mFrameType;
    RestrictCapFramesProps mProps;
};

RestrictCapFrames::RestrictCapFrames(RestrictCapFramesProps _props) : Module(TRANSFORM, "RestrictCapFrames", _props)
{
    mDetail.reset(new Detail(_props));
}

RestrictCapFrames::~RestrictCapFrames() {}

bool RestrictCapFrames::validateInputPins()
{
    framemetadata_sp metadata = getFirstInputMetadata();

    FrameMetadata::FrameType frameType = metadata->getFrameType();
    if (frameType != FrameMetadata::RAW_IMAGE)
    {
        LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be RAW_IMAGE. Actual<" << frameType << ">";
        return false;
    }

    FrameMetadata::MemType memType = metadata->getMemType();
    if (memType != FrameMetadata::MemType::HOST)
    {
        LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is expected to be HOST. Actual<" << memType << ">";
        return false;
    }

    return true;
}

bool RestrictCapFrames::validateOutputPins()
{
    if (getNumberOfOutputPins() != 1)
    {
        LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
        return false;
    }

    framemetadata_sp metadata = getFirstOutputMetadata();
    auto mOutputFrameType = metadata->getFrameType();
    if (mOutputFrameType != FrameMetadata::RAW_IMAGE)
    {
        LOG_ERROR << "<" << getId() << ">::validateOutputPins input frameType is expected to be RAW_IMAGE . Actual<" << mOutputFrameType << ">";
        return false;
    }

    FrameMetadata::MemType memType = metadata->getMemType();
    if (memType != FrameMetadata::MemType::HOST)
    {
        LOG_ERROR << "<" << getId() << ">::validateOutputPins input memType is expected to be HOST. Actual<" << memType << ">";
        return false;
    }

    return true;
}

void RestrictCapFrames::addInputPin(framemetadata_sp &metadata, string &pinId)
{
    Module::addInputPin(metadata, pinId);
    mOutputMetadata = framemetadata_sp(new RawImageMetadata(800, 800, ImageMetadata::ImageType::BG10, CV_16UC1, 2 * 800, CV_16U, FrameMetadata::MemType::HOST));
    mOutputMetadata->copyHint(*metadata.get());
    mOutputPinId = addOutputPin(mOutputMetadata);
}

bool RestrictCapFrames::init()
{
    if (!Module::init())
    {
        return false;
    }
    return true;
}

bool RestrictCapFrames::term()
{
    return Module::term();
}

bool RestrictCapFrames::process(frame_container &frames)
{
    if (mDetail->mFramesSaved < mDetail->mProps.noOfframesToCapture && mDetail->enableModule)
    {
        mDetail->mFramesSaved++;

        if (mDetail->mFramesSaved == mDetail->mProps.noOfframesToCapture)
        {
            mDetail->enableModule = false;
        }

        auto frame = frames.cbegin()->second;
        frames.insert(make_pair(mOutputPinId, frame));
        send(frames);
    }
    return true;
}

bool RestrictCapFrames::processSOS(frame_sp &frame)
{
    auto metadata = frame->getMetadata();
    setMetadata(metadata);
    return true;
}

void RestrictCapFrames::setMetadata(framemetadata_sp &metadata)
{
    if (!metadata->isSet())
    {
        return;
    }
    mOutputMetadata = framemetadata_sp(new RawImageMetadata(800, 800, ImageMetadata::ImageType::BG10, CV_16UC1, 2 * 800, CV_16U, FrameMetadata::MemType::HOST));
}

bool RestrictCapFrames::handleCommand(Command::CommandType type, frame_sp &frame)
{
    if (type == Command::CommandType::PipelineReset)
    {
        RestrictCapFramesResetCommand cmd;
        getCommand(cmd, frame);
        mDetail->resetCurrentFrameSave();
    }

    else
    {
        return Module::handleCommand(type, frame);
    }
}

bool RestrictCapFrames::resetFrameCapture()
{
    RestrictCapFramesResetCommand cmd;
    return queueCommand(cmd);
}

RestrictCapFramesProps RestrictCapFrames::getProps()
{
	fillProps(mDetail->mProps);
	return mDetail->mProps;
}

void RestrictCapFrames::setProps(RestrictCapFramesProps &props)
{
	Module::addPropsToQueue(props);
}

bool RestrictCapFrames::handlePropsChange(frame_sp &frame)
{
	RestrictCapFramesProps props(0);
	bool ret = Module::handlePropsChange(frame, props);
	mDetail->setProps(props);
	return ret;
}