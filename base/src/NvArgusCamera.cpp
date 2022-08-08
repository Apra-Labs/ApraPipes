#include "NvArgusCamera.h"

#include "DMAAllocator.h"
#include "FrameMetadata.h"
#include <sys/time.h>
#include <ctime>
#define CAMERA_HINT1 "Camera1"
#define CAMERA_HINT2 "Camera2"
#define CAMERA_HINT3 "Camera3"

class NvArgusCamera::NvArgusCameraSetAWBCommand : public Command
{
public:
    NvArgusCameraSetAWBCommand() : Command(static_cast<Command::CommandType>(Command::CommandType::SetAWB))
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

class NvArgusCamera::NvArgusCameraEnableAWBCommand : public Command
{
public:
    NvArgusCameraEnableAWBCommand() : Command(static_cast<Command::CommandType>(Command::CommandType::EnableAWB))
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

class NvArgusCamera::NvArgusCameraDisableAWBCommand : public Command
{
public:
    NvArgusCameraDisableAWBCommand() : Command(static_cast<Command::CommandType>(Command::CommandType::DisableAWB))
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

NvArgusCamera::NvArgusCamera(NvArgusCameraProps props)
	: Module(SOURCE, "NvArgusCamera", props), mProps(props)
{
	auto outputMetadata = framemetadata_sp(new RawImagePlanarMetadata(FrameMetadata::MemType::DMABUF));
	DMAAllocator::setMetadata(outputMetadata, static_cast<int>(props.width), static_cast<int>(props.height), ImageMetadata::ImageType::NV12);	
	mOutputPinId = addOutputPin(outputMetadata); 
}

NvArgusCamera::~NvArgusCamera() {}

bool NvArgusCamera::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		return false;
	}

	return true;
}

bool NvArgusCamera::init()
{
	if (!Module::init())
	{
		return false;
	}

	mHelper = NvArgusCameraHelper::create(
		mProps.maxConcurrentFrames, [&](frame_sp &frame) -> void {
		frame_container frames;
		frames.insert(make_pair(mOutputPinId, frame));
		struct timeval time_now{};
	    gettimeofday(&time_now, nullptr);
    	time_t msecs_time = (time_now.tv_sec * 1000) + (time_now.tv_usec / 1000);
		frame->timestamp =  msecs_time;
		send(frames); 
		}, [&]() -> frame_sp { return makeFrame(); });
	mHelper->start(mProps.width, mProps.height, static_cast<uint32_t>(mProps.fps), mProps.cameraId);

	return true;
}

bool NvArgusCamera::term()
{
	auto ret = mHelper->stop();
	mHelper.reset();
	auto moduleRet = Module::term();

	return ret && moduleRet;
}

bool NvArgusCamera::produce()
{
	mHelper->queueFrameToCamera();
	return true;
}

bool NvArgusCamera::handleCommand(Command::CommandType type, frame_sp &frame)
{
    if (type == Command::CommandType::SetAWB)
    {
        NvArgusCameraSetAWBCommand cmd;
        getCommand(cmd, frame);
        mHelper->toggleAutoWhiteBalance();
    }
    else if (type == Command::CommandType::EnableAWB)
    {
        NvArgusCameraEnableAWBCommand cmd;
        getCommand(cmd, frame);
        mHelper->enableAutoWhiteBalance();
    }
	else if (type == Command::CommandType::DisableAWB)
    {
        NvArgusCameraDisableAWBCommand cmd;
        getCommand(cmd, frame);
        mHelper->disableAutoWhiteBalance();
    }
    else
    {
        return Module::handleCommand(type, frame);
    }
}

bool NvArgusCamera::toggleAutoWB()
{
    NvArgusCameraSetAWBCommand cmd;
    return queueCommand(cmd);
}

bool NvArgusCamera::enableAutoWB()
{
    NvArgusCameraEnableAWBCommand cmd;
    return queueCommand(cmd);
}

bool NvArgusCamera::disableAutoWB()
{
    NvArgusCameraDisableAWBCommand cmd;
    return queueCommand(cmd);
}