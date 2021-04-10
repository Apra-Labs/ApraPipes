#include "NvArgusCamera.h"

#include "FrameMetadata.h"

NvArgusCamera::NvArgusCamera(NvArgusCameraProps props)
	: Module(SOURCE, "NvArgusCamera", props), mProps(props)
{
	auto outputMetadata = framemetadata_sp(new RawImagePlanarMetadata(static_cast<int>(props.width), static_cast<int>(props.height), ImageMetadata::ImageType::YUV420, size_t(0), CV_8U, FrameMetadata::MemType::DMABUF));
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
		send(frames); }, [&]() -> frame_sp { return makeFrame(); });
	mHelper->start(mProps.width, mProps.height, static_cast<uint32_t>(mProps.fps));

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