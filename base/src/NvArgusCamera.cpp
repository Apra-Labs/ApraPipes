#include "NvArgusCamera.h"

#include "FrameMetadata.h"

NvArgusCamera::NvArgusCamera(NvArgusCameraProps props)
	: Module(SOURCE, "NvArgusCamera", props)
{
	mOutputMetadata = framemetadata_sp(new RawImagePlanarMetadata(props.width, props.height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U, FrameMetadata::MemType::DMABUF));
	mOutputPinId = addOutputPin(mOutputMetadata);

	mHelper = NvArgusCameraHelper::create([&](frame_sp &frame) -> void {
		frame->setMetadata(mOutputMetadata);

		frame_container frames;
		frames.insert(make_pair(mOutputPinId, frame));
		send(frames);
	});

	mHelper->start(props.width, props.height, props.fps);
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

	return true;
}