#include "NvV4L2Camera.h"
#include "DMAAllocator.h"
#include "FrameMetadata.h"

NvV4L2Camera::NvV4L2Camera(NvV4L2CameraProps _props)
	: Module(SOURCE, "NvV4L2Camera", _props), props(_props)
{
	auto outputMetadata = framemetadata_sp(new RawImageMetadata(FrameMetadata::MemType::DMABUF));
	DMAAllocator::setMetadata(outputMetadata, props.width, props.height, ImageMetadata::ImageType::UYVY);
	mOutputPinId = addOutputPin(outputMetadata);

	mHelper = std::make_shared<NvV4L2CameraHelper>([&](frame_sp &frame) -> void {
			frame_container frames;
			frames.insert(make_pair(mOutputPinId, frame));
			send(frames); 
		}, [&]() -> frame_sp {
			return makeFrame(); 
		}
	);
}

NvV4L2Camera::~NvV4L2Camera() {}

bool NvV4L2Camera::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		return false;
	}

	return true;
}

bool NvV4L2Camera::init()
{
	if (!Module::init())
	{
		return false;
	}

	return mHelper->start(props.width, props.height, props.maxConcurrentFrames, props.isMirror);
}

bool NvV4L2Camera::term()
{
	auto ret = mHelper->stop();
	mHelper.reset();
	auto moduleRet = Module::term();

	return ret && moduleRet;
}

bool NvV4L2Camera::produce()
{
	return mHelper->queueBufferToCamera();
}