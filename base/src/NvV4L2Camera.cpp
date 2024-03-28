#include "NvV4L2Camera.h"
#include "DMAAllocator.h"
#include "FrameMetadata.h"

NvV4L2Camera::NvV4L2Camera(NvV4L2CameraProps _props)
	: Module(SOURCE, "NvV4L2Camera", _props), props(_props), m_receivedFirstFrame(false)
{
	auto outputMetadata = framemetadata_sp(new RawImageMetadata(FrameMetadata::MemType::DMABUF));
	// DMAAllocator::setMetadata(outputMetadata, props.width, props.height, ImageMetadata::ImageType::YUYV);
	DMAAllocator::setMetadata(outputMetadata, props.width, props.height, ImageMetadata::ImageType::YUYV);
	mOutputPinId = addOutputPin(outputMetadata);

	mHelper = std::make_shared<NvV4L2CameraHelper>([&](frame_sp &frame) -> void {
			frame_container frames;
			std::chrono::time_point<std::chrono::steady_clock> t = std::chrono::steady_clock::now();
			auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(t.time_since_epoch());
			frame->timestamp = dur.count();
			m_receivedFirstFrame = true;
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

	return mHelper->start(props.width, props.height, props.maxConcurrentFrames, props.isMirror, props.sensorType);
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

bool NvV4L2Camera::isFrameBufferReady()
{
	return m_receivedFirstFrame;
}

bool NvV4L2Camera::isCameraConnected()
{
	return mHelper->isCameraConnected();
}