#include "H264EncoderV4L2.h"
#include "H264EncoderV4L2Helper.h"

#include "FrameMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "AIPExceptions.h"

H264EncoderV4L2::H264EncoderV4L2(H264EncoderV4L2Props props) : Module(TRANSFORM, "H264EncoderV4L2", props), mProps(props)
{
	mOutputMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::H264_DATA));
	mOutputPinId = addOutputPin(mOutputMetadata);
}

H264EncoderV4L2::~H264EncoderV4L2() 
{
	if (mHelper.get())
	{
		mHelper->processEOS();
		mHelper.reset();
	}
}

bool H264EncoderV4L2::validateInputPins()
{
	if (getNumberOfInputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstInputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::RAW_IMAGE_PLANAR)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be RAW_IMAGE_PLANAR. Actual<" << frameType << ">";
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

bool H264EncoderV4L2::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstOutputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::H264_DATA)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input frameType is expected to be H264_DATA. Actual<" << frameType << ">";
		return false;
	}

	return true;
}

bool H264EncoderV4L2::init()
{
	if (!Module::init())
	{
		return false;
	}

	return true;
}

bool H264EncoderV4L2::term()
{
	mHelper->processEOS();
	mHelper.reset();

	return Module::term();
}

bool H264EncoderV4L2::process(frame_container &frames)
{
	auto frame = frames.cbegin()->second;
	mHelper->process(static_cast<uint8_t*>(frame->data()), frame->size());

	return true;
}

bool H264EncoderV4L2::processSOS(frame_sp &frame)
{
	uint32_t width = 0;
	uint32_t height = 0;

	auto metadata = frame->getMetadata();
	auto frameType = metadata->getFrameType();
	if (frameType == FrameMetadata::RAW_IMAGE)
	{
		auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
		width = static_cast<uint32_t>(rawMetadata->getWidth());
		height = static_cast<uint32_t>(rawMetadata->getHeight());
	}
	else if (frameType == FrameMetadata::RAW_IMAGE_PLANAR)
	{
		auto rawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
		width = static_cast<uint32_t>(rawMetadata->getWidth(0));
		height = static_cast<uint32_t>(rawMetadata->getHeight(0));
	}

	mHelper = std::make_unique<H264EncoderV4L2Helper>(width, height, 1024 * mProps.targetKbps, 30, [&](frame_sp &frame) -> void {
		frame->setMetadata(mOutputMetadata);

		frame_container frames;
		frames.insert(make_pair(mOutputPinId, frame));
		send(frames);
	});

	return true;
}

bool H264EncoderV4L2::shouldTriggerSOS()
{
	return mHelper.get() == nullptr;
}

bool H264EncoderV4L2::processEOS(string& pinId)
{
	mHelper->processEOS();
	return true;
}