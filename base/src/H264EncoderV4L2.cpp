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
		mHelper->stop();
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
	if (frameType != FrameMetadata::RAW_IMAGE_PLANAR && frameType != FrameMetadata::RAW_IMAGE)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be RAW_IMAGE or RAW_IMAGE_PLANAR. Actual<" << frameType << ">";
		return false;
	}

	FrameMetadata::MemType memType = metadata->getMemType();
	if(frameType == FrameMetadata::RAW_IMAGE_PLANAR && (memType != FrameMetadata::HOST && memType != FrameMetadata::DMABUF) )
	{	
		LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is expected to be HOST OR DMABUF. Actual<" << memType << ">";
		return false;
	}
	if (frameType == FrameMetadata::RAW_IMAGE && memType != FrameMetadata::CUDA_DEVICE)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is expected to be CUDA_DEVICE. Actual<" << memType << ">";
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
	return Module::term();
}

bool H264EncoderV4L2::process(frame_container &frames)
{
	auto frame = frames.cbegin()->second;
	mHelper->process(frame);

	return true;
}

bool H264EncoderV4L2::processSOS(frame_sp &frame)
{
	uint32_t width = 0;
	uint32_t height = 0;
	uint32_t step = 0;
	uint32_t pixelFormat = V4L2_PIX_FMT_YUV420M;

	auto metadata = frame->getMetadata();
	auto frameType = metadata->getFrameType();
	if (frameType == FrameMetadata::RAW_IMAGE)
	{
		auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
		width = static_cast<uint32_t>(rawMetadata->getWidth());
		height = static_cast<uint32_t>(rawMetadata->getHeight());
		step = static_cast<uint32_t>(rawMetadata->getStep());
		if (rawMetadata->getImageType() == ImageMetadata::RGB)
		{
			pixelFormat = V4L2_PIX_FMT_RGB24;
		}
		else
		{
			throw AIPException(AIP_NOTIMPLEMENTED, "Only RGB24 Supported");
		}
	}
	else if (frameType == FrameMetadata::RAW_IMAGE_PLANAR)
	{
		auto rawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
		width = static_cast<uint32_t>(rawMetadata->getWidth(0));
		height = static_cast<uint32_t>(rawMetadata->getHeight(0));
		step = static_cast<uint32_t>(rawMetadata->getStep(0));
		if (rawMetadata->getImageType() != ImageMetadata::YUV420)
		{
			throw AIPException(AIP_NOTIMPLEMENTED, "Only YUV420 Supported");
		}
		for(auto i = 0; i < 3; i++)
		{
			if(width != step)
			{
				throw AIPException(AIP_FATAL, "Width is expected to be equal to step");
			}
		}
	}

	auto v4l2MemType = V4L2_MEMORY_MMAP;
	if(metadata->getMemType() == FrameMetadata::DMABUF)
	{
		v4l2MemType = V4L2_MEMORY_DMABUF;
	}

	mHelper = H264EncoderV4L2Helper::create(v4l2MemType, pixelFormat, width, height, step, 1024 * mProps.targetKbps, 30, [&](frame_sp &frame) -> void {
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
	throw AIPException(AIP_FATAL, "H264EncoderV4L2::processEOS not handled");

	return true;
}