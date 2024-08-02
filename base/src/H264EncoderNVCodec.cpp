#include "H264EncoderNVCodec.h"
#include "H264EncoderNVCodecHelper.h"

#include "FrameMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "AIPExceptions.h"
#include "H264FrameDemuxer.h"
#include "nvjpeg.h"
#include "RawImageMetadata.h"
#include "H264Metadata.h"
#include "nvEncodeAPI.h"
#include "Command.h"

#define DEFAULT_BUFFER_THRESHOLD 30

class H264EncoderNVCodec::Detail
{
public:
	Detail(H264EncoderNVCodecProps &_props) : mProps(_props)
	{
		if(_props.bufferThres == DEFAULT_BUFFER_THRESHOLD)
		{
			helper.reset(new H264EncoderNVCodecHelper(_props.bitRateKbps, _props.cuContext,_props.gopLength,_props.frameRate,_props.vProfile,_props.enableBFrames));
		}
		else
		{
			helper.reset(new H264EncoderNVCodecHelper(_props.bitRateKbps, _props.cuContext,_props.gopLength,_props.frameRate,_props.vProfile,_props.enableBFrames,_props.bufferThres));
		}
	}

	~Detail()
	{
		helper.reset();
	}

	bool setMetadata(framemetadata_sp& metadata, std::function<frame_sp(size_t)> makeFrame, std::function<void(frame_sp&, frame_sp&)> send)
	{
		uint32_t width = 0;
		uint32_t height = 0;
		uint32_t pitch = 0;
		ImageMetadata::ImageType imageType = ImageMetadata::UNSET;

		if (metadata->getFrameType() == FrameMetadata::FrameType::RAW_IMAGE)
		{
			auto rawImageMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
			width = static_cast<uint32_t>(rawImageMetadata->getWidth());
			height = static_cast<uint32_t>(rawImageMetadata->getHeight());
			pitch = static_cast<uint32_t>(rawImageMetadata->getStep());
			imageType = rawImageMetadata->getImageType();
		}
		else if (metadata->getFrameType() == FrameMetadata::FrameType::RAW_IMAGE_PLANAR)
		{
			auto rawImagePlanarMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
			width = static_cast<uint32_t>(rawImagePlanarMetadata->getWidth(0));
			height = static_cast<uint32_t>(rawImagePlanarMetadata->getHeight(0));
			pitch = static_cast<uint32_t>(rawImagePlanarMetadata->getStep(0));
			imageType = rawImagePlanarMetadata->getImageType();
		}
		else
		{
			throw AIPException(AIP_NOTIMPLEMENTED, "Unknown frame type");
		}

		if (imageType != ImageMetadata::YUV420 && imageType != ImageMetadata::BGRA && imageType != ImageMetadata::RGBA)
		{
			throw AIPException(AIP_NOTIMPLEMENTED, "Only YUV420 Supported");
		}

		return helper->init(width, height, pitch, imageType, makeFrame, send);
	}

	bool compute(frame_sp& frame)
	{
		return helper->process(frame);
	}

	void endEncode()
	{
		return helper->endEncode();
	}

	bool getSPSPPS(void*& buffer, size_t& size, int& width, int& height)
	{
		return helper->getSPSPPS(buffer, size, width, height);
	}

	H264EncoderNVCodecProps mProps;
private:
	boost::shared_ptr<H264EncoderNVCodecHelper> helper;

};

H264EncoderNVCodec::H264EncoderNVCodec(H264EncoderNVCodecProps _props) : Module(TRANSFORM, "H264EncoderNVCodec", _props), mShouldTriggerSOS(true), props(_props)
{
	mDetail.reset(new Detail(_props));
	mOutputMetadata = framemetadata_sp(new H264Metadata());
	mOutputPinId = addOutputPin(mOutputMetadata);
}

H264EncoderNVCodec::~H264EncoderNVCodec() {}

bool H264EncoderNVCodec::validateInputPins()
{
	if (getNumberOfInputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstInputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::RAW_IMAGE && frameType != FrameMetadata::RAW_IMAGE_PLANAR)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be RAW_IMAGE or RAW_IMAGE_PLANAR. Actual<" << frameType << ">";
		return false;
	}

	FrameMetadata::MemType memType = metadata->getMemType();
	if (memType != FrameMetadata::MemType::CUDA_DEVICE)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is expected to be CUDA_DEVICE. Actual<" << memType << ">";
		return false;
	}

	mInputPinId = getInputPinIdByType(frameType);

	return true;
}

bool H264EncoderNVCodec::validateOutputPins()
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

bool H264EncoderNVCodec::init()
{
	if (!Module::init())
	{
		return false;
	}

	return true;
}

bool H264EncoderNVCodec::term()
{
	mDetail->endEncode();
	mDetail.reset();

	return Module::term();
}

bool H264EncoderNVCodec::process(frame_container &frames)
{
	auto frame = frames.cbegin()->second;

	mDetail->compute(frame);

	return true;
}

bool H264EncoderNVCodec::getSPSPPS(void*& buffer, size_t& size, int& width, int& height)
{
	return mDetail->getSPSPPS(buffer, size, width, height);
}

bool H264EncoderNVCodec::processSOS(frame_sp &frame)
{
	auto metadata=frame->getMetadata();
	mDetail->setMetadata(metadata,
		[&](size_t size) -> frame_sp {return makeFrame(size, mOutputPinId); },
		[&](frame_sp& inputFrame, frame_sp& outputFrame) {

		outputFrame->setMetadata(mOutputMetadata);

		frame_container frames;
		frames.insert(make_pair(mInputPinId, inputFrame));
		frames.insert(make_pair(mOutputPinId, outputFrame));
		send(frames);
	}
	);
	auto inputMetadata = frame->getMetadata();
	int width, height;

	switch (inputMetadata->getFrameType())
	{
		case FrameMetadata::RAW_IMAGE_PLANAR:
		{
			auto rawImageMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(inputMetadata);
			width = rawImageMetadata->getWidth(0);
			height = rawImageMetadata->getHeight(0);
			break;
		}
		case FrameMetadata::RAW_IMAGE:
		{
			auto rawImageMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(inputMetadata);
			width = rawImageMetadata->getWidth();
			height = rawImageMetadata->getHeight();
			break;
		}
		default:
			throw new AIPException(AIP_NOTEXEPCTED, "Unsupported frame type! ");
	}
	

	auto h264Metadata = H264Metadata(width, height);
	auto rawOutMetadata = FrameMetadataFactory::downcast<H264Metadata>(mOutputMetadata);
	rawOutMetadata->setData(h264Metadata);
	mShouldTriggerSOS = false;

	return true;
}

bool H264EncoderNVCodec::shouldTriggerSOS()
{
	return mShouldTriggerSOS;
}

bool H264EncoderNVCodec::processEOS(string& pinId)
{
	mShouldTriggerSOS = true;
	return true;
}