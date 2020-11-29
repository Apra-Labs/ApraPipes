#include "JPEGEncoderL4TMHelper.h"
#include "JPEGEncoderL4TM.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include <opencv2/opencv.hpp>
#include "Utils.h"
#include "AIPExceptions.h"

#define MSDK_ALIGN16(value)                      (((value + 15) >> 4) << 4) // round up to a multiple of 16
#define MSDK_ALIGN32(X) (((uint32_t)((X)+31)) & (~ (uint32_t)31))
#define MSDK_ALIGN7(X) (((uint32_t)((X)+7)) & (~ (uint32_t)7))

class JPEGEncoderL4TM::Detail
{

public:
	Detail(JPEGEncoderL4TMProps& props): color_space(JCS_YCbCr)
	{
		mProps = JPEGEncoderL4TMProps(props);
	}

	~Detail()
	{
	}

	void setImageMetadata(framemetadata_sp& metadata) 
	{
		mMetadata = metadata;
		init(metadata);
	}

	void setOutputMetadata(framemetadata_sp& metadata) 
	{
		mOutputMetadata = metadata;
	}

	framemetadata_sp getOutputMetadata()
	{
		return mOutputMetadata;
	}

	size_t compute(frame_sp& inFrame, buffer_sp& buffer)
	{
		auto in_buf = static_cast<const unsigned char*>(inFrame->data());

		if(color_space == JCS_YCbCr)
		{
			memcpy(dummyBuffer.get(), inFrame->data(), inFrame->size());	
			in_buf = dummyBuffer.get();
		}		

		size_t outLength = mDataSize;
		auto out_buf = (unsigned char*)buffer->data();	
		encHelper->encode(in_buf, &out_buf, outLength);
		
		return outLength;
	}

	size_t getDataSize()
	{
		return mDataSize;
	}

	bool shouldTriggerSOS()
	{
		return !mMetadata.get();
	}

	void resetMetadata()
	{
		mMetadata.reset();
	}

	bool validateMetadata(framemetadata_sp &metadata, std::string id)
	{
		auto rawImageMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
		if (rawImageMetadata->getChannels() != 1 && rawImageMetadata->getChannels() != 3)
		{
			LOG_ERROR << "<" << id << ">:: RAW_IMAGE CHANNELS IS EXPECTED TO BE 1 or 3";
			return false;
		}

		if(rawImageMetadata->getImageType() != ImageMetadata::MONO && rawImageMetadata->getImageType() != ImageMetadata::RGB)
		{
			LOG_ERROR << "<" << id << ">:: ImageType is expected to be MONO or RGB";
			return false;
		}

		if (rawImageMetadata->getStep()*mProps.scale != MSDK_ALIGN32(rawImageMetadata->getWidth()*rawImageMetadata->getChannels()*mProps.scale))
		{
			LOG_ERROR << "<" << id << ">:: RAW_IMAGE STEP IS EXPECTED TO BE 32 BIT ALIGNED<>" << rawImageMetadata->getWidth() << "<>" << mProps.scale;
			return false;
		}

		return true;
	}

private:

	void init(framemetadata_sp& metadata)
	{		
		auto rawImageMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
		
		if(rawImageMetadata->getImageType() == ImageMetadata::RGB)
		{
			color_space = JCS_RGB;
		}

		encHelper.reset(new JPEGEncoderL4TMHelper(mProps.quality));
		encHelper->init(rawImageMetadata->getWidth(), rawImageMetadata->getHeight(), rawImageMetadata->getStep(), color_space, mProps.scale);
		size_t dummyBufferLength = rawImageMetadata->getDataSize();

		if (color_space == JCS_YCbCr)
		{
			dummyBufferLength *= 1.5;
			dummyBuffer.reset(new unsigned char[dummyBufferLength]);
			memset(dummyBuffer.get(), 128, dummyBufferLength);
		}
		mDataSize = 	dummyBufferLength;	
	}	

	boost::shared_ptr<unsigned char[]> dummyBuffer;
	boost::shared_ptr<JPEGEncoderL4TMHelper> encHelper;

	framemetadata_sp mMetadata;		
	framemetadata_sp mOutputMetadata;	
	size_t mDataSize;	

	J_COLOR_SPACE color_space;

	JPEGEncoderL4TMProps mProps;
};

JPEGEncoderL4TM::JPEGEncoderL4TM(JPEGEncoderL4TMProps props) : Module(TRANSFORM, "JPEGEncoderL4TM", props)
{
	mDetail.reset(new Detail(props));
}

JPEGEncoderL4TM::~JPEGEncoderL4TM() {}

bool JPEGEncoderL4TM::validateInputPins()
{
	if (getNumberOfInputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstInputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::RAW_IMAGE)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be RAW_IMAGE. Actual<" << frameType << ">";
		return false;
	}

	if (metadata->isSet())
	{
		return mDetail->validateMetadata(metadata, getId());
	}

	return true;
}

bool JPEGEncoderL4TM::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstOutputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::ENCODED_IMAGE)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input frameType is expected to be ENCODED_IMAGE. Actual<" << frameType << ">";
		return false;
	}

	return true;
}

bool JPEGEncoderL4TM::init()
{
	if (!Module::init())
	{
		return false;
	}

	auto metadata = getInputMetadataByType(FrameMetadata::RAW_IMAGE);
	if (metadata->isSet())
	{
		mDetail->setImageMetadata(metadata);
	}

	metadata = getOutputMetadataByType(FrameMetadata::ENCODED_IMAGE);
	mDetail->setOutputMetadata(metadata);

	return true;
}

bool JPEGEncoderL4TM::term()
{
	return Module::term();
}

bool JPEGEncoderL4TM::process(frame_container& frames)
{
	auto frame = getFrameByType(frames, FrameMetadata::RAW_IMAGE);
	if (isFrameEmpty(frame))
	{
		return true;
	}

	auto metadata = mDetail->getOutputMetadata();
	auto buffer = makeBuffer(mDetail->getDataSize(), metadata->getMemType());		

	size_t frameLength = mDetail->compute(frame, buffer);
		
	auto outFrame = makeFrame(buffer, frameLength, metadata);

	frames.insert(make_pair(getOutputPinIdByType(FrameMetadata::ENCODED_IMAGE), outFrame));
	send(frames);
	return true;
}

bool JPEGEncoderL4TM::processSOS(frame_sp& frame)
{
	auto metadata = getInputMetadataByType(FrameMetadata::RAW_IMAGE);
	if (!mDetail->validateMetadata(metadata, getId()))
	{
		throw AIPException(AIP_NOTIMPLEMENTED, string("validateMetadata failed"));
	}
	mDetail->setImageMetadata(metadata);

	return true;
}

bool JPEGEncoderL4TM::shouldTriggerSOS()
{
	return mDetail->shouldTriggerSOS();
}

bool JPEGEncoderL4TM::processEOS(string& pinId)
{
	mDetail->resetMetadata();
	return true;
}