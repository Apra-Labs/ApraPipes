#include "JPEGDecoderL4TM.h"
#include "JPEGDecoderL4TMHelper.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include <opencv2/opencv.hpp>
#include "Utils.h"
#include "AIPExceptions.h"

class JPEGDecoderL4TM::Detail
{

public:
	Detail() : mDataSize(0),
			   mWidth2(0),
			   mHeight2(0),
			   mActualFrameSize(0)
	{
		decHelper.reset(new JPEGDecoderL4TMHelper());
	}

	~Detail()
	{
	}

	void initMetadata(frame_sp &frame, framemetadata_sp &metadata)
	{
		auto res = decHelper->init((unsigned char *)frame->data(), frame->size(), mWidth2, mHeight2);
		if (!res)
		{
			throw AIPException(AIP_FATAL, "init:: decHelper->init Failed");
		}

		// This is a cheating
		// Actual output is NV12
		// width and height are aligned by 32 bits - setting stride
		// passing frame->data() to cheat opencv to no allocate any buffer inside
		cv::Mat img(mHeight2, mWidth2, CV_8UC1, frame->data(), mWidth2);
		mDataSize = (mWidth2 * mHeight2 * 3) >> 1;
		mActualFrameSize = mWidth2 * mHeight2;

		FrameMetadataFactory::downcast<RawImageMetadata>(metadata)->setData(img);
		mMetadata = metadata;
	}

	framemetadata_sp getMetadata()
	{
		return mMetadata;
	}

	size_t compute(frame_sp &inFrame, buffer_sp &buffer)
	{
		decHelper->decode((unsigned char *)inFrame->data(), inFrame->size(), (unsigned char *)buffer->data());
		return mActualFrameSize;
	}

	size_t getDataSize()
	{
		return mDataSize;
	}

private:
	boost::shared_ptr<JPEGDecoderL4TMHelper> decHelper;

	framemetadata_sp mMetadata;
	size_t mDataSize;
	size_t mActualFrameSize;
	int mWidth2;
	int mHeight2;
};

JPEGDecoderL4TM::JPEGDecoderL4TM(JPEGDecoderL4TMProps _props) : Module(TRANSFORM, "JPEGDecoderL4TM", _props)
{
	mDetail.reset(new Detail());
}

JPEGDecoderL4TM::~JPEGDecoderL4TM() {}

bool JPEGDecoderL4TM::validateInputPins()
{
	if (getNumberOfInputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstInputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::ENCODED_IMAGE)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be ENCODED_IMAGE. Actual<" << frameType << ">";
		return false;
	}

	return true;
}

bool JPEGDecoderL4TM::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstOutputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::RAW_IMAGE)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input frameType is expected to be RAW_IMAGE. Actual<" << frameType << ">";
		return false;
	}

	return true;
}

bool JPEGDecoderL4TM::init()
{
	if (!Module::init())
	{
		return false;
	}

	auto metadata = getFirstOutputMetadata();
	if (metadata->isSet())
	{
		throw AIPException(AIP_NOTSET, string("JPEGDecoderL4TM Output Frame Metadata parameters will be automatically set. Kindly remove."));
	}

	return true;
}

bool JPEGDecoderL4TM::term()
{
	return Module::term();
}

bool JPEGDecoderL4TM::process(frame_container &frames)
{
	auto frame = getFrameByType(frames, FrameMetadata::ENCODED_IMAGE);
	if (isFrameEmpty(frame))
	{
		return true;
	}

	auto metadata = mDetail->getMetadata();
	auto buffer = makeBuffer(mDetail->getDataSize(), metadata->getMemType());

	auto frameLength = mDetail->compute(frame, buffer);
	
	auto outFrame = makeFrame(buffer, frameLength, metadata);

	frames.insert(make_pair(getOutputPinIdByType(FrameMetadata::RAW_IMAGE), outFrame));
	send(frames);
	return true;
}

bool JPEGDecoderL4TM::processSOS(frame_sp &frame)
{
	auto outMetadata = getOutputMetadataByType(FrameMetadata::RAW_IMAGE);
	mDetail->initMetadata(frame, outMetadata);
	return true;
}