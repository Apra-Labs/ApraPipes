#include "JPEGEncoderL4TMHelper.h"
#include "JPEGEncoderL4TM.h"
#include "FrameMetadata.h"
#include "DMAFDWrapper.h"
#include "Frame.h"
#include "Logger.h"
#include <opencv2/opencv.hpp>
#include "Utils.h"
#include "AIPExceptions.h"

#define MSDK_ALIGN16(value) (((value + 15) >> 4) << 4) // round up to a multiple of 16
#define MSDK_ALIGN32(X) (((uint32_t)((X) + 31)) & (~(uint32_t)31))
#define MSDK_ALIGN7(X) (((uint32_t)((X) + 7)) & (~(uint32_t)7))

class JPEGEncoderL4TM::Detail
{

	class PlanesStrategy
	{
	public:
		PlanesStrategy()
		{
		}

		virtual void fillPlanes(uint8_t **planes, frame_sp &frame)
		{
			planes[0] = static_cast<unsigned char *>(frame->data());
		}
	};

	class PlanesYUVStrategy : public PlanesStrategy
	{
	public:
		PlanesYUVStrategy(int offsetU, int offsetV) : PlanesStrategy(), mOffsetU(offsetU), mOffsetV(offsetV)
		{
		}

		virtual void fillPlanes(uint8_t **planes, frame_sp &frame)
		{
			planes[0] = static_cast<unsigned char *>(frame->data());
			planes[1] = planes[0] + mOffsetU;
			planes[2] = planes[0] + mOffsetV;
		}

	private:
		int mOffsetU;
		int mOffsetV;
	};

	class PlanesDMAYUVStrategy : public PlanesStrategy
	{
	public:
		PlanesDMAYUVStrategy() : PlanesStrategy()
		{
		}

		virtual void fillPlanes(uint8_t **planes, frame_sp &frame)
		{
			auto ptr = static_cast<DMAFDWrapper *>(frame->data());
			planes[0] = static_cast<unsigned char *>(ptr->getHostPtrY());
			planes[1] = static_cast<unsigned char *>(ptr->getHostPtrU());
			planes[2] = static_cast<unsigned char *>(ptr->getHostPtrV());
		}
	};

public:
	Detail(JPEGEncoderL4TMProps &props) : color_space(JCS_YCbCr)
	{
		mProps = JPEGEncoderL4TMProps(props);
	}

	~Detail()
	{
	}

	void setImageMetadata(framemetadata_sp &metadata)
	{
		mMetadata = metadata;
		init(metadata);
	}

	void setOutputMetadata(framemetadata_sp &metadata)
	{
		mOutputMetadata = metadata;
	}

	framemetadata_sp getOutputMetadata()
	{
		return mOutputMetadata;
	}

	size_t compute(frame_sp &inFrame, frame_sp &frame)
	{

		mPlanesStrategy->fillPlanes(&mInBuffer[0], inFrame);

		size_t outLength = mDataSize;
		auto out_buf = (unsigned char *)frame->data();
		encHelper->encode(mInBuffer, &out_buf, outLength);

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
		FrameMetadata::FrameType frameType = metadata->getFrameType();
		if (frameType == FrameMetadata::RAW_IMAGE)
		{
			auto rawImageMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
			if (rawImageMetadata->getChannels() != 1 && rawImageMetadata->getChannels() != 3)
			{
				LOG_ERROR << "<" << id << ">:: RAW_IMAGE CHANNELS IS EXPECTED TO BE 1 or 3";
				return false;
			}

			if (rawImageMetadata->getImageType() != ImageMetadata::MONO && rawImageMetadata->getImageType() != ImageMetadata::RGB)
			{
				LOG_ERROR << "<" << id << ">:: ImageType is expected to be MONO, RGB or YUV420";
				return false;
			}

			if (rawImageMetadata->getStep() != MSDK_ALIGN32(rawImageMetadata->getWidth() * rawImageMetadata->getChannels()))
			{
				LOG_ERROR << "<" << id << ">:: RAW_IMAGE STEP IS EXPECTED TO BE 32 BIT ALIGNED<>" << rawImageMetadata->getWidth() << "<>" << mProps.scale;
				return false;
			}

			return true;
		}

		if (frameType == FrameMetadata::RAW_IMAGE_PLANAR)
		{
			auto rawImagePlanarMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
			if (rawImagePlanarMetadata->getImageType() != ImageMetadata::YUV420)
			{
				LOG_ERROR << "Image Type is expected to be YUV420";
				return false;
			}
			auto channels = rawImagePlanarMetadata->getChannels();
			if (channels != 3)
			{
				LOG_ERROR << "<" << id << ">:: RAW_PLANAR_IMAGE CHANNELS IS EXPECTED TO BE 3";
				return false;
			}

			return true;
		}

		return false;
	}
	std::string outputPinId;

private:
	void init(framemetadata_sp &metadata)
	{
		FrameMetadata::FrameType frameType = metadata->getFrameType();
		if (frameType == FrameMetadata::RAW_IMAGE)
		{
			mPlanesStrategy.reset(new PlanesStrategy);
			auto rawImageMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
			mImageType = rawImageMetadata->getImageType();
			if (rawImageMetadata->getImageType() == ImageMetadata::RGB)
			{
				color_space = JCS_RGB;
			}

			auto step = static_cast<uint32_t>(rawImageMetadata->getStep());
			uint32_t strideArr[3] = {step, step >> 1, step >> 1};
			encHelper.reset(new JPEGEncoderL4TMHelper(mProps.quality));
			encHelper->init(rawImageMetadata->getWidth(), rawImageMetadata->getHeight(), strideArr, color_space, mProps.scale);
			mDataSize = rawImageMetadata->getDataSize();

			if (color_space == JCS_YCbCr)
			{
				dummyBuffer.reset(new unsigned char[mDataSize >> 1]);
				memset(dummyBuffer.get(), 128, mDataSize >> 1);
				mInBuffer[1] = dummyBuffer.get();
				mInBuffer[2] = mInBuffer[1] + (mDataSize >> 2);
			}
			return;
		}

		if (frameType == FrameMetadata::RAW_IMAGE_PLANAR)
		{
			auto rawImagePlanarMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
			mImageType = rawImagePlanarMetadata->getImageType();
			encHelper.reset(new JPEGEncoderL4TMHelper(mProps.quality));
			auto width = rawImagePlanarMetadata->getWidth(0);
			auto height = rawImagePlanarMetadata->getHeight(0);
			uint32_t strideArr[3] = {static_cast<uint32_t>(rawImagePlanarMetadata->getStep(0)), static_cast<uint32_t>(rawImagePlanarMetadata->getStep(1)), static_cast<uint32_t>(rawImagePlanarMetadata->getStep(2))};
			encHelper->init(width, height, strideArr, color_space, mProps.scale);
			mDataSize = rawImagePlanarMetadata->getDataSize();
			if (metadata->getMemType() == FrameMetadata::MemType::HOST)
			{
				mPlanesStrategy.reset(new PlanesYUVStrategy(width * height, width * height * 1.25));
			}
			else if (metadata->getMemType() == FrameMetadata::MemType::DMABUF)
			{
				mPlanesStrategy.reset(new PlanesDMAYUVStrategy);
			}
		}
	}

	boost::shared_ptr<unsigned char[]> dummyBuffer;
	boost::shared_ptr<JPEGEncoderL4TMHelper> encHelper;

	framemetadata_sp mMetadata;
	framemetadata_sp mOutputMetadata;
	size_t mDataSize;
	unsigned char *mInBuffer[3];

	J_COLOR_SPACE color_space;
	ImageMetadata::ImageType mImageType;

	JPEGEncoderL4TMProps mProps;

	class PlanesStrategy;
	boost::shared_ptr<PlanesStrategy> mPlanesStrategy;
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
	if (frameType != FrameMetadata::RAW_IMAGE && frameType != FrameMetadata::RAW_IMAGE_PLANAR)
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

	auto metadata = getOutputMetadataByType(FrameMetadata::ENCODED_IMAGE);
	mDetail->setOutputMetadata(metadata);
	mDetail->outputPinId = getOutputPinIdByType(FrameMetadata::ENCODED_IMAGE);

	return true;
}

bool JPEGEncoderL4TM::term()
{
	return Module::term();
}

bool JPEGEncoderL4TM::process(frame_container &frames)
{
	auto frame = frames.cbegin()->second;
	if (isFrameEmpty(frame))
	{
		return true;
	}

	auto metadata = mDetail->getOutputMetadata();
	auto bufferFrame = makeFrame(mDetail->getDataSize());

	size_t frameLength = mDetail->compute(frame, bufferFrame);

	auto outFrame = makeFrame(bufferFrame, frameLength, mDetail->outputPinId);

	frames.insert(make_pair(mDetail->outputPinId, outFrame));
	send(frames);
	return true;
}

bool JPEGEncoderL4TM::processSOS(frame_sp &frame)
{
	auto metadata = frame->getMetadata();
	if (!mDetail->validateMetadata(metadata, getId()))
	{
		throw AIPException(AIP_NOTIMPLEMENTED, string("validateMetadata failed"));
	}
	// Mdetail-> setstrate (metadata)
	mDetail->setImageMetadata(metadata);

	return true;
}

bool JPEGEncoderL4TM::shouldTriggerSOS()
{
	return mDetail->shouldTriggerSOS();
}

bool JPEGEncoderL4TM::processEOS(string &pinId)
{
	mDetail->resetMetadata();
	return true;
}