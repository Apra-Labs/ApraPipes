#include "RotateCV.h"
#include "FrameMetadata.h"
#include "RawImageMetadata.h"
#include "RawImagePlanarMetadata.h"

#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "AIPExceptions.h"

class RotateCV::Detail
{
public:
	Detail(RotateCVProps &_props) : props(_props), mFrameType(FrameMetadata::GENERAL), mFrameLength(0)
	{
		if (abs(props.angle) != 90)
		{
			throw AIPException(AIP_NOTIMPLEMENTED, "Only 90 degree rotation supported currently.");
		}
		rotateFlag = (props.angle == 90.0) ? cv::ROTATE_90_CLOCKWISE : cv::ROTATE_90_COUNTERCLOCKWISE;
	}

	~Detail()
	{
	}

	void setMetadata(framemetadata_sp &metadata)
	{
		if (mFrameType != metadata->getFrameType())
		{
			mFrameType = metadata->getFrameType();
			switch (mFrameType)
			{
			case FrameMetadata::RAW_IMAGE:
				mOutputMetadata = framemetadata_sp(new RawImageMetadata(FrameMetadata::MemType::HOST));
				break;
			default:
				throw AIPException(AIP_FATAL, "Unsupported frameType<" + std::to_string(mFrameType) + ">");
			}
		}

		if (!metadata->isSet())
		{
			return;
		}

		ImageMetadata::ImageType imageType = ImageMetadata::MONO;
		auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
		height = rawMetadata->getHeight();
		width = rawMetadata->getWidth();
		type = rawMetadata->getType();
		RawImageMetadata outputMetadata(height, width, rawMetadata->getImageType(), rawMetadata->getType(), 0, rawMetadata->getDepth(), FrameMetadata::HOST, true);
		auto rawOutMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata);
		rawOutMetadata->setData(outputMetadata); // new function required
		imageType = rawMetadata->getImageType();
		depth = rawMetadata->getDepth();

		mFrameLength = mOutputMetadata->getDataSize();
	}

	bool compute(void *buffer, void *outBuffer)
	{
		cv::Mat input(height, width, type, buffer);
		cv::Mat output(width, height, type, outBuffer);

		cv::rotate(input, output, rotateFlag);

		return true;
	}

public:
	size_t mFrameLength;
	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;

private:
	FrameMetadata::FrameType mFrameType;
	uint32_t height, width, type, depth;
	RotateCVProps props;
	int rotateFlag;
};

RotateCV::RotateCV(RotateCVProps props) : Module(TRANSFORM, "RotateCV", props)
{
	mDetail.reset(new Detail(props));
}

RotateCV::~RotateCV() {}

bool RotateCV::validateInputPins()
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

	FrameMetadata::MemType memType = metadata->getMemType();
	if (memType != FrameMetadata::MemType::HOST)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is expected to be HOST. Actual<" << memType << ">";
		return false;
	}

	return true;
}

bool RotateCV::validateOutputPins()
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

	FrameMetadata::MemType memType = metadata->getMemType();
	if (memType != FrameMetadata::MemType::HOST)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input memType is expected to be HOST. Actual<" << memType << ">";
		return false;
	}

	return true;
}

void RotateCV::addInputPin(framemetadata_sp &metadata, string &pinId)
{
	Module::addInputPin(metadata, pinId);

	mDetail->setMetadata(metadata);

	mDetail->mOutputMetadata->copyHint(*metadata.get());
	mDetail->mOutputPinId = addOutputPin(mDetail->mOutputMetadata);
}

bool RotateCV::init()
{
	if (!Module::init())
	{
		return false;
	}

	return true;
}

bool RotateCV::term()
{
	mDetail.reset();
	return Module::term();
}

bool RotateCV::process(frame_container &frames)
{
	auto frame = frames.cbegin()->second;
	auto outFrame = makeFrame(mDetail->mFrameLength);

	mDetail->compute(frame->data(), outFrame->data());

	frames.insert({mDetail->mOutputPinId, outFrame});
	send(frames);

	return true;
}

bool RotateCV::processSOS(frame_sp &frame)
{
	auto metadata = frame->getMetadata();
	mDetail->setMetadata(metadata);
	return true;
}

bool RotateCV::shouldTriggerSOS()
{
	return mDetail->mFrameLength == 0;
}

bool RotateCV::processEOS(string &pinId)
{
	mDetail->mFrameLength = 0;
	return true;
}