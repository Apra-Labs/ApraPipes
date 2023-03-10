#include "FrameMetadata.h"
#include "Frame.h"

class FacialLandmarkCV::Detail
{
public:
	Detail(FacialLandmarkCVProps &_props) : props(_props), mFrameType(FrameMetadata::GENERAL), mFrameLength(0)
	{
	}

	~Detail()
	{
	}

	void setMetadata(framemetadata_sp &metadata)
	{
	}

	bool compute(void *buffer, void *outBuffer)
	{
		cv::Mat input(height, width, type, buffer);
		cv::Mat output(width, height, type, outBuffer);

	

		return true;
	}

public:
	size_t mFrameLength;
	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;

private:
	FrameMetadata::FrameType mFrameType;
	uint32_t height, width, type;
	FacialLandmarkCVProps props;
};

FacialLandmarkCV::FacialLandmarkCV(FacialLandmarkCVProps props) : Module(TRANSFORM, "FacialLandmarkCV", props)
{
	mDetail.reset(new Detail(props));
}

FacialLandmarkCV::~FacialLandmarkCV() {}

bool FacialLandmarkCV::validateInputPins()
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

bool FacialLandmarkCV::validateOutputPins()
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

void FacialLandmarkCV::addInputPin(framemetadata_sp &metadata, string &pinId)
{
	Module::addInputPin(metadata, pinId);

	mDetail->setMetadata(metadata);

	mDetail->mOutputMetadata->copyHint(*metadata.get());
	mDetail->mOutputPinId = addOutputPin(mDetail->mOutputMetadata);
}

bool FacialLandmarkCV::init()
{
	if (!Module::init())
	{
		return false;
	}

	return true;
}

bool FacialLandmarkCV::term()
{
	mDetail.reset();
	return Module::term();
}

bool FacialLandmarkCV::process(frame_container &frames)
{
	auto frame = frames.cbegin()->second;
	auto outFrame = makeFrame(mDetail->mFrameLength);

	mDetail->compute(frame->data(), outFrame->data());

	frames.insert(make_pair(mDetail->mOutputPinId, outFrame));
	send(frames);

	return true;
}

bool FacialLandmarkCV::processSOS(frame_sp &frame)
{
	auto metadata = frame->getMetadata();
	mDetail->setMetadata(metadata);
	return true;
}

bool FacialLandmarkCV::shouldTriggerSOS()
{
	return mDetail->mFrameLength == 0;
}

bool FacialLandmarkCV::processEOS(string &pinId)
{
	mDetail->mFrameLength = 0;
	return true;
}