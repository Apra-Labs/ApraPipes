#include "DMAFDToHostCopy.h"
#include "DMAFrameUtils.h"

class DMAFDToHostCopy::Detail
{
public:
	Detail() : mOutputPinId(""),
			   mFrameType(FrameMetadata::FrameType::RAW_IMAGE),
			   mSize(0), mNumPlanes(0)
	{
	}

public:
	framemetadata_sp mOutputMetadata;
	FrameMetadata::FrameType mFrameType;
	std::string mOutputPinId;
	size_t mSize;
	int mNumPlanes;

	ImagePlanes mImagePlanes;
	DMAFrameUtils::GetImagePlanes mGetImagePlanes;
};

DMAFDToHostCopy::DMAFDToHostCopy(DMAFDToHostCopyProps _props) : Module(TRANSFORM, "DMAFDToHostCopy", _props)
{
	mDetail = std::make_shared<Detail>();
}

DMAFDToHostCopy::~DMAFDToHostCopy()
{
}

bool DMAFDToHostCopy::validateInputPins()
{
	if (getNumberOfInputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	return true;
}

bool DMAFDToHostCopy::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
		return false;
	}

	return true;
}

void DMAFDToHostCopy::addInputPin(framemetadata_sp &metadata, std::string_view pinId)
{
	Module::addInputPin(metadata, pinId);

	if (metadata->getMemType() != FrameMetadata::MemType::DMABUF)
	{
		throw AIPException(AIP_FATAL, "DMABUF Expected. Actual<" + std::to_string(metadata->getMemType()) + ">");
	}

	mDetail->mFrameType = metadata->getFrameType();
	switch (mDetail->mFrameType)
	{
	case FrameMetadata::FrameType::RAW_IMAGE:
		mDetail->mOutputMetadata = framemetadata_sp(new RawImageMetadata(FrameMetadata::MemType::HOST));
		break;
	case FrameMetadata::FrameType::RAW_IMAGE_PLANAR:
		mDetail->mOutputMetadata = framemetadata_sp(new RawImagePlanarMetadata(FrameMetadata::MemType::HOST));
		break;
	default:
		throw AIPException(AIP_FATAL, "Expected Raw Image or RAW_IMAGE_PLANAR. Actual<" + std::to_string(mDetail->mFrameType) + ">");
	}

	mDetail->mOutputMetadata->copyHint(*metadata.get());
	mDetail->mOutputPinId = addOutputPin(mDetail->mOutputMetadata);
}

bool DMAFDToHostCopy::init()
{
	if (!Module::init())
	{
		return false;
	}

	return true;
}

bool DMAFDToHostCopy::term()
{
	return Module::term();
}

bool DMAFDToHostCopy::process(frame_container &frames)
{
	auto frame = frames.cbegin()->second;
	auto outFrame = makeFrame(mDetail->mSize, mDetail->mOutputPinId);

	mDetail->mGetImagePlanes(frame, mDetail->mImagePlanes);

	auto dstPtr = static_cast<uint8_t *>(outFrame->data());
	for (auto i = 0; i < mDetail->mNumPlanes; i++)
	{
		mDetail->mImagePlanes[i]->mCopyToData(mDetail->mImagePlanes[i].get(), dstPtr);
		dstPtr += mDetail->mImagePlanes[i]->imageSize;
	}

	frames[mDetail->mOutputPinId] = outFrame;
	send(frames);

	return true;
}

bool DMAFDToHostCopy::processSOS(frame_sp &frame)
{
	auto inputMetadata = frame->getMetadata();
	if (inputMetadata->getFrameType() != mDetail->mFrameType)
	{
		throw AIPException(AIP_FATAL, "FrameType changed");
	}

	ImageMetadata::ImageType imageType = ImageMetadata::ImageType::YUV420;

	switch (mDetail->mFrameType)
	{
	case FrameMetadata::FrameType::RAW_IMAGE:
	{
		auto inputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(inputMetadata);
		auto outputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mDetail->mOutputMetadata);

		imageType = inputRawMetadata->getImageType();
		RawImageMetadata rawMetadata(inputRawMetadata->getWidth(), inputRawMetadata->getHeight(), imageType, inputRawMetadata->getType(), 0, inputRawMetadata->getDepth(), FrameMetadata::MemType::HOST, true);
		outputRawMetadata->setData(rawMetadata);
	}
	break;
	case FrameMetadata::FrameType::RAW_IMAGE_PLANAR:
	{
		auto inputRawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(inputMetadata);
		auto outputRawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(mDetail->mOutputMetadata);

		imageType = inputRawMetadata->getImageType();
		RawImagePlanarMetadata rawMetadata(inputRawMetadata->getWidth(0), inputRawMetadata->getHeight(0), imageType, size_t(0), inputRawMetadata->getDepth(), FrameMetadata::MemType::HOST);
		outputRawMetadata->setData(rawMetadata);
	}
	break;
	default:
		break;
	}

	// FOR EACH PLANE
	// IF STEP IS EQUAL TO STEP
	// IF STEP NOT EQUAL TO STEP

	switch (imageType)
	{
	case ImageMetadata::ImageType::RGBA:
	case ImageMetadata::ImageType::BGRA:
	case ImageMetadata::ImageType::YUYV:
		break;	
	case ImageMetadata::ImageType::UYVY:
		break;
	case ImageMetadata::ImageType::YUV420:
	case ImageMetadata::ImageType::NV12:
		break;
	default:
		throw AIPException(AIP_FATAL, "Expected <RGBA/BGRA/UYVY/YUV420/NV12> Actual<" + std::to_string(imageType) + ">");
	}

	mDetail->mGetImagePlanes = DMAFrameUtils::getImagePlanesFunction(inputMetadata, mDetail->mImagePlanes);
	mDetail->mSize = mDetail->mOutputMetadata->getDataSize();
	mDetail->mNumPlanes = static_cast<int>(mDetail->mImagePlanes.size());

	return true;
}

bool DMAFDToHostCopy::processEOS(std::string_view pinId)
{
	mDetail->mOutputMetadata->reset();
	return true;
}