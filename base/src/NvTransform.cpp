#include "NvTransform.h"
#include "nvbuf_utils.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "AIPExceptions.h"
#include "DMAFDWrapper.h"
#include "DMAAllocator.h"

#include "npp.h"

class NvTransform::Detail
{
public:
	Detail(NvTransformProps &_props) : props(_props)
	{
		src_rect.top = _props.top;
		src_rect.left = _props.left;
		src_rect.width = _props.width;
		src_rect.height = _props.height;

		memset(&transParams, 0, sizeof(transParams));
		switch(_props.filterType)
		{
		case NvTransformProps::NvTransformFilter::NEAREST:
			transParams.transform_filter = NvBufferTransform_Filter_Nearest;
			break;
		case NvTransformProps::NvTransformFilter::BILINEAR:
			transParams.transform_filter = NvBufferTransform_Filter_Bilinear;
			break;
		case NvTransformProps::NvTransformFilter::TAP_5:
			transParams.transform_filter = NvBufferTransform_Filter_5_Tap;
			break;
		case NvTransformProps::NvTransformFilter::TAP_10:
			transParams.transform_filter = NvBufferTransform_Filter_10_Tap;
			break;
		case NvTransformProps::NvTransformFilter::SMART:
			transParams.transform_filter = NvBufferTransform_Filter_Smart;
			break;
		case NvTransformProps::NvTransformFilter::NICEST:
			transParams.transform_filter = NvBufferTransform_Filter_Nicest;		
			break;
		default:
			throw AIPException(AIP_FATAL, "Filter Not Supported");
		}
		
		if (src_rect.width != 0)
		{
			transParams.src_rect = src_rect;
			transParams.transform_flag = NVBUFFER_TRANSFORM_FILTER | NVBUFFER_TRANSFORM_CROP_SRC;
		}
		else
		{
			transParams.transform_flag = NVBUFFER_TRANSFORM_FILTER;
		}
	}

	~Detail()
	{
	}

	bool compute(frame_sp &frame, int outFD)
	{
		auto dmaFDWrapper = static_cast<DMAFDWrapper *>(frame->data());
		NvBufferTransform(dmaFDWrapper->getFd(), outFD, &transParams);

		return true;
	}

public:
	NvBufferRect src_rect;
	framemetadata_sp outputMetadata;
	std::string outputPinId;
	NvTransformProps props;

private:
	NvBufferTransformParams transParams;
};

NvTransform::NvTransform(NvTransformProps props) : Module(TRANSFORM, "NvTransform", props)
{
	mDetail.reset(new Detail(props));
}

NvTransform::~NvTransform() {}

bool NvTransform::validateInputPins()
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
	if (memType != FrameMetadata::MemType::DMABUF)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is expected to be DMABUF. Actual<" << memType << ">";
		return false;
	}

	return true;
}

bool NvTransform::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstOutputMetadata();
	auto frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::RAW_IMAGE && frameType != FrameMetadata::RAW_IMAGE_PLANAR)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input frameType is expected to be RAW_IMAGE or RAW_IMAGE_PLANAR. Actual<" << frameType << ">";
		return false;
	}

	FrameMetadata::MemType memType = metadata->getMemType();
	if (memType != FrameMetadata::MemType::DMABUF)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input memType is expected to be DMABUF. Actual<" << memType << ">";
		return false;
	}

	return true;
}

void NvTransform::addInputPin(framemetadata_sp &metadata, string &pinId)
{
	Module::addInputPin(metadata, pinId);
	switch (mDetail->props.imageType)
	{
	case ImageMetadata::BGRA:
	case ImageMetadata::RGBA:
		mDetail->outputMetadata = framemetadata_sp(new RawImageMetadata(FrameMetadata::MemType::DMABUF));
		break;
	case ImageMetadata::NV12:
	case ImageMetadata::YUV420:
		mDetail->outputMetadata = framemetadata_sp(new RawImagePlanarMetadata(FrameMetadata::MemType::DMABUF));
		break;
	default:
		throw AIPException(AIP_FATAL, "Unsupported Image Type<" + std::to_string(mDetail->props.imageType) + ">");
	}

	mDetail->outputMetadata->copyHint(*metadata.get());
	mDetail->outputPinId = addOutputPin(mDetail->outputMetadata);
}

bool NvTransform::init()
{
	if (!Module::init())
	{
		return false;
	}

	return true;
}

bool NvTransform::term()
{
	return Module::term();
}

bool NvTransform::process(frame_container &frames)
{
	auto frame = frames.cbegin()->second;
	try
	{
		auto outFrame = makeFrame(mDetail->outputMetadata->getDataSize(), mDetail->outputPinId);

		if (!outFrame.get())
		{
			LOG_ERROR << "FAILED TO GET BUFFER";
			return false;
		}

		auto dmaFdWrapper = static_cast<DMAFDWrapper *>(outFrame->data());
		dmaFdWrapper->tempFD = dmaFdWrapper->getFd();

		mDetail->compute(frame, dmaFdWrapper->tempFD);

		frames.insert(make_pair(mDetail->outputPinId, outFrame));
		send(frames);
	}
	catch(std::exception & e)
	{
		LOG_ERROR<<"NvTransform seg fault";
	}

	return true;
}

bool NvTransform::processSOS(frame_sp &frame)
{
	auto metadata = frame->getMetadata();
	setMetadata(metadata);

	return true;
}

void NvTransform::setMetadata(framemetadata_sp &metadata)
{
	auto frameType = metadata->getFrameType();
	int width = 0;
	int height = 0;
	int depth = CV_8U;
	ImageMetadata::ImageType inputImageType = ImageMetadata::ImageType::MONO;

	switch (frameType)
	{
	case FrameMetadata::FrameType::RAW_IMAGE:
	{
		auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
		width = rawMetadata->getWidth();
		height = rawMetadata->getHeight();
		depth = rawMetadata->getDepth();
		inputImageType = rawMetadata->getImageType();
	}
	break;
	case FrameMetadata::FrameType::RAW_IMAGE_PLANAR:
	{
		auto rawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
		width = rawMetadata->getWidth(0);
		height = rawMetadata->getHeight(0);
		depth = rawMetadata->getDepth();
		inputImageType = rawMetadata->getImageType();
	}
	break;
	default:
		throw AIPException(AIP_NOTIMPLEMENTED, "Unsupported FrameType<" + std::to_string(frameType) + ">");
	}

	if (mDetail->props.width != 0)
	{
		width = mDetail->props.width;
		height = mDetail->props.height;
	}
	if(mDetail->props.scaleHeight != 0 && mDetail->props.scaleWidth != 0)
	{
		width = width * mDetail->props.scaleWidth;
		height = height * mDetail->props.scaleHeight;
	}


	DMAAllocator::setMetadata(mDetail->outputMetadata, width, height, mDetail->props.imageType);
}

bool NvTransform::processEOS(string &pinId)
{
	//THE FOLLOWING LINE IS COMMENTED FOR SPECIFIC USE IN NVR - MP4READER PASSING EOS WAS COMING HERE AND CAUSING EOS WHICH IS NOT REQUIRED FOR NVR
	// mDetail->outputMetadata.reset();
	return true;
}