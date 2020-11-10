#include "ResizeNPPI.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "AIPExceptions.h"

#include "npp.h"

class ResizeNPPI::Detail
{
public:
	Detail(ResizeNPPIProps &_props) : props(_props)
	{
		nppStreamCtx.hStream = props.stream;
	}

	~Detail()
	{
		
	}

	bool setMetadata(framemetadata_sp& input, framemetadata_sp& output)
	{	
		frameType = input->getFrameType();
		if (frameType == FrameMetadata::RAW_IMAGE)
		{
			auto inputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(input);
			auto outputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(output);
			channels = inputRawMetadata->getChannels();
			srcSize[0] = {inputRawMetadata->getWidth(), inputRawMetadata->getHeight()};
			srcRect[0] = { 0, 0, inputRawMetadata->getWidth(), inputRawMetadata->getHeight() };
			srcPitch[0] = static_cast<int>(inputRawMetadata->getStep());
			srcNextPtrOffset[0] = 0;

			dstSize[0] = { outputRawMetadata->getWidth(), outputRawMetadata->getHeight() };
			dstRect[0] = { 0, 0, outputRawMetadata->getWidth(), outputRawMetadata->getHeight() };
			dstPitch[0] = static_cast<int>(outputRawMetadata->getStep());
			dstNextPtrOffset[0] = 0;
		}
		else if (frameType == FrameMetadata::RAW_IMAGE_PLANAR)
		{
			auto inputRawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(input);
			auto outputRawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(output);
			channels = inputRawMetadata->getChannels();

			for (auto i = 0; i < channels; i++)
			{
				srcSize[i] = { inputRawMetadata->getWidth(i), inputRawMetadata->getHeight(i) };
				srcRect[i] = { 0, 0, inputRawMetadata->getWidth(i), inputRawMetadata->getHeight(i) };
				srcPitch[i] = static_cast<int>(inputRawMetadata->getStep(i));
				srcNextPtrOffset[i] = inputRawMetadata->getNextPtrOffset(i);

				dstSize[i] = { outputRawMetadata->getWidth(i), outputRawMetadata->getHeight(i) };
				dstRect[i] = { 0, 0, outputRawMetadata->getWidth(i), outputRawMetadata->getHeight(i) };
				dstPitch[i] = static_cast<int>(outputRawMetadata->getStep(i));
				dstNextPtrOffset[i] = outputRawMetadata->getNextPtrOffset(i);
			}
		}
		
		return true;
	}

	bool compute(void* buffer, void* outBuffer)
	{
		
		auto status = NPP_SUCCESS;

		if (frameType == FrameMetadata::RAW_IMAGE_PLANAR)
		{
			for (auto i = 0; i < channels; i++)
			{
				status = nppiResize_8u_C1R_Ctx(static_cast<Npp8u*>(buffer) + srcNextPtrOffset[i],
					srcPitch[i],
					srcSize[i],
					srcRect[i],
					static_cast<Npp8u*>(outBuffer) + dstNextPtrOffset[i],
					dstPitch[i],
					dstSize[i],
					dstRect[i],
					props.eInterpolation,
					nppStreamCtx
				);

				if (status != NPP_SUCCESS)
				{
					break;
				}
			}
		}
		else if (channels == 1)
		{
			status = nppiResize_8u_C1R_Ctx(const_cast<const Npp8u*>(static_cast<Npp8u*>(buffer)),
				srcPitch[0],
				srcSize[0],
				srcRect[0],
				static_cast<Npp8u*>(outBuffer),
				dstPitch[0],
				dstSize[0],
				dstRect[0],
				props.eInterpolation,
				nppStreamCtx
			);			
		}
		else if (channels == 3)
		{
			status = nppiResize_8u_C3R_Ctx(const_cast<const Npp8u *>(static_cast<Npp8u*>(buffer)),
				srcPitch[0],
				srcSize[0],
				srcRect[0],
				static_cast<Npp8u*>(outBuffer),
				dstPitch[0],
				dstSize[0],
				dstRect[0],
				props.eInterpolation,
				nppStreamCtx
			);
		}
		else if (channels == 4)
		{
			status = nppiResize_8u_C4R_Ctx(const_cast<const Npp8u *>(static_cast<Npp8u*>(buffer)),
				srcPitch[0],
				srcSize[0],
				srcRect[0],
				static_cast<Npp8u*>(outBuffer),
				dstPitch[0],
				dstSize[0],
				dstRect[0],
				props.eInterpolation,
				nppStreamCtx);
		}
		else
		{
			throw AIPException(AIP_NOTIMPLEMENTED, "resize not implemented");
		}

		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "resize failed<" << status << ">";
		}

		return true;
	}

private:
	
	FrameMetadata::FrameType frameType;
	int channels;
	NppiSize srcSize[4];
	NppiRect srcRect[4];
	int srcPitch[4];
	size_t srcNextPtrOffset[4];
	NppiSize dstSize[4];
	NppiRect dstRect[4];
	int dstPitch[4];
	size_t dstNextPtrOffset[4];
	
	ResizeNPPIProps props;
	NppStreamContext nppStreamCtx;
};

ResizeNPPI::ResizeNPPI(ResizeNPPIProps _props) : Module(TRANSFORM, "ResizeNPPI", _props), props(_props), mFrameLength(0), mFrameType(FrameMetadata::GENERAL)
{
	mDetail.reset(new Detail(_props));	
}

ResizeNPPI::~ResizeNPPI() {}

bool ResizeNPPI::validateInputPins()
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

	return true;
}

bool ResizeNPPI::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstOutputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::RAW_IMAGE && frameType != FrameMetadata::RAW_IMAGE_PLANAR)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input frameType is expected to be RAW_IMAGE or RAW_IMAGE_PLANAR. Actual<" << frameType << ">";
		return false;
	}

	FrameMetadata::MemType memType = metadata->getMemType();
	if (memType != FrameMetadata::MemType::CUDA_DEVICE)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input memType is expected to be CUDA_DEVICE. Actual<" << memType << ">";
		return false;
	}	

	return true;
}

void ResizeNPPI::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);
	
	setMetadata(metadata);

	mOutputMetadata->copyHint(*metadata.get());
	mOutputPinId = addOutputPin(mOutputMetadata);
}

bool ResizeNPPI::init()
{
	if (!Module::init())
	{
		return false;
	}

	auto metadata = getFirstInputMetadata();
	setMetadata(metadata);	

	return true;
}

bool ResizeNPPI::term()
{
	return Module::term();
}

bool ResizeNPPI::process(frame_container &frames)
{
	auto frame = frames.cbegin()->second;
	auto outFrame = makeFrame(mFrameLength, mOutputMetadata);	

	mDetail->compute(frame->data(), outFrame->data());	

	frames.insert(make_pair(mOutputPinId, outFrame));
	send(frames);

	return true;
}

bool ResizeNPPI::processSOS(frame_sp &frame)
{	
	auto metadata = frame->getMetadata();
	setMetadata(metadata);
	return true;
}

void ResizeNPPI::setMetadata(framemetadata_sp& metadata)
{	
	if (mFrameType != metadata->getFrameType())
	{
		mFrameType = metadata->getFrameType();
		switch (mFrameType)
		{
		case FrameMetadata::RAW_IMAGE:
			mOutputMetadata = framemetadata_sp(new RawImageMetadata(FrameMetadata::MemType::CUDA_DEVICE));
			break;
		case FrameMetadata::RAW_IMAGE_PLANAR:
			mOutputMetadata = framemetadata_sp(new RawImagePlanarMetadata(FrameMetadata::MemType::CUDA_DEVICE));
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
	if (mFrameType == FrameMetadata::RAW_IMAGE)
	{
		auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
		RawImageMetadata outputMetadata(props.width, props.height, rawMetadata->getImageType(), rawMetadata->getType(), 512, rawMetadata->getDepth(), FrameMetadata::CUDA_DEVICE, true);
		auto rawOutMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata);
		rawOutMetadata->setData(outputMetadata); // new function required
		imageType = rawMetadata->getImageType();
	}
	else if (mFrameType == FrameMetadata::RAW_IMAGE_PLANAR)
	{
		auto rawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
		RawImagePlanarMetadata outputMetadata(props.width, props.height, rawMetadata->getImageType(), 512, rawMetadata->getDepth());
		auto rawOutMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(mOutputMetadata);
		rawOutMetadata->setData(outputMetadata);
		imageType = rawMetadata->getImageType();
	}

	switch (imageType)
	{
	case ImageMetadata::MONO:
	case ImageMetadata::YUV444:
	case ImageMetadata::YUV420:
	case ImageMetadata::BGR:
	case ImageMetadata::BGRA:
	case ImageMetadata::RGB:
	case ImageMetadata::RGBA:
		break;	
	default:
		throw AIPException(AIP_NOTIMPLEMENTED, "Resize not supported for ImageType<" + std::to_string(imageType) + ">");
	}

	mFrameLength = mOutputMetadata->getDataSize();
	mDetail->setMetadata(metadata, mOutputMetadata);	
}

bool ResizeNPPI::shouldTriggerSOS()
{
	return mFrameLength == 0;
}

bool ResizeNPPI::processEOS(string& pinId)
{
	mFrameLength = 0;
	return true;
}