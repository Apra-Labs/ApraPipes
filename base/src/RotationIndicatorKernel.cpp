#include "RawImageMetadata.h"
#include "RawImagePlanarMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "AIPExceptions.h"
#include "CCKernel.h"
#include "RotationIndicator.h"
#include "RotationIndicatorKernel.h"
#include "npp.h"
#include "DMAFDWrapper.h"
#include "CudaStreamSynchronize.h"

class RotationIndicatorKernel::Detail
{
public:
	Detail(RotationIndicatorProps &_props) : props(_props)
	{
		nppStreamCtx.hStream = props.stream;
	}

	~Detail()
	{
	}

	RotationIndicatorProps getProps()
	{
		return props;
	}

	void setProps(RotationIndicatorProps &_props)
	{
		props = _props;
	}

	bool setMetadata(framemetadata_sp &input, framemetadata_sp &output)
	{
		inputFrameType = input->getFrameType();
		outputFrameType = output->getFrameType();
		if (inputFrameType == FrameMetadata::RAW_IMAGE)
		{
			auto inputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(input);
			inputImageType = inputRawMetadata->getImageType();
			inputChannels = inputRawMetadata->getChannels();
			srcSize[0] = {inputRawMetadata->getWidth(), inputRawMetadata->getHeight()};
			srcRect[0] = {0, 0, inputRawMetadata->getWidth(), inputRawMetadata->getHeight()};
			srcPitch[0] = static_cast<int>(inputRawMetadata->getStep());
			srcNextPtrOffset[0] = 0;
			srcRowSize[0] = inputRawMetadata->getRowSize();
		}
		else if (inputFrameType == FrameMetadata::RAW_IMAGE_PLANAR)
		{
			auto inputRawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(input);
			inputImageType = inputRawMetadata->getImageType();
			inputChannels = inputRawMetadata->getChannels();

			for (auto i = 0; i < inputChannels; i++)
			{
				srcSize[i] = {inputRawMetadata->getWidth(i), inputRawMetadata->getHeight(i)};
				srcRect[i] = {0, 0, inputRawMetadata->getWidth(i), inputRawMetadata->getHeight(i)};
				srcPitch[i] = static_cast<int>(inputRawMetadata->getStep(i));
				srcNextPtrOffset[i] = inputRawMetadata->getNextPtrOffset(i);
				srcRowSize[i] = inputRawMetadata->getRowSize(i);
			}
		}

		if (outputFrameType == FrameMetadata::RAW_IMAGE)
		{
			auto outputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(output);
			outputImageType = outputRawMetadata->getImageType();
			outputChannels = outputRawMetadata->getChannels();

			dstSize[0] = {outputRawMetadata->getWidth(), outputRawMetadata->getHeight()};
			dstRect[0] = {0, 0, outputRawMetadata->getWidth(), outputRawMetadata->getHeight()};
			dstPitch[0] = static_cast<int>(outputRawMetadata->getStep());
			dstNextPtrOffset[0] = 0;
		}
		else if (outputFrameType == FrameMetadata::RAW_IMAGE_PLANAR)
		{
			auto outputRawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(output);
			outputImageType = outputRawMetadata->getImageType();
			outputChannels = outputRawMetadata->getChannels();

			for (auto i = 0; i < outputChannels; i++)
			{
				dstSize[i] = {outputRawMetadata->getWidth(i), outputRawMetadata->getHeight(i)};
				dstRect[i] = {0, 0, outputRawMetadata->getWidth(i), outputRawMetadata->getHeight(i)};
				dstPitch[i] = static_cast<int>(outputRawMetadata->getStep(i));
				dstNextPtrOffset[i] = outputRawMetadata->getNextPtrOffset(i);
			}
		}

		return true;
	}

	bool compute(void *buffer, void *outBuffer)
	{
		for (auto i = 0; i < inputChannels; i++)
		{
			src[i] = static_cast<const Npp8u *>(buffer) + srcNextPtrOffset[i];
			dst[i] = static_cast<Npp8u *>(outBuffer) + dstNextPtrOffset[i];
		}
		LOG_DEBUG << "Indicator Enabled ============>>>>> Current Angle To Rotate is " << props.rotationAngle << "================";
        applySquareRotationIndicator((unsigned char *)buffer, (unsigned char *)outBuffer, srcPitch[0], srcSize[0].height, props.rotationAngle, props.stream);
        return true;
	}

private:
	FrameMetadata::FrameType inputFrameType;
	FrameMetadata::FrameType outputFrameType;
	ImageMetadata::ImageType inputImageType;
	ImageMetadata::ImageType outputImageType;
	int inputChannels;
	int outputChannels;
	const Npp8u *src[4];
	NppiSize srcSize[4];
	NppiRect srcRect[4];
	int srcPitch[4];
	size_t srcNextPtrOffset[4];
	size_t srcRowSize[4];
	Npp8u *dst[4];
	NppiSize dstSize[4];
	NppiRect dstRect[4];
	int dstPitch[4];
	size_t dstNextPtrOffset[4];

	RotationIndicatorProps props;
	NppStreamContext nppStreamCtx;
};

RotationIndicatorKernel::RotationIndicatorKernel(RotationIndicatorProps _props) : Module(TRANSFORM, "RotationIndicatorKernel", _props), props(_props), mFrameLength(0)
{
	mDetail.reset(new Detail(_props));
}

RotationIndicatorKernel::~RotationIndicatorKernel() {}

bool RotationIndicatorKernel::validateInputPins()
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
		LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is expected to be DMA Memory. Actual<" << memType << ">";
		return false;
	}

	return true;
}

bool RotationIndicatorKernel::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstOutputMetadata();
	mOutputFrameType = metadata->getFrameType();
	if (mOutputFrameType != FrameMetadata::RAW_IMAGE && mOutputFrameType != FrameMetadata::RAW_IMAGE_PLANAR)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input frameType is expected to be RAW_IMAGE or RAW_IMAGE_PLANAR. Actual<" << mOutputFrameType << ">";
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

void RotationIndicatorKernel::addInputPin(framemetadata_sp &metadata, string &pinId)
{
	Module::addInputPin(metadata, pinId);

	mInputFrameType = metadata->getFrameType();
	if (mInputFrameType == FrameMetadata::RAW_IMAGE)
	{
		mOutputMetadata = framemetadata_sp(new RawImageMetadata(FrameMetadata::MemType::DMABUF));
	}
	else if (mInputFrameType == FrameMetadata::RAW_IMAGE_PLANAR)
	{
		mOutputMetadata = framemetadata_sp(new RawImagePlanarMetadata(FrameMetadata::MemType::DMABUF));
	}
	else
	{
		throw AIPException(AIP_NOTIMPLEMENTED, "Mask NPPI not supported for Frame Type<" + std::to_string(mInputFrameType) + ">");
	}

	mOutputMetadata->copyHint(*metadata.get());
	mOutputPinId = addOutputPin(mOutputMetadata);
}

bool RotationIndicatorKernel::init()
{
	if (!Module::init())
	{
		return false;
	}

	return true;
}

bool RotationIndicatorKernel::term()
{
	return Module::term();
}

bool RotationIndicatorKernel::process(frame_container &frames)
{
	cudaFree(0);
	auto frame = frames.cbegin()->second;
	if(mDetail->getProps().enableMask)
	{
		auto outFrame = makeFrame();
		auto srcCudaPtr = static_cast<DMAFDWrapper *>(frame->data())->getCudaPtr();
		auto dstCudaPtr = static_cast<DMAFDWrapper *>(outFrame->data())->getCudaPtr();
		mDetail->compute(srcCudaPtr, dstCudaPtr);
	}
	frames.insert(make_pair(mOutputPinId, frame));
	send(frames);

	return true;
}

bool RotationIndicatorKernel::processSOS(frame_sp &frame)
{
	auto metadata = frame->getMetadata();
	setMetadata(metadata);
	return true;
}

void RotationIndicatorKernel::setMetadata(framemetadata_sp &metadata)
{
	mInputFrameType = metadata->getFrameType();

	int width = NOT_SET_NUM;
	int height = NOT_SET_NUM;
	int type = NOT_SET_NUM;
	int depth = NOT_SET_NUM;
	ImageMetadata::ImageType inputImageType = ImageMetadata::MONO;

	if (mInputFrameType == FrameMetadata::RAW_IMAGE)
	{
		auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
		width = rawMetadata->getWidth();
		height = rawMetadata->getHeight();
		type = rawMetadata->getType();
		depth = rawMetadata->getDepth();
		inputImageType = rawMetadata->getImageType();
	}
	else if (mInputFrameType == FrameMetadata::RAW_IMAGE_PLANAR)
	{
		auto rawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
		width = rawMetadata->getWidth(0);
		height = rawMetadata->getHeight(0);
		depth = rawMetadata->getDepth();
		inputImageType = rawMetadata->getImageType();
	}

	switch (inputImageType)
	{
	case ImageMetadata::YUV444:
	case ImageMetadata::YUYV:
	case ImageMetadata::UYVY:
	case ImageMetadata::RGBA:
	case ImageMetadata::BGRA:
		break;
	default:
		throw AIPException(AIP_NOTIMPLEMENTED, "Mask NPPI not supported for ImageType<" + std::to_string(inputImageType) + ">");
	}

	if (mOutputFrameType == FrameMetadata::RAW_IMAGE)
	{
		auto rawOutMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata);
		RawImageMetadata outputMetadata(width, height, inputImageType, type, 512, depth, FrameMetadata::DMABUF, true);
		rawOutMetadata->setData(outputMetadata);
	}
	else if (mOutputFrameType == FrameMetadata::RAW_IMAGE_PLANAR)
	{
		auto rawOutMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(mOutputMetadata);
		RawImagePlanarMetadata outputMetadata(width, height, inputImageType, 512, depth);
		rawOutMetadata->setData(outputMetadata);
	}

	mFrameLength = mOutputMetadata->getDataSize();
	mDetail->setMetadata(metadata, mOutputMetadata);
}

bool RotationIndicatorKernel::shouldTriggerSOS()
{
	return mFrameLength == 0;
}

bool RotationIndicatorKernel::processEOS(string &pinId)
{
	mFrameLength = 0;
	return true;
}

RotationIndicatorProps RotationIndicatorKernel::getProps()
{
	auto props = mDetail->getProps();
	fillProps(props);
	return props;
}

void RotationIndicatorKernel::setProps(RotationIndicatorProps &props)
{
	Module::addPropsToQueue(props);
}

bool RotationIndicatorKernel::handlePropsChange(frame_sp &frame)
{
	auto stream = mDetail->getProps().stream_sp;
	RotationIndicatorProps props(0, false, stream);
	bool ret = Module::handlePropsChange(frame, props);
	mDetail->setProps(props);

	return ret;
}