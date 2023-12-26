#include "MaskNPPI.h"
#include "RawImageMetadata.h"
#include "RawImagePlanarMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "AIPExceptions.h"
#include "CCKernel.h"
#include "MaskKernel.h"
#include "npp.h"
#include "DMAFDWrapper.h"

class MaskNPPI::Detail
{
public:
	Detail(MaskNPPIProps &_props) : props(_props)
	{
		nppStreamCtx.hStream = props.stream;
	}

	~Detail()
	{
	}
	MaskNPPIProps getProps()
	{
		return props;
	}

	void setProps(MaskNPPIProps &_props)
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
		}

		for (auto i = 0; i < outputChannels; i++)
		{
			dst[i] = static_cast<Npp8u *>(outBuffer) + dstNextPtrOffset[i];
		}
		LOG_ERROR << "COPYING BUFFFER";
		// iterate for no of channels
		auto cudaStatus = cudaMemcpy2DAsync(dst[0], dstPitch[0], src[0], srcPitch[0], srcRowSize[0], srcSize[0].height, cudaMemcpyDeviceToDevice, props.stream);
		LOG_ERROR << "Destination WIDTH " << dstSize[0].width << "Destination Height " << dstSize[0].height;
		LOG_ERROR << "Source WIDTH " << srcSize[0].width << "Source Height " << srcSize[0].height;

		if (inputImageType == ImageMetadata::ImageType::YUYV || inputImageType == ImageMetadata::ImageType::UYVY)
		{
			applyCircularMask((uint8_t *)buffer, (uint8_t *)outBuffer, dstSize[0].width, dstSize[0].height, 200, 200, 50, props.stream);
		}

		else if (inputImageType == ImageMetadata::ImageType::RGBA || inputImageType == ImageMetadata::ImageType::BGRA)
		{
			LOG_ERROR << "HAVING RGBA OR BGRA IMAGE TYPE ";
			applyCircularMaskRGBA((uint8_t *)buffer, (uint8_t *)outBuffer, dstSize[0].width, dstSize[0].height, 200, 200, 50, props.stream);
		}
		// else if (inputImageType == ImageMetadata::ImageType::YUV444 && true && true) // indicator on yuv444 square frame
		// {
		// 	LOG_ERROR << "Will Use Square Mask with Indicator";
		// 	addKernelIndicatorSquareMask((unsigned char *)buffer, (unsigned char *)outBuffer, dstSize[0].width, dstSize[0].height, dstPitch[0], props.stream);
		// }
		// else if (inputImageType == ImageMetadata::ImageType::YUV444 && true)
		// {
		// 	LOG_ERROR << "Will Use Diamond YUV444 Kernel";
		// 	LOG_ERROR << "width is " << dstSize[0].width << "Height is " << dstSize[0].height << "Stride is " << dstPitch[0];
		// 	applyDiamondMaskYUV444((unsigned char *)buffer, (unsigned char *)outBuffer, dstSize[0].width, dstSize[0].height, dstPitch[0], 200, 200, 50, props.stream);
		// }
		else if (inputImageType == ImageMetadata::ImageType::YUV444)
		{
			LOG_ERROR << "Will Use YUV444 Kernel";
			applyCircularMaskYUV444((unsigned char *)buffer, (unsigned char *)outBuffer, srcPitch[0], srcSize[0].height, 200, 200, 50, props.stream);
		}
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

	MaskNPPIProps props;
	NppStreamContext nppStreamCtx;
};

MaskNPPI::MaskNPPI(MaskNPPIProps _props) : Module(TRANSFORM, "MaskNPPI", _props), props(_props), mFrameLength(0)
{
	mDetail.reset(new Detail(_props));
}

MaskNPPI::~MaskNPPI() {}

bool MaskNPPI::validateInputPins()
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

bool MaskNPPI::validateOutputPins()
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

void MaskNPPI::addInputPin(framemetadata_sp &metadata, string &pinId)
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

bool MaskNPPI::init()
{
	if (!Module::init())
	{
		return false;
	}

	return true;
}

bool MaskNPPI::term()
{
	return Module::term();
}

bool MaskNPPI::process(frame_container &frames)
{
	cudaFree(0);
	auto frame = frames.cbegin()->second;
	auto outFrame = makeFrame();
	auto srcCudaPtr = static_cast<DMAFDWrapper *>(frame->data())->getCudaPtr();
	auto dstCudaPtr = static_cast<DMAFDWrapper *>(outFrame->data())->getCudaPtr();

	mDetail->compute((uint8_t *)srcCudaPtr, (uint8_t *)srcCudaPtr);

	frames.insert(make_pair(mOutputPinId, frame));
	send(frames);

	return true;
}

bool MaskNPPI::processSOS(frame_sp &frame)
{
	auto metadata = frame->getMetadata();
	setMetadata(metadata);
	return true;
}

void MaskNPPI::setMetadata(framemetadata_sp &metadata)
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

bool MaskNPPI::shouldTriggerSOS()
{
	return mFrameLength == 0;
}

bool MaskNPPI::processEOS(string &pinId)
{
	mFrameLength = 0;
	return true;
}

MaskNPPIProps MaskNPPI::getProps()
{
	auto props = mDetail->getProps();
	fillProps(props);

	return props;
}

void MaskNPPI::setProps(MaskNPPIProps &props)
{
	Module::addPropsToQueue(props);
}

bool MaskNPPI::handlePropsChange(frame_sp &frame)
{
	auto stream = mDetail->getProps().stream_sp;
	MaskNPPIProps props(stream);
	bool ret = Module::handlePropsChange(frame, props);

	mDetail->setProps(props);

	return ret;
}