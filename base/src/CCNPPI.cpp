#include "CCNPPI.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "AIPExceptions.h"
#include "CCKernel.h"

#include "npp.h"

class CCNPPI::Detail
{
public:
	Detail(CCNPPIProps &_props) : props(_props)
	{
		nppStreamCtx.hStream = props.stream;
	}

	~Detail()
	{
		
	}

	bool setMetadata(framemetadata_sp& input, framemetadata_sp& output)
	{	
		inputFrameType = input->getFrameType();
		outputFrameType = output->getFrameType();
		if (inputFrameType == FrameMetadata::RAW_IMAGE)
		{
			auto inputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(input);
			inputImageType = inputRawMetadata->getImageType();
			inputChannels = inputRawMetadata->getChannels();
			srcSize[0] = {inputRawMetadata->getWidth(), inputRawMetadata->getHeight()};
			srcRect[0] = { 0, 0, inputRawMetadata->getWidth(), inputRawMetadata->getHeight() };
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
				srcSize[i] = { inputRawMetadata->getWidth(i), inputRawMetadata->getHeight(i) };
				srcRect[i] = { 0, 0, inputRawMetadata->getWidth(i), inputRawMetadata->getHeight(i) };
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

			dstSize[0] = { outputRawMetadata->getWidth(), outputRawMetadata->getHeight() };
			dstRect[0] = { 0, 0, outputRawMetadata->getWidth(), outputRawMetadata->getHeight() };
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
		for(auto i = 0; i < inputChannels; i++)
		{
			src[i] = static_cast<const Npp8u*>(buffer) + srcNextPtrOffset[i];
		}

		for(auto i = 0; i < outputChannels; i++)
		{
			dst[i] = static_cast<Npp8u*>(outBuffer) + dstNextPtrOffset[i];
		}

		if (inputImageType == ImageMetadata::YUV411_I)
		{
			lanuchAPPYUV411ToYUV444(src[0], srcPitch[0], dst, dstPitch[0], srcSize[0], props.stream);
		}
		else if (inputImageType == ImageMetadata::MONO && outputImageType == ImageMetadata::BGRA)
		{
			auto status = nppiDup_8u_C1C4R_Ctx(src[0],
				srcPitch[0],
				dst[0],
				dstPitch[0],
				srcSize[0],
				nppStreamCtx
			);
			if (status != NPP_SUCCESS)
			{
				LOG_ERROR << "nppiCopy_8u_C1C4R_Ctx failed<" << status << ">";
				return false;
			}

			status = nppiSet_8u_C4CR_Ctx(255,
				dst[0] + 3,
				dstPitch[0],
				dstSize[0],
				nppStreamCtx
			);
			if (status != NPP_SUCCESS)
			{
				LOG_ERROR << "nppiSet_8u_C4CR_Ctx failed<" << status << ">";
				return false;
			}

		}
		else if (inputImageType == ImageMetadata::YUV420 && outputImageType == ImageMetadata::BGRA)
		{
			auto status = nppiYUV420ToBGR_8u_P3C4R_Ctx(src,
				srcPitch,
				dst[0],
				dstPitch[0],
				srcSize[0],
				nppStreamCtx
			);

			if (status != NPP_SUCCESS)
			{
				LOG_ERROR << "nppiYUV420ToBGR_8u_P3C4R_Ctx failed<" << status << ">";
			}
		}
		else if (inputImageType == ImageMetadata::BGRA && outputImageType == ImageMetadata::YUV420)
		{
			auto status = nppiBGRToYUV420_8u_AC4P3R_Ctx(src[0],
				srcPitch[0],
				dst,
				dstPitch,
				srcSize[0],
				nppStreamCtx
			);

			if (status != NPP_SUCCESS)
			{
				LOG_ERROR << "nppiBGRToYUV420_8u_AC4P3R_Ctx failed<" << status << ">";
			}
		}
		else if (inputImageType == ImageMetadata::MONO && outputImageType == ImageMetadata::YUV420)
		{
			// CUDA MEMCPY Y
			auto cudaStatus = cudaMemcpy2DAsync(dst[0], dstPitch[0], src[0], srcPitch[0], srcRowSize[0], srcSize[0].height, cudaMemcpyDeviceToDevice, props.stream);
			// CUDA MEMSET U V

			if (cudaStatus != cudaSuccess)
			{
				LOG_ERROR << "copy failed<" << cudaStatus << ">";
				return false;
			}

			cudaStatus = cudaMemset2DAsync(dst[1],
				dstPitch[1],
				128,
				dstSize[1].width,
				dstSize[0].height,
				props.stream
			); 

			if (cudaStatus != cudaSuccess)
			{
				LOG_ERROR << "cudaMemset2DAsync failed<" << cudaStatus << ">";
				return false;
			}

		}
		else
		{
			return false;
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
	const Npp8u* src[4];
	NppiSize srcSize[4];
	NppiRect srcRect[4];
	int srcPitch[4];
	size_t srcNextPtrOffset[4];
	size_t srcRowSize[4];
	Npp8u* dst[4];
	NppiSize dstSize[4];
	NppiRect dstRect[4];
	int dstPitch[4];
	size_t dstNextPtrOffset[4];
	
	CCNPPIProps props;
	NppStreamContext nppStreamCtx;
};

CCNPPI::CCNPPI(CCNPPIProps _props) : Module(TRANSFORM, "CCNPPI", _props), props(_props), mFrameLength(0), mNoChange(false)
{
	mDetail.reset(new Detail(_props));	
}

CCNPPI::~CCNPPI() {}

bool CCNPPI::validateInputPins()
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

bool CCNPPI::validateOutputPins()
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
	if (memType != FrameMetadata::MemType::CUDA_DEVICE)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input memType is expected to be CUDA_DEVICE. Actual<" << memType << ">";
		return false;
	}	

	return true;
}

void CCNPPI::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);

	mInputFrameType = metadata->getFrameType();
	switch (props.imageType)
	{			
	case ImageMetadata::MONO:
	case ImageMetadata::BGR:
	case ImageMetadata::BGRA:
	case ImageMetadata::RGB:
	case ImageMetadata::RGBA:
	case ImageMetadata::YUV411_I:
		mOutputMetadata = framemetadata_sp(new RawImageMetadata(FrameMetadata::MemType::CUDA_DEVICE));
		break;
	case ImageMetadata::YUV420:	
	case ImageMetadata::YUV444:	
		mOutputMetadata = framemetadata_sp(new RawImagePlanarMetadata(FrameMetadata::MemType::CUDA_DEVICE));
		break;
	default:
		throw AIPException(AIP_FATAL, "Unsupported frameType<" + std::to_string(mInputFrameType) + ">");
	}

	mOutputMetadata->copyHint(*metadata.get());
	mOutputPinId = addOutputPin(mOutputMetadata);	
}

bool CCNPPI::init()
{
	if (!Module::init())
	{
		return false;
	}

	auto metadata = getFirstInputMetadata();
	if (metadata->isSet())
	{
		setMetadata(metadata);
	}

	return true;
}

bool CCNPPI::term()
{
	return Module::term();
}

bool CCNPPI::process(frame_container &frames)
{
	auto frame = frames.cbegin()->second;	

	frame_sp outFrame;
	if (!mNoChange)
	{
		outFrame = makeFrame(mFrameLength, mOutputMetadata);
		if (!mDetail->compute(frame->data(), outFrame->data()))
		{
			return true;
		}
	}	
	else
	{
		outFrame = frame;
	}

	frames.insert(make_pair(mOutputPinId, outFrame));
	send(frames);

	return true;
}

bool CCNPPI::processSOS(frame_sp &frame)
{	
	auto metadata = frame->getMetadata();
	setMetadata(metadata);
	return true;
}

void CCNPPI::setMetadata(framemetadata_sp& metadata)
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

	mNoChange = false;
	if (inputImageType == props.imageType)
	{
		mNoChange = true;
		return;
	}

	if ((props.imageType != ImageMetadata::YUV444 || inputImageType != ImageMetadata::YUV411_I)
		&& (props.imageType != ImageMetadata::BGRA || (inputImageType != ImageMetadata::MONO && inputImageType != ImageMetadata::YUV420))
		&& (inputImageType != ImageMetadata::BGRA || props.imageType != ImageMetadata::YUV420)
		&& (inputImageType != ImageMetadata::MONO || props.imageType != ImageMetadata::YUV420)
		)
	{
		throw AIPException(AIP_NOTIMPLEMENTED, "Color conversion not supported");
	}

	if (mOutputFrameType == FrameMetadata::RAW_IMAGE)
	{
		auto rawOutMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata);
		RawImageMetadata outputMetadata(width, height, props.imageType, type, 512, depth, FrameMetadata::CUDA_DEVICE, true);		
		rawOutMetadata->setData(outputMetadata);
	}
	else if (mOutputFrameType == FrameMetadata::RAW_IMAGE_PLANAR)
	{
		auto rawOutMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(mOutputMetadata);
		RawImagePlanarMetadata outputMetadata(width, height, props.imageType, 512, depth);		
		rawOutMetadata->setData(outputMetadata);
	}

	mFrameLength = mOutputMetadata->getDataSize();
	mDetail->setMetadata(metadata, mOutputMetadata);	
}

bool CCNPPI::shouldTriggerSOS()
{
	return mFrameLength == 0;
}

bool CCNPPI::processEOS(string& pinId)
{
	mFrameLength = 0;
	return true;
}