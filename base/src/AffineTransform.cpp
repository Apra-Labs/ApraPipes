#include <npp.h>
#include <opencv2/core.hpp>
#include <CuCtxSynchronize.h>
#include "AffineTransform.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "AIPExceptions.h"
#include "math.h"
#include "ImageMetadata.h"
#include "RawImagePlanarMetadata.h"

#define PI 3.14159265

class AffineTransform::Detail
{
public:
	Detail(AffineTransformProps &_props) : props(_props), shiftX(0), shiftY(0), mFrameType(FrameMetadata::GENERAL), mFrameLength(0)
	{
		nppStreamCtx.hStream = props.stream->getCudaStream();
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

		ImageMetadata::ImageType imageType;
		if (mFrameType == FrameMetadata::RAW_IMAGE)
		{
			auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
			int x, y, w, h;
			w = rawMetadata->getWidth();
			h = rawMetadata->getHeight();
			RawImageMetadata outputMetadata(w * props.scale, h * props.scale, rawMetadata->getImageType(), rawMetadata->getType(), rawMetadata->getStep(), rawMetadata->getDepth(), FrameMetadata::CUDA_DEVICE, true);
			auto rawOutMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata);
			rawOutMetadata->setData(outputMetadata);
			imageType = rawMetadata->getImageType();
			depth = rawMetadata->getDepth();
		}

		if (mFrameType == FrameMetadata::RAW_IMAGE_PLANAR)
		{
			auto rawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
			int x, y, w, h;
			w = rawMetadata->getWidth(0);
			h = rawMetadata->getHeight(0);
			RawImagePlanarMetadata outputMetadata(w * props.scale, h * props.scale, rawMetadata->getImageType(), rawMetadata->getStep(0), rawMetadata->getDepth(), FrameMetadata::CUDA_DEVICE);
			auto rawOutMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(mOutputMetadata);
			rawOutMetadata->setData(outputMetadata);
			imageType = rawMetadata->getImageType();
			depth = rawMetadata->getDepth();
		}

		switch (imageType)
		{
		case ImageMetadata::MONO:
			if (depth != CV_8U)
			{
				throw AIPException(AIP_NOTIMPLEMENTED, "Rotate not supported for bit depth<" + std::to_string(depth) + ">");
			}
			break;
		case ImageMetadata::BGR:
		case ImageMetadata::RGB:
		case ImageMetadata::RGBA:
		case ImageMetadata::BGRA:
		case ImageMetadata::YUV444:
		case ImageMetadata::YUV420:
			if (depth != CV_8U)
			{
				throw AIPException(AIP_NOTIMPLEMENTED, "Rotate not supported for bit depth<" + std::to_string(depth) + ">");
			}
			break;
		}

		mFrameLength = mOutputMetadata->getDataSize();
		setMetadataHelper(metadata, mOutputMetadata);
		double si, co;
		si = sin(props.angle * PI / 180);
		co = props.scale * cos(props.angle * PI / 180);
		acoeff[0][0] = co;
		acoeff[0][1] = -si;
		acoeff[0][2] = props.x;
		acoeff[1][0] = si;
		acoeff[1][1] = co;
		acoeff[1][2] = props.y;
	}

	bool compute(void *buffer, void *outBuffer)
	{
		auto status = NPP_SUCCESS;
		auto bufferNPP = static_cast<Npp8u *>(buffer);
		auto outBufferNPP = static_cast<Npp8u *>(outBuffer);
		if (mFrameType == FrameMetadata::RAW_IMAGE_PLANAR)
		{
			for (auto i = 0; i < channels; i++)
			{
				src[i] = bufferNPP + srcNextPtrOffset[i];
				dst[i] = outBufferNPP + dstNextPtrOffset[i];

				status = nppiWarpAffine_8u_C1R_Ctx(src[i],
												   srcSize[i],
												   srcPitch[i],
												   srcRect[i],
												   dst[i],
												   dstPitch[i],
												   dstRect[i],
												   acoeff,
												   NPPI_INTER_NN,
												   nppStreamCtx);
			}
		}

		if (mFrameType == FrameMetadata::RAW_IMAGE)
		{

			if (channels == 1 && depth == CV_8UC1)
			{
				status = nppiWarpAffine_8u_C1R_Ctx(const_cast<const Npp8u *>(bufferNPP),
												   srcSize[0],
												   srcPitch[0],
												   srcRect[0],
												   outBufferNPP,
												   dstPitch[0],
												   dstRect[0],
												   acoeff,
												   NPPI_INTER_NN,
												   nppStreamCtx);
			}
			else if (channels == 3)
			{
				status = nppiWarpAffine_8u_C3R_Ctx(const_cast<const Npp8u *>(bufferNPP),
												   srcSize[0],
												   srcPitch[0],
												   srcRect[0],
												   outBufferNPP,
												   dstPitch[0],
												   dstRect[0],
												   acoeff,
												   NPPI_INTER_NN,
												   nppStreamCtx);
			}
			else if (channels == 4)
			{
				status = nppiWarpAffine_8u_C4R_Ctx(const_cast<const Npp8u *>(bufferNPP),
												   srcSize[0],
												   srcPitch[0],
												   srcRect[0],
												   outBufferNPP,
												   dstPitch[0],
												   dstRect[0],
												   acoeff,
												   NPPI_INTER_NN,
												   nppStreamCtx);
			}
		}

		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "resize failed<" << status << ">";
			throw AIPException(AIP_FATAL, "Failed to tranform the image");
		}
		return true;
	}

	void setProps(AffineTransformProps &mprops)
	{
		if (!mOutputMetadata.get())
		{
			return;
		}
		auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata);
		props = mprops;
	}

public:
	size_t mFrameLength;
	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
	AffineTransformProps props;
	bool setMetadataHelper(framemetadata_sp &input, framemetadata_sp &output)
	{
		if (mFrameType == FrameMetadata::RAW_IMAGE)
		{
			auto inputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(input);
			auto outputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(output);

			channels = inputRawMetadata->getChannels();
			srcSize[0] = {inputRawMetadata->getWidth(), inputRawMetadata->getHeight()};
			srcRect[0] = {0, 0, inputRawMetadata->getWidth(), inputRawMetadata->getHeight()};
			srcPitch[0] = static_cast<int>(inputRawMetadata->getStep());
			srcNextPtrOffset[0] = 0;
			dstSize[0] = {outputRawMetadata->getWidth(), outputRawMetadata->getHeight()};
			dstRect[0] = {0, 0, outputRawMetadata->getWidth(), outputRawMetadata->getHeight()};
			dstPitch[0] = static_cast<int>(outputRawMetadata->getStep());
			dstNextPtrOffset[0] = 0;
		}
		else if (mFrameType == FrameMetadata::RAW_IMAGE_PLANAR)
		{
			auto inputRawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(input);
			auto outputRawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(output);

			channels = inputRawMetadata->getChannels();

			for (auto i = 0; i < channels; i++)
			{
				srcSize[i] = {inputRawMetadata->getWidth(i), inputRawMetadata->getHeight(i)};
				srcRect[i] = {0, 0, inputRawMetadata->getWidth(i), inputRawMetadata->getHeight(i)};
				srcPitch[i] = static_cast<int>(inputRawMetadata->getStep(i));
				srcNextPtrOffset[i] = inputRawMetadata->getNextPtrOffset(i);

				dstSize[i] = {outputRawMetadata->getWidth(i), outputRawMetadata->getHeight(i)};
				dstRect[i] = {0, 0, outputRawMetadata->getWidth(i), outputRawMetadata->getHeight(i)};
				dstPitch[i] = static_cast<int>(outputRawMetadata->getStep(i));
				dstNextPtrOffset[i] = outputRawMetadata->getNextPtrOffset(i);
			}
		}
		return true;
	}

	FrameMetadata::FrameType mFrameType;
	int depth;
	int channels;
	NppiSize srcSize[4];
	NppiRect srcRect[4];
	int srcPitch[4];
	size_t srcNextPtrOffset[4];
	NppiSize dstSize[4];
	NppiRect dstRect[4];
	int dstPitch[4];
	size_t dstNextPtrOffset[4];

	double shiftX;
	double shiftY;
	void *ctx;
	NppStreamContext nppStreamCtx;

	Npp8u *src[3];
	Npp8u *dst[3];
	double acoeff[2][3] = {{-1, -1, -1}, {-1, -1, -1}};
};

AffineTransform::AffineTransform(AffineTransformProps props) : Module(TRANSFORM, "AffineTransform", props)
{
	mDetail.reset(new Detail(props));
}

AffineTransform::~AffineTransform() {}

bool AffineTransform::validateInputPins()
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

bool AffineTransform::validateOutputPins()
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

void AffineTransform::addInputPin(framemetadata_sp &metadata, string &pinId)
{
	Module::addInputPin(metadata, pinId);

	mDetail->setMetadata(metadata);

	mDetail->mOutputMetadata->copyHint(*metadata.get());
	mDetail->mOutputPinId = addOutputPin(mDetail->mOutputMetadata);
}

bool AffineTransform::init()
{
	if (!Module::init())
	{
		return false;
	}

	return true;
}

bool AffineTransform::term()
{
	mDetail.reset();
	return Module::term();
}

bool AffineTransform::process(frame_container &frames)
{
	auto frame = frames.cbegin()->second;
	auto outFrame = makeFrame(mDetail->mFrameLength);
	cudaMemset(outFrame->data(), 0, outFrame->size());
	mDetail->compute(frame->data(), outFrame->data());
	frames.insert(make_pair(mDetail->mOutputPinId, outFrame));
	send(frames);

	return true;
}

bool AffineTransform::processSOS(frame_sp &frame)
{
	auto metadata = frame->getMetadata();
	mDetail->setMetadata(metadata);
	return true;
}

bool AffineTransform::shouldTriggerSOS()
{
	return mDetail->mFrameLength == 0;
}

bool AffineTransform::processEOS(string &pinId)
{
	mDetail->mFrameLength = 0;
	return true;
}

void AffineTransform::setProps(AffineTransformProps &props)
{
	Module::addPropsToQueue(props);
}

AffineTransformProps AffineTransform::getProps()
{
	fillProps(mDetail->props);
	return mDetail->props;
}

bool AffineTransform::handlePropsChange(frame_sp &frame)
{
	AffineTransformProps props(mDetail->props.stream, 0);
	bool ret = Module::handlePropsChange(frame, props);
	mDetail->setProps(props);
	return ret;
}