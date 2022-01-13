include "AffineTransform.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "AIPExceptions.h"
#include "DMAFDWrapper.h"
#include "math.h"
#include "opencv2/core.hpp"
#include "npp.h"
#include "CuCtxSynchronize.h"

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
				mOutputMetadata = framemetadata_sp(new RawImageMetadata(FrameMetadata::MemType::DMABUF));
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
			int x, y, w, h;
			w = rawMetadata->getWidth();
			h = rawMetadata->getHeight();
			RawImageMetadata outputMetadata(w, h, rawMetadata->getImageType(), rawMetadata->getType(), 512, rawMetadata->getDepth(), FrameMetadata::DMABUF, true);
			auto rawOutMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata);
			rawOutMetadata->setData(outputMetadata); // new function required
			imageType = rawMetadata->getImageType();
			depth = rawMetadata->getDepth();
		}

		switch (imageType)
		{
		case ImageMetadata::MONO:
			if (depth != CV_8U && depth != CV_16U)
			{
				throw AIPException(AIP_NOTIMPLEMENTED, "Rotate not supported for bit depth<" + std::to_string(depth) + ">");
			}
			break;
		case ImageMetadata::BGR:
		case ImageMetadata::RGB:
			if (depth != CV_8U)
			{
				throw AIPException(AIP_NOTIMPLEMENTED, "Rotate not supported for bit depth<" + std::to_string(depth) + ">");
			}
			break;
		case ImageMetadata::RGBA:
			if (depth != CV_8U)
			{
				throw AIPException(AIP_NOTIMPLEMENTED, "Rotate not supported for bit depth<" + std::to_string(depth) + ">");
			}
			break;
		case ImageMetadata::YUV444:
		case ImageMetadata::YUV420:
		case ImageMetadata::BGRA:
		default:
			throw AIPException(AIP_NOTIMPLEMENTED, "Rotate not supported for ImageType<" + std::to_string(imageType) + ">");
		}

		mFrameLength = mOutputMetadata->getDataSize();
		setMetadataHelper(metadata, mOutputMetadata);
	}

	bool compute(void *buffer, void *outBuffer)
	{

		auto status = NPP_SUCCESS;

		// assuming raw_image - planar not supported
		if (channels == 1 && depth == CV_8UC1)
		{
			status = nppiRotate_8u_C1R_Ctx(const_cast<const Npp8u *>(static_cast<Npp8u *>(buffer)),
				srcSize[0],
				srcPitch[0],
				srcRect[0],
				static_cast<Npp8u *>(outBuffer),
				dstPitch[0],
				dstRect[0],
				props.angle,
				shiftX,
				shiftY,
				NPPI_INTER_NN,
				nppStreamCtx);
		}
		else if (channels == 1 && depth == CV_16UC1)
		{
			status = nppiRotate_16u_C1R_Ctx(const_cast<const Npp16u *>(static_cast<Npp16u *>(buffer)),
				srcSize[0],
				srcPitch[0],
				srcRect[0],
				static_cast<Npp16u *>(outBuffer),
				dstPitch[0],
				dstRect[0],
				props.angle,
				shiftX,
				shiftY,
				NPPI_INTER_NN,
				nppStreamCtx);
		}
		else if (channels == 3)
		{
			status = nppiRotate_8u_C3R_Ctx(const_cast<const Npp8u *>(static_cast<Npp8u *>(buffer)),
				srcSize[0],
				srcPitch[0],
				srcRect[0],
				static_cast<Npp8u *>(outBuffer),
				dstPitch[0],
				dstRect[0],
				props.angle,
				shiftX,
				shiftY,
				NPPI_INTER_NN,
				nppStreamCtx);
		}
		else if (channels == 4)
		{
			double si, co;
			si = props.scale * sin(props.angle * PI / 180);
			co = props.scale * cos(props.angle * PI / 180);
			double shx, shy;
			shx = (1 - co) * (srcSize[0].width / 2) + si * srcSize[0].height / 2;
			shy = (-si) * (srcSize[0].width / 2) + (1 - co) * srcSize[0].height / 2;
			double acoeff[2][3] = { {co, -si, shx + props.x}, {si, co, shy + props.y} };
			status = nppiWarpAffine_8u_C4R_Ctx(const_cast<const Npp8u *>(static_cast<Npp8u *>(buffer)),
				srcSize[0],
				srcPitch[0],
				srcRect[0],
				static_cast<Npp8u *>(outBuffer),
				dstPitch[0],
				dstRect[0],
				acoeff,
				NPPI_INTER_NN,
				nppStreamCtx);
		}

		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "resize failed<" << status << ">";
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
			srcSize[0] = { inputRawMetadata->getWidth(), inputRawMetadata->getHeight() };
			srcRect[0] = { 0, 0, inputRawMetadata->getWidth(), inputRawMetadata->getHeight() };
			srcPitch[0] = static_cast<int>(inputRawMetadata->getStep());
			srcNextPtrOffset[0] = 0;
			dstSize[0] = { outputRawMetadata->getWidth(), outputRawMetadata->getHeight() };
			dstRect[0] = { 0, 0, outputRawMetadata->getWidth(), outputRawMetadata->getHeight() };
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
	if (memType != FrameMetadata::MemType::DMABUF)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is expected to be DMABUF. Actual<" << memType << ">";
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
	if (memType != FrameMetadata::MemType::DMABUF)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input memType is expected to be DMABUF. Actual<" << memType << ">";
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
	cudaFree(0);
	cudaMemset(static_cast<DMAFDWrapper *>(outFrame->data())->getCudaPtr(), 0, outFrame->size());

	mDetail->compute(static_cast<DMAFDWrapper *>(frame->data())->getCudaPtr(), static_cast<DMAFDWrapper *>(outFrame->data())->getCudaPtr());
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