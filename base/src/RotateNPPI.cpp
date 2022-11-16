#include "RotateNPPI.h"
#include "FrameMetadata.h"
#include "RawImageMetadata.h"
#include "RawImagePlanarMetadata.h"

#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "AIPExceptions.h"

#include "npp.h"

class RotateNPPI::Detail
{
public:
	Detail(RotateNPPIProps &_props) : props(_props), shiftX(0), shiftY(0), mFrameType(FrameMetadata::GENERAL), mFrameLength(0)
	{
		nppStreamCtx.hStream = props.stream->getCudaStream();

		if (abs(props.angle) != 90)
		{
			throw AIPException(AIP_NOTIMPLEMENTED, "Only 90 degree rotation supported currently.");
		}
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
				// everything works - just add the NPP functions in compute and uncomment the below lines
				// mOutputMetadata = framemetadata_sp(new RawImagePlanarMetadata(FrameMetadata::MemType::CUDA_DEVICE));
				// break;
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
			RawImageMetadata outputMetadata(rawMetadata->getHeight(), rawMetadata->getWidth(), rawMetadata->getImageType(), rawMetadata->getType(), 512, rawMetadata->getDepth(), FrameMetadata::CUDA_DEVICE, true);
			auto rawOutMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata);
			rawOutMetadata->setData(outputMetadata); // new function required
			imageType = rawMetadata->getImageType();
			depth = rawMetadata->getDepth();
		}
		else if (mFrameType == FrameMetadata::RAW_IMAGE_PLANAR)
		{
			auto rawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
			RawImagePlanarMetadata outputMetadata(rawMetadata->getHeight(0), rawMetadata->getWidth(0), rawMetadata->getImageType(), 512, rawMetadata->getDepth());
			auto rawOutMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(mOutputMetadata);
			rawOutMetadata->setData(outputMetadata);
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
		case ImageMetadata::YUV444:
		case ImageMetadata::YUV420:
		case ImageMetadata::BGRA:
		case ImageMetadata::RGBA:
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

		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "resize failed<" << status << ">";
		}

		return true;
	}

public:
	size_t mFrameLength;
	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
	RotateNPPIProps props;

private:
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

		if (props.angle == 90)
		{
			shiftX = 0;
			shiftY = dstSize[0].height - 1;
		}
		else if (props.angle == -90)
		{
			shiftX = dstSize[0].width - 1;
			shiftY = 0;
		}
		else
		{
			throw AIPException(AIP_NOTIMPLEMENTED, "currently rotation only 90 or -90 is supported");
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

	NppStreamContext nppStreamCtx;
};

RotateNPPI::RotateNPPI(RotateNPPIProps props) : Module(TRANSFORM, "RotateNPPI", props)
{
	mDetail.reset(new Detail(props));
}

RotateNPPI::~RotateNPPI() {}

bool RotateNPPI::validateInputPins()
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

bool RotateNPPI::validateOutputPins()
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

void RotateNPPI::addInputPin(framemetadata_sp &metadata, string &pinId)
{
	Module::addInputPin(metadata, pinId);

	mDetail->setMetadata(metadata);

	mDetail->mOutputMetadata->copyHint(*metadata.get());
	mDetail->mOutputPinId = addOutputPin(mDetail->mOutputMetadata);
}

bool RotateNPPI::init()
{
	if (!Module::init())
	{
		return false;
	}

	return true;
}

bool RotateNPPI::term()
{
	mDetail.reset();
	return Module::term();
}

bool RotateNPPI::process(frame_container &frames)
{
	auto frame = frames.cbegin()->second;
	auto outFrame = makeFrame(mDetail->mFrameLength);

	mDetail->compute(frame->data(), outFrame->data());

	frames.insert(make_pair(mDetail->mOutputPinId, outFrame));
	send(frames);

	return true;
}

bool RotateNPPI::processSOS(frame_sp &frame)
{
	auto metadata = frame->getMetadata();
	mDetail->setMetadata(metadata);
	return true;
}

bool RotateNPPI::shouldTriggerSOS()
{
	return mDetail->mFrameLength == 0;
}

bool RotateNPPI::processEOS(string &pinId)
{
	mDetail->mFrameLength = 0;
	return true;
}