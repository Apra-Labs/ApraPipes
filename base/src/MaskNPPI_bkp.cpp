#include "MaskNPPI.h"
#include "FrameMetadata.h"
#include "RawImageMetadata.h"
#include "RawImagePlanarMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "AIPExceptions.h"
#include <stdint.h>

#include "npp.h"
#include "MaskKernel.h"

class MaskNPPI::Detail
{
public:
	Detail(MaskNPPIProps &_props, std::function<frame_sp(size_t)> _makeFrame) : props(_props), dataSize(0), channels(0), noEffects(false), frameType(FrameMetadata::GENERAL)
	{
		makeFrame = _makeFrame;	
		nppStreamCtx.hStream = _props.stream;
		setProps(_props);	
	}

	~Detail()
	{

	}

	MaskNPPIProps getProps()
	{
		return props;
	}

	void setProps(MaskNPPIProps& _props)
	{		
		props = _props;
	}

	bool setMetadata(framemetadata_sp& input)
	{
		frameType = input->getFrameType();
		if (frameType == FrameMetadata::RAW_IMAGE)
		{
			auto inputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(input);
			channels = inputRawMetadata->getChannels();
			srcSize[0] = { inputRawMetadata->getWidth(), inputRawMetadata->getHeight() };
			rowSize[0] = inputRawMetadata->getRowSize();
			height[0] = inputRawMetadata->getHeight();
			srcPitch[0] = static_cast<int>(inputRawMetadata->getStep());
			nPitch[0] = static_cast<int>(srcPitch[0]);
			srcNextPtrOffset[0] = 0;
		}
		else if (frameType == FrameMetadata::RAW_IMAGE_PLANAR)
		{
			auto inputRawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(input);
			if (inputRawMetadata->getImageType() != ImageMetadata::YUV420)
			{
				throw AIPException(AIP_FATAL, "PLANAR - ONLY YUV420 SUPPORTED");
			}
			channels = inputRawMetadata->getChannels();

			for (auto i = 0; i < channels; i++)
			{
				srcSize[i] = { inputRawMetadata->getWidth(i), inputRawMetadata->getHeight(i) };
				rowSize[i] = inputRawMetadata->getRowSize(i);
				height[i] = inputRawMetadata->getHeight(i);
				srcPitch[i] = static_cast<int>(inputRawMetadata->getStep(i));
				nPitch[i] = static_cast<int>(srcPitch[i]);
				srcNextPtrOffset[i] = inputRawMetadata->getNextPtrOffset(i);
			}
		}

		dataSize = input->getDataSize();

		setProps(props);

		return true;
	}

	bool compute(uint8_t* buffer, uint8_t* outBuffer)
	{
		auto cudaStatus = cudaMemcpy2DAsync((void*)outBuffer, srcPitch[0], (void*) buffer, srcPitch[0], rowSize[0], srcSize[0].height, cudaMemcpyDeviceToDevice, props.stream);
        // applyCircularMask(buffer, outBuffer, 640, 480, 200, 200 , 50, props.stream);
	}

private:
	Npp8u brightness;
	Npp8u brightnessArr[3];
	Npp8u contrast;
	Npp8u contrastArr[3];

	// for yuv420
	Npp32f fBrightness;
	Npp32f fContrast;
	Npp32f fHue;
	Npp32f fSaturation;

	std::function<NppStatus(const Npp8u *, int, const Npp8u, Npp8u *, int, NppiSize, int, NppStreamContext)> monoAdd;
	std::function<NppStatus(const Npp8u *, int, const Npp8u, Npp8u *, int, NppiSize, int, NppStreamContext)> monoMul;
	std::function<NppStatus(const Npp8u *, int, const Npp8u[3], Npp8u *, int, NppiSize, int, NppStreamContext)> bgrAdd;
	std::function<NppStatus(const Npp8u *, int, const Npp8u[3], Npp8u *, int, NppiSize, int, NppStreamContext)> bgrMul;
	std::function<NppStatus(const Npp8u *, int, const Npp8u[3], Npp8u *, int, NppiSize, int, NppStreamContext)> bgraAdd;
	std::function<NppStatus(const Npp8u *, int, const Npp8u[3], Npp8u *, int, NppiSize, int, NppStreamContext)> bgraMul;

	frame_sp bufferMul;

	std::function<frame_sp(size_t)> makeFrame; 
	size_t dataSize;

private:

	FrameMetadata::FrameType frameType;
	int channels;
	NppiSize srcSize[4];
	int nPitch[4];
	size_t srcPitch[4];
	size_t rowSize[4];
	size_t height[4];
	size_t srcNextPtrOffset[4];

	bool noEffects;
	MaskNPPIProps props;
	NppStreamContext nppStreamCtx;
};

MaskNPPI::MaskNPPI(MaskNPPIProps _props) : Module(TRANSFORM, "MaskNPPI", _props), props(_props), mFrameLength(0), mFrameType(FrameMetadata::GENERAL)
{
	mDetail.reset(new Detail(_props, [&](size_t size) -> frame_sp {return makeFrame(size); }));
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
	if (memType != FrameMetadata::MemType::CUDA_DEVICE)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is expected to be CUDA_DEVICE. Actual<" << memType << ">";
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

void MaskNPPI::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);

	setMetadata(metadata);

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
	auto frame = frames.cbegin()->second;
	auto outFrame = makeFrame();

	mDetail->compute((uint8_t *)frame->data(), (uint8_t *)outFrame->data());

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

void MaskNPPI::setMetadata(framemetadata_sp& metadata)
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
		auto rawOutMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata);
		rawOutMetadata->setData(*rawMetadata); // new function required
		imageType = rawMetadata->getImageType();
	}
	else if (mFrameType == FrameMetadata::RAW_IMAGE_PLANAR)
	{
		auto rawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
		auto rawOutMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(mOutputMetadata);
		rawOutMetadata->setData(*rawMetadata);
		imageType = rawMetadata->getImageType();
	}

	switch (imageType)
	{
	case ImageMetadata::YUYV:
    case ImageMetadata::UYVY:
		break;
	default:
		throw AIPException(AIP_NOTIMPLEMENTED, "Resize not supported for ImageType<" + std::to_string(imageType) + ">");
	}

	mFrameLength = mOutputMetadata->getDataSize();
	mDetail->setMetadata(metadata);
	Module::setMetadata(mOutputPinId, mOutputMetadata);
}

bool MaskNPPI::shouldTriggerSOS()
{
	return mFrameLength == 0;
}

bool MaskNPPI::processEOS(string& pinId)
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

void MaskNPPI::setProps(MaskNPPIProps& props)
{
	Module::addPropsToQueue(props);
}

bool MaskNPPI::handlePropsChange(frame_sp& frame)
{
	auto stream = mDetail->getProps().stream_sp;
	MaskNPPIProps props(stream);
	bool ret = Module::handlePropsChange(frame, props);

	mDetail->setProps(props);

	return ret;
}
