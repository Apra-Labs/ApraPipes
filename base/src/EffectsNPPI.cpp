#include "EffectsNPPI.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "AIPExceptions.h"

#include "npp.h"
#include "EffectsKernel.h"

class EffectsNPPI::Detail
{
public:
	Detail(EffectsNPPIProps &_props, std::function<buffer_sp(size_t)> _makeBuffer) : props(_props), dataSize(0), channels(0), noEffects(false), frameType(FrameMetadata::GENERAL)
	{
		nppStreamCtx.hStream = _props.stream;
		setProps(_props);	
		makeBuffer = _makeBuffer;	
	}

	~Detail()
	{

	}

	EffectsNPPIProps getProps()
	{
		return props;
	}

	void setProps(EffectsNPPIProps& _props)
	{		
		props = _props;
		if (channels == 0)
		{
			// width height not available 
			return;
		}

		if (props.contrast < 0)
		{
			props.contrast = 0;
		}

		if (frameType == FrameMetadata::RAW_IMAGE)
		{
			if (props.hue != 0 || props.saturation != 1)
			{
				LOG_ERROR << "Hue and Saturation not implemented for mono/rgba";
			}

			if (props.brightness == 0 && props.contrast == 1)
			{
				noEffects = true;
				return;
			}
		}
		else if (frameType == FrameMetadata::RAW_IMAGE_PLANAR)
		{
			if (props.brightness == 0 && props.contrast == 1 && props.hue == 0 && props.saturation == 1)
			{
				noEffects = true;
				return;
			}

			fBrightness = static_cast<Npp32f>(props.brightness);
			fContrast = static_cast<Npp32f>(props.contrast);
			fHue = static_cast<Npp32f>(props.hue);
			fSaturation = static_cast<Npp32f>(props.saturation);
		}

		noEffects = false;
		

		if (props.brightness > 0)
		{
			brightness = props.brightness;
			monoAdd = nppiAddC_8u_C1RSfs_Ctx;
			bgrAdd = nppiAddC_8u_C3RSfs_Ctx;
			bgraAdd = nppiAddC_8u_C4RSfs_Ctx;
		}
		else
		{
			brightness = -1 * props.brightness;
			monoAdd = nppiSubC_8u_C1RSfs_Ctx;
			bgrAdd = nppiSubC_8u_C3RSfs_Ctx;
			bgraAdd = nppiSubC_8u_C4RSfs_Ctx;
		}

		if (props.contrast > 1 || props.contrast == 0)
		{
			contrast = static_cast<Npp8u>(props.contrast);
			monoMul = nppiMulC_8u_C1RSfs_Ctx;
			bgrMul = nppiMulC_8u_C3RSfs_Ctx;
			bgraMul = nppiMulC_8u_C4RSfs_Ctx;
		}
		else
		{
			contrast = static_cast<Npp8u>(1.0 / props.contrast);
			monoMul = nppiDivC_8u_C1RSfs_Ctx;
			bgrMul = nppiDivC_8u_C3RSfs_Ctx;
			bgraMul = nppiDivC_8u_C4RSfs_Ctx;
		}

		for (auto i = 0; i < 3; i++)
		{
			brightnessArr[i] = brightness;
			contrastArr[i] = contrast;
		}

		bufferMul.reset();
		if (contrast != 1 && brightness != 0)
		{
			bufferMul = makeBuffer(dataSize);
		}

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

	bool compute(void* buffer, void* outBuffer)
	{
		if (noEffects)
		{
			int loops = 1;
			if (frameType == FrameMetadata::RAW_IMAGE_PLANAR)
			{	
				loops = channels;
			}			

			cudaError_t cudaStatus;
			for (auto i = 0; i < loops; i++)
			{
				auto src = static_cast<uint8_t*>(buffer) + srcNextPtrOffset[i];
				auto dst = static_cast<uint8_t*>(outBuffer) + srcNextPtrOffset[i];

				cudaStatus = cudaMemcpy2DAsync(dst, srcPitch[i], src, srcPitch[i], rowSize[i], height[i], cudaMemcpyDeviceToDevice, props.stream);
				if (cudaStatus != cudaSuccess)
				{
					break;
				}
			}

			return true;
		}


		auto nextInput = buffer;
		auto nextOutput = outBuffer;
		if (contrast != 1 && brightness != 0)
		{
			nextOutput = bufferMul->data();
		}

		auto status = NPP_SUCCESS;

		if (frameType == FrameMetadata::RAW_IMAGE_PLANAR)
		{
			nextInput = buffer;
			nextOutput = outBuffer;
			launchYUV420Effects(const_cast<const Npp8u *>(static_cast<Npp8u*>(nextInput) + srcNextPtrOffset[0]),
				const_cast<const Npp8u *>(static_cast<Npp8u*>(nextInput) + srcNextPtrOffset[1]),
				const_cast<const Npp8u *>(static_cast<Npp8u*>(nextInput) + srcNextPtrOffset[2]),
				static_cast<Npp8u*>(nextOutput) + srcNextPtrOffset[0],
				static_cast<Npp8u*>(nextOutput) + srcNextPtrOffset[1],
				static_cast<Npp8u*>(nextOutput) + srcNextPtrOffset[2],
				fBrightness, fContrast, fHue, fSaturation, 
				nPitch[0], nPitch[1], srcSize[0], props.stream);

			
		}
		else if (channels == 1)
		{
			if (contrast != 1)
			{
				status = monoMul(const_cast<const Npp8u *>(static_cast<Npp8u*>(nextInput)),
					nPitch[0],
					contrast,
					static_cast<Npp8u*>(nextOutput),
					nPitch[0],
					srcSize[0],
					0,
					nppStreamCtx
				);

				if (brightness != 0)
				{
					nextInput = bufferMul->data();
					nextOutput = outBuffer;
				}
			}

			if (status != NPP_SUCCESS)
			{
				LOG_ERROR << "contrast failed<" << status << ">";
			}

			if (brightness != 0)
			{
				status = monoAdd(const_cast<const Npp8u *>(static_cast<Npp8u*>(nextInput)),
					nPitch[0],
					brightness,
					static_cast<Npp8u*>(nextOutput),
					nPitch[0],
					srcSize[0],
					0,
					nppStreamCtx
				);
			}
		}
		else if (channels == 3)
		{
			if (contrast != 1)
			{
				status = bgrMul(const_cast<const Npp8u *>(static_cast<Npp8u*>(nextInput)),
					nPitch[0],
					contrastArr,
					static_cast<Npp8u*>(nextOutput),
					nPitch[0],
					srcSize[0],
					0,
					nppStreamCtx
				);

				if (brightness != 0)
				{
					nextInput = bufferMul->data();
					nextOutput = outBuffer;
				}
			}

			if (status != NPP_SUCCESS)
			{
				LOG_ERROR << "contrast failed<" << status << ">";
			}

			if (brightness != 0)
			{
				status = bgrAdd(const_cast<const Npp8u *>(static_cast<Npp8u*>(nextInput)),
					nPitch[0],
					brightnessArr,
					static_cast<Npp8u*>(nextOutput),
					nPitch[0],
					srcSize[0],
					0,
					nppStreamCtx
				);
			}

		}
		else if (channels == 4)
		{
		if (contrast != 1)
		{
			status = bgraMul(const_cast<const Npp8u *>(static_cast<Npp8u*>(nextInput)),
				nPitch[0],
				contrastArr,
				static_cast<Npp8u*>(nextOutput),
				nPitch[0],
				srcSize[0],
				0,
				nppStreamCtx
			);

			if (brightness != 0)
			{
				nextInput = bufferMul->data();
				nextOutput = outBuffer;
			}
		}

		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "contrast failed<" << status << ">";
		}

		if (brightness != 0)
		{
			status = bgraAdd(const_cast<const Npp8u *>(static_cast<Npp8u*>(nextInput)),
				nPitch[0],
				brightnessArr,
				static_cast<Npp8u*>(nextOutput),
				nPitch[0],
				srcSize[0],
				0,
				nppStreamCtx
			);
		}
		}
		else
		{
			throw AIPException(AIP_NOTIMPLEMENTED, "resize not implemented");
		}

		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "effects failed<" << status << ">";
		}

		return true;
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

	buffer_sp bufferMul;

	std::function<buffer_sp(size_t)> makeBuffer;
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
	EffectsNPPIProps props;
	NppStreamContext nppStreamCtx;
};

EffectsNPPI::EffectsNPPI(EffectsNPPIProps _props) : Module(TRANSFORM, "EffectsNPPI", _props), props(_props), mFrameLength(0), mFrameType(FrameMetadata::GENERAL)
{
	mDetail.reset(new Detail(_props, [&](size_t size) -> buffer_sp {return makeBuffer(size, FrameMetadata::CUDA_DEVICE); }));
}

EffectsNPPI::~EffectsNPPI() {}

bool EffectsNPPI::validateInputPins()
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

bool EffectsNPPI::validateOutputPins()
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

void EffectsNPPI::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);

	setMetadata(metadata);

	mOutputMetadata->copyHint(*metadata.get());
	mOutputPinId = addOutputPin(mOutputMetadata);
}

bool EffectsNPPI::init()
{
	if (!Module::init())
	{
		return false;
	}

	auto metadata = getFirstInputMetadata();
	setMetadata(metadata);

	return true;
}

bool EffectsNPPI::term()
{
	return Module::term();
}

bool EffectsNPPI::process(frame_container &frames)
{
	auto frame = frames.cbegin()->second;
	auto outFrame = makeFrame(mFrameLength, mOutputMetadata);

	mDetail->compute(frame->data(), outFrame->data());

	frames.insert(make_pair(mOutputPinId, outFrame));
	send(frames);	

	return true;
}

bool EffectsNPPI::processSOS(frame_sp &frame)
{
	auto metadata = frame->getMetadata();
	setMetadata(metadata);
	return true;
}

void EffectsNPPI::setMetadata(framemetadata_sp& metadata)
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
	mDetail->setMetadata(metadata);
}

bool EffectsNPPI::shouldTriggerSOS()
{
	return mFrameLength == 0;
}

bool EffectsNPPI::processEOS(string& pinId)
{
	mFrameLength = 0;
	return true;
}

EffectsNPPIProps EffectsNPPI::getProps()
{
	auto props = mDetail->getProps();
	fillProps(props);

	return props;
}

void EffectsNPPI::setProps(EffectsNPPIProps& props)
{
	Module::addPropsToQueue(props);
}

bool EffectsNPPI::handlePropsChange(frame_sp& frame)
{
	auto stream = mDetail->getProps().stream_sp;
	EffectsNPPIProps props(stream);
	bool ret = Module::handlePropsChange(frame, props);

	mDetail->setProps(props);

	return ret;
}
