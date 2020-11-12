#include "OverlayNPPI.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "MetadataHints.h"
#include "AIPExceptions.h"

#include "npp.h"
#include "OverlayKernel.h"

// https://en.wikipedia.org/wiki/Alpha_compositing
// https://software.intel.com/en-us/node/628460
// https://docs.nvidia.com/cuda/npp/group__image__alpha__composition__operations.html


class OverlayNPPI::Detail
{
public:
	Detail(OverlayNPPIProps &_props) : props(_props), fGlobalAlpha(-1)
	{
		for (auto i = 0; i < 4; i++)
		{
			pitch[i] = 0;
			rowSize[i] = 0;
			height[i] = 0;
			offset[i] = 0;
			nSrcStep[i] = 0;
			nOverlayStep[i] = 0;
			nDstStep[i] = 0;
			oSizeROI[i] = { 0, 0 };
		}

		nppStreamCtx.hStream = _props.stream;
	}

	~Detail()
	{

	}

	void setProps(OverlayNPPIProps &_props)
	{
		props = _props;
		if (props.offsetX < 0 || props.offsetY < 0)
		{
			throw AIPException(AIP_PARAM_OUTOFRANGE, "Offset should be a positive number");
		}

		doCopy = false;
		if (props.offsetX != 0 || props.offsetY != 0 || width[0] != oSizeROI[0].width || height[0] != oSizeROI[0].height)
		{
			doCopy = true;
		}
			   
		if (props.globalAlpha == 0)
		{
			doCopy = true;
		}

		if (props.globalAlpha >= 0)
		{
			fGlobalAlpha = static_cast<float>(props.globalAlpha / 255.0);
		}
		else
		{
			fGlobalAlpha = -1;
		}

		if (nOverlayStep != 0 &&
			(
			(props.offsetX + oSizeROI[0].width) > width[0]
				|| (props.offsetY + oSizeROI[0].height) > height[0]
				)
			)
		{
			throw AIPException(AIP_PARAM_OUTOFRANGE, "ROI out of range");
		}

		if (frameType == FrameMetadata::RAW_IMAGE)
		{
			auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
			offset[0] = rawMetadata->getOffset(props.offsetX, props.offsetY);
		}
		else
		{
			auto rawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
			for (auto i = 0; i < channels; i++)
			{
				offset[i] = rawMetadata->getOffset(i, props.offsetX, props.offsetY);
			}
		}

		preMulGlobalAlpha();
	}

	bool setMetadata(framemetadata_sp& input)
	{
		frameType = input->getFrameType();
		channels = 1;
		if (frameType == FrameMetadata::RAW_IMAGE)
		{
			auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(input);
			// both input and output are of same size type 

			nSrcStep[0] = static_cast<int>(rawMetadata->getStep());
			nDstStep[0] = nSrcStep[0];

			pitch[0] = rawMetadata->getStep();
			rowSize[0] = rawMetadata->getRowSize();
			width[0] = rawMetadata->getWidth();
			height[0] = rawMetadata->getHeight();
			nextPtrOffset[0] = 0;
		}
		else
		{
			auto rawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(input);
			if (rawMetadata->getImageType() != ImageMetadata::YUV420)
			{
				throw AIPException(AIP_FATAL, "PLANAR - ONLY YUV420 SUPPORTED");
			}
			// both input and output are of same size type 

			channels = rawMetadata->getChannels();

			for (auto i = 0; i < channels; i++)
			{
				nSrcStep[i] = static_cast<int>(rawMetadata->getStep(i));
				nDstStep[i] = nSrcStep[i];

				pitch[i] = rawMetadata->getStep(i);
				rowSize[i] = rawMetadata->getRowSize(i);
				width[i] = rawMetadata->getWidth(i);
				height[i] = rawMetadata->getHeight(i);
				nextPtrOffset[i] = rawMetadata->getNextPtrOffset(i);
			}
		}

		metadata = input;

		setProps(props);

		return true;
	}

	bool setOverlayMetadata(framemetadata_sp& metadata, frame_sp& frame)
	{
		overlayTempFrame = frame; // for premultiplying global alpha
		frameType = metadata->getFrameType();
		channels = 1;

		if (frameType == FrameMetadata::RAW_IMAGE)
		{
			auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
			nOverlayStep[0] = static_cast<int>(rawMetadata->getStep());
			oSizeROI[0] = { rawMetadata->getWidth(), rawMetadata->getHeight() };
		}
		else
		{
			auto rawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
			channels = rawMetadata->getChannels();

			for (auto i = 0; i < channels; i++)
			{
				nOverlayStep[i] = static_cast<int>(rawMetadata->getStep(i));
				oSizeROI[i] = { rawMetadata->getWidth(i), rawMetadata->getHeight(i) };
				overlayNextPtrOffset[i] = rawMetadata->getNextPtrOffset(i);
			}
		}

		doCopy = false;
		if (props.offsetX != 0 || props.offsetY != 0 || width[0] != oSizeROI[0].width || height[0] != oSizeROI[0].height)
		{
			doCopy = true;
		}

		if (props.globalAlpha == 0)
		{
			doCopy = true;
		}

		return true;
	}

	bool compute(void* buffer, void* outBuffer)
	{
		if (doCopy || !nOverlayStep)
		{
			cudaError_t cudaStatus;

			for (auto i = 0; i < channels; i++)
			{
				auto src = static_cast<uint8_t*>(buffer) + nextPtrOffset[i];
				auto dst = static_cast<uint8_t*>(outBuffer) + nextPtrOffset[i];

				cudaStatus = cudaMemcpy2DAsync(dst, pitch[i], src, pitch[i], rowSize[i], height[i], cudaMemcpyDeviceToDevice, props.stream);
				if (cudaStatus != cudaSuccess)
				{
					break;
				}
			}
			if (cudaStatus != cudaSuccess)
			{
				LOG_ERROR << "copy failed<" << cudaStatus << ">";
				return false;
			}
		}

		
		if (!nOverlayStep || props.globalAlpha == 0)
		{
			// overlay not ready or global Alpha is set to zero - so no composition
			return true;
		}

		NppStatus status = NPP_SUCCESS;

		if (frameType == FrameMetadata::RAW_IMAGE)
		{
			status = nppiAlphaComp_8u_AC4R_Ctx(
				const_cast<const Npp8u *>(static_cast<Npp8u *>(overlayFrame->data())),
				nOverlayStep[0],
				const_cast<const Npp8u *>(static_cast<Npp8u *>(buffer) + offset[0]),
				nSrcStep[0],
				static_cast<Npp8u *>(outBuffer) + offset[0],
				nDstStep[0],
				oSizeROI[0],
				NPPI_OP_ALPHA_OVER_PREMUL,
				nppStreamCtx);
		}
		else
		{		
			const Npp8u* src_[3] = { static_cast<Npp8u *>(buffer) + nextPtrOffset[0] + offset[0],
									static_cast<Npp8u *>(buffer) + nextPtrOffset[1] + offset[1],
									static_cast<Npp8u *>(buffer) + nextPtrOffset[2] + offset[2] };
			const Npp8u* overlay_[3] = { static_cast<Npp8u *>(overlayFrame->data()) + overlayNextPtrOffset[0],
									static_cast<Npp8u *>(overlayFrame->data()) + overlayNextPtrOffset[1],
									static_cast<Npp8u *>(overlayFrame->data()) + overlayNextPtrOffset[2] };
			Npp8u* dst_[3] = { static_cast<Npp8u *>(outBuffer) + nextPtrOffset[0] + offset[0],
									static_cast<Npp8u *>(outBuffer) + nextPtrOffset[1] + offset[1],
									static_cast<Npp8u *>(outBuffer) + nextPtrOffset[2] + offset[2] };
			launchYUVOverlayKernel(src_, overlay_, dst_, fGlobalAlpha, nSrcStep, nOverlayStep, oSizeROI[0], nppStreamCtx.hStream);
		}

		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "Alpha Composition Failed. <" << status << ">";

			return false;
		}

		return true;
	}

	void setOverlayFrame(frame_sp& frame)
	{
		overlayOriginalFrame = frame;
		overlayFrame = frame;
		preMulGlobalAlpha();		
	}

private:
	
	void preMulGlobalAlpha()
	{		
		if (!overlayOriginalFrame.get())
		{
			// overlay is not set yet
			return;
		}

		if (frameType == FrameMetadata::RAW_IMAGE)
		{
			auto status = nppiAlphaPremul_8u_AC4IR_Ctx(static_cast<Npp8u *>(overlayOriginalFrame->data()),
				nOverlayStep[0],				
				oSizeROI[0],
				nppStreamCtx
			);

			if (status != NPP_SUCCESS)
			{
				LOG_ERROR << "Alpha nppiAlphaPremul_8u_AC4IR_Ctx Failed. <" << status << ">";
			}

			if (props.globalAlpha != 0 && props.globalAlpha != -1)
			{
				// globalAlpha = 0 no need of compositing
				// globalAlpha = -1 - global alpha not enabled

				status = nppiAlphaPremulC_8u_C4R_Ctx(static_cast<Npp8u *>(overlayOriginalFrame->data()),
					nOverlayStep[0],
					props.globalAlpha,
					static_cast<Npp8u *>(overlayTempFrame->data()),
					nOverlayStep[0],
					oSizeROI[0],
					nppStreamCtx
				);

				if (status != NPP_SUCCESS)
				{
					LOG_ERROR << "Alpha nppiAlphaPremulC_8u_C4R_Ctx Failed. <" << status << ">";
				}

				overlayFrame = overlayTempFrame;
			}
			else
			{
				overlayFrame = overlayOriginalFrame;
			}
		}
		else
		{
			overlayFrame = overlayOriginalFrame;
		}
	}

private:
	FrameMetadata::FrameType frameType;
	frame_sp overlayOriginalFrame;
	frame_sp overlayFrame;
	frame_sp overlayTempFrame; // global Alpha is computed

	bool doCopy;

	// copy params
	size_t pitch[4];
	size_t rowSize[4];
	size_t width[4];
	size_t height[4];
	size_t nextPtrOffset[4];
	int channels;

	size_t offset[4];
	int nSrcStep[4];
	int nOverlayStep[4];
	size_t overlayNextPtrOffset[4];
	int nDstStep[4];
	NppiSize oSizeROI[4];

	float fGlobalAlpha;

	framemetadata_sp metadata;
	OverlayNPPIProps props;
	NppStreamContext nppStreamCtx;
};

OverlayNPPI::OverlayNPPI(OverlayNPPIProps _props) : Module(TRANSFORM, "OverlayNPPI", _props), props(_props), mFrameLength(0), mFrameType(FrameMetadata::GENERAL)
{
	mDetail.reset(new Detail(_props));
}

OverlayNPPI::~OverlayNPPI() {}

bool OverlayNPPI::validateInputOutputPins()
{
	if (getNumberOfInputPins() != 2)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 2. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	auto inputPinIdMetadataMap = getInputMetadata();
	for (auto const &element : inputPinIdMetadataMap)
	{
		auto& metadata = element.second;
		auto frameType = metadata->getFrameType();
		if (mFrameType != frameType)
		{
			LOG_ERROR << "<" << getId() << ">::validateInputPins both the input pins should be of same type. Expected<" << mFrameType << "> Actual<" << mFrameType << ">";
			return false;
		}

		FrameMetadata::MemType memType = metadata->getMemType();
		if (memType != FrameMetadata::MemType::CUDA_DEVICE)
		{
			LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is expected to be CUDA_DEVICE. Actual<" << memType << ">";
			return false;
		}
	}

	return true;
}

bool OverlayNPPI::validateInputPins()
{
	if (getNumberOfInputPins() > 2)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 2. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	auto inputPinIdMetadataMap = getInputMetadata();
	for (auto const &element : inputPinIdMetadataMap)
	{
		auto& metadata = element.second;
		mFrameType = metadata->getFrameType();
		if (mFrameType != FrameMetadata::RAW_IMAGE && mFrameType != FrameMetadata::RAW_IMAGE_PLANAR)
		{
			LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be RAW_IMAGE or RAW_IMAGE_PLANAR. Actual<" << mFrameType << ">";
			return false;
		}

		FrameMetadata::MemType memType = metadata->getMemType();
		if (memType != FrameMetadata::MemType::CUDA_DEVICE)
		{
			LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is expected to be CUDA_DEVICE. Actual<" << memType << ">";
			return false;
		}
	}

	return true;
}

bool OverlayNPPI::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstOutputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != mFrameType)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins output frameType is expected to be <" << mFrameType << ">. Actual<" << frameType << ">";
		return false;
	}

	FrameMetadata::MemType memType = metadata->getMemType();
	if (memType != FrameMetadata::MemType::CUDA_DEVICE)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins output memType is expected to be CUDA_DEVICE. Actual<" << memType << ">";
		return false;
	}

	return true;
}

void OverlayNPPI::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);

	if (metadata->getHint() != OVERLAY_HINT)
	{
		if (mFrameType == FrameMetadata::RAW_IMAGE)
		{
			mOutputMetadata = framemetadata_sp(new RawImageMetadata(FrameMetadata::MemType::CUDA_DEVICE));
		}
		if (mFrameType == FrameMetadata::RAW_IMAGE_PLANAR)
		{
			mOutputMetadata = framemetadata_sp(new RawImagePlanarMetadata(FrameMetadata::MemType::CUDA_DEVICE));
		}
		mOutputPinId = addOutputPin(mOutputMetadata);
	}
}

bool OverlayNPPI::init()
{
	if (!Module::init())
	{
		return false;
	}

	auto inputPinIdMetadataMap = getInputMetadata();
	for (auto const &element : inputPinIdMetadataMap)
	{
		auto metadata = element.second;
		if (metadata->isSet())
		{
			if (metadata->getHint() != OVERLAY_HINT)
			{
				setMetadata(metadata);
			}
			else
			{
				auto frame = makeFrame(metadata->getDataSize(), metadata);
				mDetail->setOverlayMetadata(metadata, frame);
			}
		}
	}

	return true;
}

bool OverlayNPPI::term()
{
	return Module::term();
}

bool OverlayNPPI::process(frame_container &frames)
{
	frame_sp frame;
	for (auto const& element : frames)
	{
		auto tempFrame = element.second;
		if (tempFrame->getMetadata()->getHint() == OVERLAY_HINT)
		{
			// overlay frame is cached
			mDetail->setOverlayFrame(tempFrame);
		}
		else
		{
			frame = tempFrame;
		}
	}

	if (isFrameEmpty(frame))
	{
		return true;
	}

	auto outFrame = makeFrame(mFrameLength, mOutputMetadata);

	if (!mDetail->compute(frame->data(), outFrame->data()))
	{
		return true;
	}

	frames.insert(make_pair(mOutputPinId, outFrame));
	send(frames);

	return true;
}

bool OverlayNPPI::processSOS(frame_sp &frame)
{
	auto metadata = frame->getMetadata();
	if (metadata->getFrameType() != mFrameType)
	{
		throw AIPException(AIP_FATAL, "FrameType change not supported");
	}

	if (metadata->getHint() != OVERLAY_HINT)
	{
		setMetadata(metadata);
	}
	else
	{
		auto frame = makeFrame(metadata->getDataSize(), metadata);
		mDetail->setOverlayMetadata(metadata, frame);
	}
	return true;
}

void OverlayNPPI::setMetadata(framemetadata_sp& metadata)
{
	if (mFrameType == FrameMetadata::RAW_IMAGE)
	{
		auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
		auto rawOutMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata);
		rawOutMetadata->setData(*rawMetadata);
	}
	else
	{
		auto rawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
		auto rawOutMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(mOutputMetadata);
		rawOutMetadata->setData(*rawMetadata);
	}

	mFrameLength = mOutputMetadata->getDataSize();
	mDetail->setMetadata(metadata);
}

bool OverlayNPPI::shouldTriggerSOS()
{
	return mFrameLength == 0;
}

bool OverlayNPPI::processEOS(string& pinId)
{
	mFrameLength = 0;
	return true;
}

OverlayNPPIProps OverlayNPPI::getProps()
{
	fillProps(props);

	return props;
}

void OverlayNPPI::setProps(OverlayNPPIProps& props)
{
	Module::addPropsToQueue(props);
}

bool OverlayNPPI::handlePropsChange(frame_sp& frame)
{
	bool ret = Module::handlePropsChange(frame, props);

	mDetail->setProps(props);

	return ret;
}
