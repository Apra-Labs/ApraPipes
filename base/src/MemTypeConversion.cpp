#include "MemTypeConversion.h"
#include "Logger.h"
#include "RawImageMetadata.h"
#include "RawImagePlanarMetadata.h"
#include "CudaCommon.h"

#if defined(__arm__) || defined(__aarch64__)
#include "DMAFrameUtils.h"
#include "DMAFDWrapper.h"
#include "ImagePlaneData.h"
#include "DMAAllocator.h"
#endif

class DetailMemory
{
public:
	DetailMemory(MemTypeConversionProps &_props) : mOutputPinId(""),
												   mFrameType(FrameMetadata::FrameType::RAW_IMAGE),
												   mSize(0), mNumPlanes(0), props(_props)
	{
	}

	~DetailMemory() {}

	virtual bool compute() = 0;

	bool setMetadataHelper(framemetadata_sp &input, framemetadata_sp &output, frame_sp &frame)
	{
		auto inputMetadata = frame->getMetadata();
		FrameMetadata::MemType inputMemType = inputMetadata->getMemType();
		if (inputMemType == FrameMetadata::MemType::DMABUF && props.outputMemType == FrameMetadata::MemType::HOST)
		{
#if defined(__arm__) || defined(__aarch64__)
			mGetImagePlanes = DMAFrameUtils::getImagePlanesFunction(inputMetadata, mImagePlanes);
			mNumPlanes = static_cast<int>(mImagePlanes.size());
#endif
		}
		else if (inputMemType == FrameMetadata::MemType::HOST && props.outputMemType == FrameMetadata::MemType::DMABUF)
		{
#if defined(__arm__) || defined(__aarch64__)
			mGetImagePlanes = DMAFrameUtils::getImagePlanesFunction(inputMetadata, mImagePlanes);
			mNumPlanes = static_cast<int>(mImagePlanes.size());
#endif
		}
		else
		{
			if (mFrameType == FrameMetadata::RAW_IMAGE)
			{
				auto inputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(input);
				auto outputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(output);
				imageType = inputRawMetadata->getImageType();
				size_t pitch[4] = {0, 0, 0, 0};
				imageChannels = 1;
				srcPitch[0] = inputRawMetadata->getStep();
				dstPitch[0] = outputRawMetadata->getStep();
				srcNextPtrOffset[0] = 0;
				dstNextPtrOffset[0] = 0;
				rowSize[0] = inputRawMetadata->getRowSize();
				height[0] = inputRawMetadata->getHeight();
				width[0] = inputRawMetadata->getWidth();

				if (inputMemType == FrameMetadata::MemType::DMABUF || props.outputMemType == FrameMetadata::MemType::DMABUF)
				{
#if defined(__arm__) || defined(__aarch64__)
					int type = CV_8UC4;
					switch (imageType)
					{
					case ImageMetadata::ImageType::RGBA:
					case ImageMetadata::ImageType::BGRA:
						type = CV_8UC4;
						break;
					case ImageMetadata::ImageType::UYVY:
					case ImageMetadata::ImageType::YUYV:
						type = CV_8UC3;
						break;
					default:
						throw AIPException(AIP_FATAL, "Only Image Type accepted are UYVY or ARGB found " + std::to_string(imageType));
					}
					auto metadata = framemetadata_sp(new RawImageMetadata(width[0], height[0], imageType, type, size_t(0), CV_8U, FrameMetadata::MemType::DMABUF, true));
					DMAAllocator::setMetadata(metadata, width[0], height[0], imageType, pitch);
					if (inputMemType == FrameMetadata::MemType::DMABUF)
					{
						srcPitch[0] = pitch[0];
					}

					else if (props.outputMemType == FrameMetadata::MemType::DMABUF)
					{
						dstPitch[0] = pitch[0];
					}
#endif
				}
			}
			else if (mFrameType == FrameMetadata::RAW_IMAGE_PLANAR)
			{
				auto inputRawPlanarMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(input);
				auto outputRawPlanarMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(output);

				imageType = inputRawPlanarMetadata->getImageType();
				size_t pitch[4] = {0, 0, 0, 0};
				size_t offset[4] = {0, 0, 0, 0};
				imageChannels = inputRawPlanarMetadata->getChannels();
				for (auto i = 0; i < imageChannels; i++)
				{
					srcPitch[i] = inputRawPlanarMetadata->getStep(i);
					srcNextPtrOffset[i] = inputRawPlanarMetadata->getNextPtrOffset(i);
					rowSize[i] = inputRawPlanarMetadata->getRowSize(i);
					height[i] = inputRawPlanarMetadata->getHeight(i);

					dstPitch[i] = outputRawPlanarMetadata->getStep(i);
					dstNextPtrOffset[i] = outputRawPlanarMetadata->getNextPtrOffset(i);
				}

				if (inputMemType == FrameMetadata::MemType::DMABUF || props.outputMemType == FrameMetadata::MemType::DMABUF)
				{
#if defined(__arm__) || defined(__aarch64__)
					auto metadata = framemetadata_sp(new RawImagePlanarMetadata(rowSize[0], height[0], imageType, size_t(0), CV_8U, FrameMetadata::MemType::DMABUF));
					DMAAllocator::setMetadata(metadata, rowSize[0], height[0], imageType, pitch, offset);
					if (inputMemType == FrameMetadata::MemType::DMABUF)
					{
						for (int i = 0; i < imageChannels; i++)
						{
							srcNextPtrOffset[i] = offset[i];
							srcPitch[i] = pitch[i];
						}
					}

					else if (props.outputMemType == FrameMetadata::MemType::DMABUF)
					{
						for (int i = 0; i < imageChannels; i++)
						{
							dstNextPtrOffset[i] = offset[i];
							dstPitch[i] = pitch[i];
						}
					}
#endif
				}
			}
		}
		return true;
	}

public:
	framemetadata_sp mOutputMetadata;
	FrameMetadata::FrameType mFrameType;
	ImageMetadata::ImageType imageType;
	frame_sp inputFrame;
	frame_sp outputFrame;
	MemTypeConversionProps props;
	std::string mOutputPinId;
	size_t mAlignLength = 0;
	size_t mSize;

protected:
	int mNumPlanes;
	void *srcPtr;
	void *dstPtr;
	bool sync = true;
	int imageChannels = 0;
	size_t srcPitch[4];
	size_t dstPitch[4];
	size_t srcNextPtrOffset[4];
	size_t dstNextPtrOffset[4];
	size_t rowSize[4];
	size_t height[4];
	size_t width[4];

#if defined(__arm__) || defined(__aarch64__)
	ImagePlanes mImagePlanes;
	DMAFrameUtils::GetImagePlanes mGetImagePlanes;
#endif
};

class DetailDMAtoHOST : public DetailMemory
{
public:
	DetailDMAtoHOST(MemTypeConversionProps &_props) : DetailMemory(_props) {}

	bool compute()
	{
#if defined(__arm__) || defined(__aarch64__)
		mGetImagePlanes(inputFrame, mImagePlanes);
		dstPtr = static_cast<uint8_t *>(outputFrame->data());

		for (auto i = 0; i < mNumPlanes; i++)
		{
			mImagePlanes[i]->mCopyToData(mImagePlanes[i].get(), dstPtr);
			dstPtr += mImagePlanes[i]->imageSize;
		}
#endif
		return true;
	}
};

class DetailHOSTtoDMA : public DetailMemory
{
public:
	DetailHOSTtoDMA(MemTypeConversionProps &_props) : DetailMemory(_props) {}

	bool compute()
	{
#if defined(__arm__) || defined(__aarch64__)
		mGetImagePlanes(inputFrame, mImagePlanes);
		dstPtr = (static_cast<DMAFDWrapper *>(outputFrame->data()))->getHostPtr();

		for (auto i = 0; i < mNumPlanes; i++)
		{
			mImagePlanes[i]->mCopyToData(mImagePlanes[i].get(), dstPtr);
			dstPtr += mImagePlanes[i]->imageSize;
		}
#endif
		return true;
	}
};

class DetailDEVICEtoDMA : public DetailMemory
{
public:
	DetailDEVICEtoDMA(MemTypeConversionProps &_props) : DetailMemory(_props) {}

	bool compute()
	{
#if defined(__arm__) || defined(__aarch64__)
		auto cudaStatus = cudaSuccess;

		for (auto i = 0; i < imageChannels; i++)
		{
			srcPtr = static_cast<uint8_t *>(inputFrame->data()) + srcNextPtrOffset[i];
			dstPtr = (static_cast<DMAFDWrapper *>(outputFrame->data())->getCudaPtr() + dstNextPtrOffset[i]);

			cudaStatus = cudaMemcpy2DAsync(dstPtr, dstPitch[i], srcPtr, srcPitch[i], rowSize[i], height[i], cudaMemcpyDeviceToDevice, props.stream);
			if (cudaStatus != cudaSuccess)
			{
				LOG_ERROR << "cudaMemcpy2DAsync failed <" << cudaStatus << "> Kind :" << props.outputMemType
						  << " from " << (uint64_t)srcPtr << " to " << (uint64_t)dstPtr << " for " << dstPitch[i] << "," << srcPitch[i] << "," << rowSize[i] << " x " << height[i];
				return true;
			}
		}
#endif
		return true;
	}
};

class DetailDMAtoDEVICE : public DetailMemory
{
public:
	DetailDMAtoDEVICE(MemTypeConversionProps &_props) : DetailMemory(_props) {}

	bool compute()
	{
#if defined(__arm__) || defined(__aarch64__)
		auto cudaStatus = cudaSuccess;

		for (auto i = 0; i < imageChannels; i++)
		{
			srcPtr = (static_cast<DMAFDWrapper *>(inputFrame->data())->getCudaPtr()) + srcNextPtrOffset[i];
			dstPtr = (static_cast<uint8_t *>(outputFrame->data())) + dstNextPtrOffset[i];

			cudaStatus = cudaMemcpy2DAsync(dstPtr, dstPitch[i], srcPtr, srcPitch[i], rowSize[i], height[i], cudaMemcpyDeviceToDevice, props.stream);
			if (cudaStatus != cudaSuccess)
			{
				LOG_ERROR << "cudaMemcpy2DAsync failed <" << cudaStatus << "> Kind :" << props.outputMemType
						  << " from " << (uint64_t)srcPtr << " to " << (uint64_t)dstPtr << " for " << dstPitch[i] << "," << srcPitch[i] << "," << rowSize[i] << " x " << height[i];
				return true;
			}
		}
#endif
		return true;
	}
};

class DetailDEVICEtoHOST : public DetailMemory
{
public:
	DetailDEVICEtoHOST(MemTypeConversionProps &_props) : DetailMemory(_props) {}

	bool compute()
	{
		auto cudaStatus = cudaSuccess;

		for (auto i = 0; i < imageChannels; i++)
		{
			srcPtr = static_cast<uint8_t *>(inputFrame->data()) + srcNextPtrOffset[i];
			dstPtr = static_cast<uint8_t *>(outputFrame->data()) + dstNextPtrOffset[i];

			cudaStatus = cudaMemcpy2DAsync(dstPtr, dstPitch[i], srcPtr, srcPitch[i], rowSize[i], height[i], cudaMemcpyDeviceToHost, props.stream);
			if (cudaStatus != cudaSuccess)
			{
				LOG_ERROR << "cudaMemcpy2DAsync failed <" << cudaStatus << "> Kind :" << props.outputMemType
						  << " from " << (uint64_t)srcPtr << " to " << (uint64_t)dstPtr << " for " << dstPitch[i] << "," << srcPitch[i] << "," << rowSize[i] << " x " << height[i];
				return true;
			}
		}

		if (sync)
		{
			cudaStatus = cudaStreamSynchronize(props.stream);
			if (cudaStatus != cudaSuccess)
			{
				LOG_ERROR << "cudaStreamSynchronize failed <" << cudaStatus << ">";
			}
		}
		return true;
	}
};

class DetailHOSTtoDEVICE : public DetailMemory
{
public:
	DetailHOSTtoDEVICE(MemTypeConversionProps &_props) : DetailMemory(_props) {}

	bool compute()
	{
		auto cudaStatus = cudaSuccess;

		for (auto i = 0; i < imageChannels; i++)
		{
			srcPtr = static_cast<uint8_t *>(inputFrame->data()) + srcNextPtrOffset[i];
			dstPtr = static_cast<uint8_t *>(outputFrame->data()) + dstNextPtrOffset[i];

			cudaStatus = cudaMemcpy2DAsync(dstPtr, dstPitch[i], srcPtr, srcPitch[i], rowSize[i], height[i], cudaMemcpyHostToDevice, props.stream);
			if (cudaStatus != cudaSuccess)
			{
				LOG_ERROR << "cudaMemcpy2DAsync failed <" << cudaStatus << "> Kind :" << props.outputMemType
						  << " from " << (uint64_t)srcPtr << " to " << (uint64_t)dstPtr << " for " << dstPitch[i] << "," << srcPitch[i] << "," << rowSize[i] << " x " << height[i];
				return true;
			}
		}

		return true;
	}
};

MemTypeConversion::MemTypeConversion(MemTypeConversionProps _props) : Module(TRANSFORM, "MemTypeConversion", _props), mProps(_props) {}

MemTypeConversion::~MemTypeConversion() {}

bool MemTypeConversion::validateInputPins()
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

	return true;
}

bool MemTypeConversion::validateOutputPins()
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
	return true;
}

void MemTypeConversion::addInputPin(framemetadata_sp &metadata, string &pinId)
{
	Module::addInputPin(metadata, pinId);
	FrameMetadata::MemType inputMemType = metadata->getMemType();
	if (inputMemType != FrameMetadata::MemType::CUDA_DEVICE && inputMemType != FrameMetadata::MemType::DMABUF && inputMemType != FrameMetadata::MemType::HOST)
	{
		throw AIPException(AIP_FATAL, "Input memType is expected to be CUDA_DEVICE or DMABUF or HOST. Actual<" + std::to_string(metadata->getMemType()) + ">");
	}

	if (inputMemType == FrameMetadata::MemType::HOST && mProps.outputMemType == FrameMetadata::MemType::DMABUF)
	{
		mDetail.reset(new DetailHOSTtoDMA(mProps));
	}

	else if (inputMemType == FrameMetadata::MemType::DMABUF && mProps.outputMemType == FrameMetadata::MemType::HOST)
	{
		mDetail.reset(new DetailDMAtoHOST(mProps));
	}

	else if (inputMemType == FrameMetadata::MemType::CUDA_DEVICE && mProps.outputMemType == FrameMetadata::MemType::DMABUF)
	{
		mDetail.reset(new DetailDEVICEtoDMA(mProps));
	}

	else if (inputMemType == FrameMetadata::MemType::DMABUF && mProps.outputMemType == FrameMetadata::MemType::CUDA_DEVICE)
	{
		mDetail.reset(new DetailDMAtoDEVICE(mProps));
	}

	else if (inputMemType == FrameMetadata::MemType::HOST && mProps.outputMemType == FrameMetadata::MemType::CUDA_DEVICE)
	{
		mDetail.reset(new DetailHOSTtoDEVICE(mProps));
		if (mDetail->mAlignLength == 0)
		{
			mDetail->mAlignLength = 512;
		}
		else if ((mDetail->mAlignLength) % 512 != 0)
		{
			mDetail->mAlignLength += FrameMetadata::getPaddingLength(mDetail->mAlignLength, 512);
		}
	}

	else if (inputMemType == FrameMetadata::MemType::CUDA_DEVICE && mProps.outputMemType == FrameMetadata::MemType::HOST)
	{
		mDetail.reset(new DetailDEVICEtoHOST(mProps));
	}

	else
	{
		throw std::runtime_error("conversion not supported");
	}

	mDetail->mFrameType = metadata->getFrameType();
	switch (mDetail->mFrameType)
	{
	case FrameMetadata::FrameType::RAW_IMAGE:
		mDetail->mOutputMetadata = framemetadata_sp(new RawImageMetadata(mProps.outputMemType));
		break;
	case FrameMetadata::FrameType::RAW_IMAGE_PLANAR:
		mDetail->mOutputMetadata = framemetadata_sp(new RawImagePlanarMetadata(mProps.outputMemType));
		break;
	default:
		throw AIPException(AIP_FATAL, "Expected Raw Image or RAW_IMAGE_PLANAR. Actual<" + std::to_string(mDetail->mFrameType) + ">");
	}

	mDetail->mOutputMetadata->copyHint(*metadata.get());
	mDetail->mOutputPinId = addOutputPin(mDetail->mOutputMetadata);
}

bool MemTypeConversion::init()
{
	if (!Module::init())
	{
		return false;
	}
	return true;
}

bool MemTypeConversion::term()
{
	return Module::term();
}

bool MemTypeConversion::process(frame_container &frames)
{
	mDetail->inputFrame = frames.cbegin()->second;
	mDetail->outputFrame = makeFrame(mDetail->mSize, mDetail->mOutputPinId);
	mDetail->compute();

	frames.insert({mDetail->mOutputPinId, mDetail->outputFrame});
	send(frames);

	return true;
}

bool MemTypeConversion::processSOS(frame_sp &frame)
{
	auto inputMetadata = frame->getMetadata();
	FrameMetadata::MemType mInputMemType = inputMetadata->getMemType();
	if (inputMetadata->getFrameType() != mDetail->mFrameType)
	{
		throw AIPException(AIP_FATAL, "FrameType changed");
	}

	switch (mDetail->mFrameType)
	{
	case FrameMetadata::FrameType::RAW_IMAGE:
	{
		auto inputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(inputMetadata);
		auto outputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mDetail->mOutputMetadata);

		mDetail->imageType = inputRawMetadata->getImageType();
		RawImageMetadata rawMetadata(inputRawMetadata->getWidth(), inputRawMetadata->getHeight(), mDetail->imageType, inputRawMetadata->getType(), mDetail->mAlignLength, inputRawMetadata->getDepth(), mInputMemType, true);
		outputRawMetadata->setData(rawMetadata);
	}
	break;
	case FrameMetadata::FrameType::RAW_IMAGE_PLANAR:
	{
		if (mInputMemType == FrameMetadata::MemType::HOST && mProps.outputMemType == FrameMetadata::MemType::DMABUF)
		{
			throw AIPException(AIP_FATAL, "Not yet Implemented for Planar Images");
		}
		auto inputRawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(inputMetadata);
		auto outputRawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(mDetail->mOutputMetadata);

		mDetail->imageType = inputRawMetadata->getImageType();
		RawImagePlanarMetadata rawMetadata(inputRawMetadata->getWidth(0), inputRawMetadata->getHeight(0), mDetail->imageType, mDetail->mAlignLength, inputRawMetadata->getDepth(), mInputMemType);
		outputRawMetadata->setData(rawMetadata);
	}
	break;
	default:
		break;
	}

	if (mInputMemType == FrameMetadata::MemType::DMABUF || mProps.outputMemType == FrameMetadata::MemType::DMABUF)
	{
		switch (mDetail->imageType)
		{
		case ImageMetadata::ImageType::RGBA:
		case ImageMetadata::ImageType::BGRA:
		case ImageMetadata::ImageType::YUYV:
		case ImageMetadata::ImageType::UYVY:
		case ImageMetadata::ImageType::YUV420:
		case ImageMetadata::ImageType::NV12:
			break;
		default:
			throw AIPException(AIP_FATAL, "Expected <RGBA/BGRA/UYVY/YUV420/NV12/YUYV> Actual<" + std::to_string(mDetail->imageType) + ">");
		}
	}

	mDetail->mSize = mDetail->mOutputMetadata->getDataSize();
	mDetail->setMetadataHelper(inputMetadata, mDetail->mOutputMetadata, frame);
	return true;
}

bool MemTypeConversion::processEOS(string &pinId)
{
	mDetail->mOutputMetadata->reset();
	return true;
}