#include "CudaMemCopy.h"


CudaMemCopy::CudaMemCopy(CudaMemCopyProps _props) : Module(TRANSFORM, "CudaMemCopy", _props), props(_props), mOutputPinId(""), mCopy2D(false), mChannels(NOT_SET_NUM)
{		
}

CudaMemCopy::~CudaMemCopy()
{
}

bool CudaMemCopy::validateInputPins()
{
	if (getNumberOfInputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	return true;
}

bool CudaMemCopy::validateOutputPins()
{
	if (getNumberOfOutputPins() !=  1)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
		return false;
	}

	return true;
}

void CudaMemCopy::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);

	switch (props.memcpyKind)
	{			
	case cudaMemcpyDeviceToHost:
		mMemType = FrameMetadata::MemType::HOST;
		break;
	case cudaMemcpyHostToDevice:	
		mMemType = FrameMetadata::MemType::CUDA_DEVICE;
		break;
	case cudaMemcpyDeviceToDevice:
	case cudaMemcpyHostToHost:
	default:
		throw AIPException(AIP_FATAL, "Unknown copy kind<" + std::to_string(props.memcpyKind) + ">");
	}

	mFrameType = metadata->getFrameType();
	mOutputMetadata = cloneMetadata(metadata, mMemType);	
	mOutputMetadata->copyHint(*metadata.get());
	mOutputPinId = addOutputPin(mOutputMetadata);
}

bool CudaMemCopy::init()
{
	if (!Module::init())
	{
		return false;
	}	
	
	auto inputMetadata = getFirstInputMetadata();			
	if (mFrameType == FrameMetadata::RAW_IMAGE || mFrameType == FrameMetadata::RAW_IMAGE_PLANAR)
	{
		mCopy2D = true;
	}

	if(inputMetadata->isSet())
	{
		setOutputMetadata(inputMetadata);
	}	

	return true;
}

bool CudaMemCopy::term()
{		
	return Module::term();
}

bool CudaMemCopy::process(frame_container &frames)
{
	auto frame = frames.cbegin()->second;
	frame_sp outFrame;

	auto cudaStatus = cudaSuccess;

	if (mCopy2D)
	{
		outFrame = makeFrame(mOutputMetadata->getDataSize(), mOutputMetadata);		
		for (auto i = 0; i < mChannels; i++)
		{
			auto src = static_cast<uint8_t*>(frame->data()) + mSrcNextPtrOffset[i];
			auto dst = static_cast<uint8_t*>(outFrame->data()) + mDstNextPtrOffset[i];

			cudaStatus = cudaMemcpy2DAsync(dst, mDstPitch[i], src, mSrcPitch[i], mRowSize[i], mHeight[i], props.memcpyKind, props.stream);
			if (cudaStatus != cudaSuccess)
			{
				break;
			}
		}
	}
	else
	{
		auto copySize = frame->size();
		outFrame = makeFrame(copySize, mOutputMetadata);
		cudaStatus = cudaMemcpyAsync(outFrame->data(), frame->data(), copySize, props.memcpyKind, props.stream);
	}
	if (cudaStatus != cudaSuccess)
	{
		// not throwing error and not returning false - next frame will also be attempted		
		LOG_ERROR << "cudaMemcpyAsync failed <" << cudaStatus << ">";
		return true;
	}
	
	
	if(props.sync)
	{
		cudaStatus = cudaStreamSynchronize(props.stream);
		if(cudaStatus != cudaSuccess)
		{
			LOG_ERROR << "cudaStreamSynchronize failed <" << cudaStatus << ">";
		}
	}

	if (cudaStatus == cudaSuccess)
	{
		frames.insert(make_pair(mOutputPinId, outFrame));
		send(frames);
	}

	return true;
}

CudaMemCopyProps CudaMemCopy::getProps()
{	
	fillProps(props);

	return props;
}

bool CudaMemCopy::processSOS(frame_sp &frame)
{	
	auto inputMetadata = frame->getMetadata();	
	setOutputMetadata(inputMetadata);
	return true;
}

void CudaMemCopy::setOutputMetadata(framemetadata_sp& inputMetadata)
{
	if (mFrameType != inputMetadata->getFrameType())
	{
		mFrameType = inputMetadata->getFrameType();
		mOutputMetadata = cloneMetadata(inputMetadata, mMemType);
	}

	mChannels = 1;
	if (mFrameType == FrameMetadata::RAW_IMAGE)
	{
		auto rawOutMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata);
		auto rawImageMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(inputMetadata);
		RawImageMetadata other(
			rawImageMetadata->getWidth(),
			rawImageMetadata->getHeight(),
			rawImageMetadata->getImageType(),
			rawImageMetadata->getType(),
			props.alignLength,
			rawImageMetadata->getDepth(),
			FrameMetadata::MemType::HOST,
			true);
		rawOutMetadata->setData(other);

		mSrcPitch[0] = rawImageMetadata->getStep();
		mDstPitch[0] = rawOutMetadata->getStep();

		mSrcNextPtrOffset[0] = 0;
		mDstNextPtrOffset[0] = 0;
		mRowSize[0] = rawImageMetadata->getRowSize();
		mHeight[0] = rawImageMetadata->getHeight();
	}
	else if (mFrameType == FrameMetadata::RAW_IMAGE_PLANAR)
	{
		auto rawImagePlanarMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(inputMetadata);
		auto rawOutMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(mOutputMetadata);
		RawImagePlanarMetadata other(
			rawImagePlanarMetadata->getWidth(0),
			rawImagePlanarMetadata->getHeight(0),
			rawImagePlanarMetadata->getImageType(),
			props.alignLength,
			rawImagePlanarMetadata->getDepth(),
			FrameMetadata::MemType::HOST);
		rawOutMetadata->setData(other);

		mChannels = rawImagePlanarMetadata->getChannels();
		for (auto i = 0; i < mChannels; i++)
		{
			mSrcPitch[i] = rawImagePlanarMetadata->getStep(i);
			mSrcNextPtrOffset[i] = rawImagePlanarMetadata->getNextPtrOffset(i);
			mRowSize[i] = rawImagePlanarMetadata->getRowSize(i);
			mHeight[i] = rawImagePlanarMetadata->getHeight(i);
		}
		
		for (auto i = 0; i < mChannels; i++)
		{
			mDstPitch[i] = rawOutMetadata->getStep(i);
			mDstNextPtrOffset[i] = rawOutMetadata->getNextPtrOffset(i);
		}
	}
}

framemetadata_sp CudaMemCopy::cloneMetadata(framemetadata_sp metadata, FrameMetadata::MemType memType)
{
	if (memType == FrameMetadata::MemType::CUDA_DEVICE)
	{
		if (props.alignLength == 0)
		{
			props.alignLength = 512;
		}
		else if (props.alignLength % 512 != 0)
		{
			props.alignLength += FrameMetadata::getPaddingLength(props.alignLength, 512);
		}
	}

	framemetadata_sp other;
	switch (mFrameType)
	{
	case FrameMetadata::RAW_IMAGE:
	{

		other = framemetadata_sp(new RawImageMetadata(memType));
		break;
	}
	case FrameMetadata::RAW_IMAGE_PLANAR:
	{
		other = framemetadata_sp(new RawImagePlanarMetadata(memType));
		break;
	}
	case FrameMetadata::ARRAY:		
	case FrameMetadata::PROPS_CHANGE:
	case FrameMetadata::PAUSE_PLAY:
		throw AIPException(AIP_NOTIMPLEMENTED, "not supported<" + std::to_string(mFrameType) + ">");
		break;
	default:
		other = framemetadata_sp(new FrameMetadata(mFrameType, memType));
		other->setData(*metadata.get());
	}

	return other;
}


bool CudaMemCopy::shouldTriggerSOS()
{
	return mChannels == NOT_SET_NUM;
}

bool CudaMemCopy::processEOS(string& pinId)
{
	mChannels = NOT_SET_NUM;
	return true;
}