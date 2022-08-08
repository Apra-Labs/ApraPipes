#include "RandomDMA.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "AIPExceptions.h"
#include "DMAFDWrapper.h"
#include "DMAUtils.h"
#include "nvbuf_utils.h"
// #include "EGL/egl.h"
#include "cudaEGL.h"
#include <Argus/Argus.h>
#include <deque>
// #include "launchDivision.h"

#include "npp.h"

class RandomDMA::Detail
{
public:
	Detail(RandomDMAProps &_props) : props(_props)
	{
        // eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        // if(eglDisplay == EGL_NO_DISPLAY)
        // {
        //     throw AIPException(AIP_FATAL, "eglGetDisplay failed");
        // } 

        // if (!eglInitialize(eglDisplay, NULL, NULL))
        // {
        // throw AIPException(AIP_FATAL, "eglInitialize failed");
        // }
	}

	~Detail()
	{

	}

	bool setMetadata(framemetadata_sp& input, framemetadata_sp& output)
	{
		auto inputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(input);
		inputImageType = inputRawMetadata->getImageType();
		inputChannels = inputRawMetadata->getChannels();
		srcSize[0] = {inputRawMetadata->getWidth(), inputRawMetadata->getHeight()};
		srcPitch[0] = static_cast<int>(inputRawMetadata->getStep());
		auto outputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(output);
		outputImageType = outputRawMetadata->getImageType();
		outputChannels = outputRawMetadata->getChannels();
		dstPitch = static_cast<int>(outputRawMetadata->getStep());
        rowSize = inputRawMetadata->getRowSize();
		height = inputRawMetadata->getHeight();
        dstNextPtrOffset[0] = 0;
		return true;
	}

    bool compute(DMAFDWrapper *frame1DMAFdWrapper, DMAFDWrapper *frame2DMAFdWrapper, DMAFDWrapper *frame3DMAFdWrapper, DMAFDWrapper *outDMAFdWrapper)
    {
        auto frame1Buffer = frame1DMAFdWrapper->getCudaPtr();
        auto frame2Buffer = frame1DMAFdWrapper->getCudaPtr();
        auto frame3Buffer = frame1DMAFdWrapper->getCudaPtr();
    
        auto dstPtr = outDMAFdWrapper->getCudaPtr();

        cudaMemcpy2DAsync(dstPtr, dstPitch, frame1Buffer, srcPitch[0], rowSize, height, cudaMemcpyDeviceToDevice, props.stream);
        
        auto dstPtr1 = static_cast<uint8_t *>(dstPtr) + ( rowSize * 4 );
		cudaMemcpy2DAsync(dstPtr1, dstPitch, frame2Buffer, srcPitch[0], rowSize, height, cudaMemcpyDeviceToDevice, props.stream);

        auto dstPtr2 = static_cast<uint8_t *>(dstPtr) + ( height * dstPitch);
		cudaMemcpy2DAsync(dstPtr2, dstPitch, frame3Buffer, srcPitch[0], rowSize, height, cudaMemcpyDeviceToDevice, props.stream); 

        return true;
    }

private:
	int height;
	size_t rowSize;
	ImageMetadata::ImageType inputImageType;
	ImageMetadata::ImageType outputImageType;
	int inputChannels;
	int outputChannels;
    Npp32f* src[4];
	NppiSize srcSize[4];
	int srcPitch[4];
	NppiSize dstSize[4];
	int dstPitch;
    size_t dstNextPtrOffset[4];

	RandomDMAProps props;
};

RandomDMA::RandomDMA(RandomDMAProps _props) : Module(TRANSFORM, "RandomDMA", _props), props(_props), mFrameChecker(0)
{
	mDetail.reset(new Detail(_props));	
}

RandomDMA::~RandomDMA() {}

bool RandomDMA::validateInputPins()
{
	framemetadata_sp metadata = getFirstInputMetadata();
	
	if (getNumberOfInputPins() != 3)
    {
        LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 3. Actual<" << getNumberOfInputPins() << ">";
        return false;
    }

    FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::RAW_IMAGE)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be RAW_IMAGE. Actual<" << frameType << ">";
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

bool RandomDMA::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstOutputMetadata();
	auto mOutputFrameType = metadata->getFrameType();
	if (mOutputFrameType != FrameMetadata::RAW_IMAGE)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input frameType is expected to be RAW_IMAGE. Actual<" << mOutputFrameType << ">";
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

// void RandomDMA::addInputPin(framemetadata_sp& metadata, string& pinId)
// {
// 	Module::addInputPin(metadata, pinId);
// 	auto inputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
// 	switch (inputRawMetadata->getImageType())
// 	{
// 		case ImageMetadata::RGBA:
// 			break;
// 		case ImageMetadata::MONO:
// 			if(!mOutputMetadata){
// 				mOutputMetadata = framemetadata_sp(new RawImageMetadata(FrameMetadata::MemType::DMABUF));
// 				mOutputMetadata->copyHint(*metadata.get());
// 				mOutputPinId = addOutputPin(mOutputMetadata);
// 			}
// 			break;
// 		default:
// 			throw AIPException(AIP_FATAL, "Unsupported Image Type<" + std::to_string(inputRawMetadata->getImageType()) + ">");
// 	}	
// }

void RandomDMA::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);
	auto inputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);

	mOutputMetadata = framemetadata_sp(new RawImageMetadata(FrameMetadata::MemType::DMABUF));
	mOutputMetadata->copyHint(*metadata.get());
	mOutputPinId = addOutputPin(mOutputMetadata);
			
}	

bool RandomDMA::init()
{
	if (!Module::init())
	{
		return false;
	}

	return true;
}

bool RandomDMA::term()
{
	return Module::term();
}

bool RandomDMA::process(frame_container &frames)
{
    cudaFree(0);
    frame_sp frame1, frame2, frame3;
    int i =0;
    for (auto const &element : frames)
    {
        auto frame = element.second;
        if (i == 0)
        {
            frame1 = frame;
        }
        else if (i == 1)
        {
            frame2 = frame;
        }
        else
            frame3 = frame;
        i++;    
    }
    auto outFrame = makeFrame(mOutputMetadata->getDataSize(), mOutputPinId);

    mDetail->compute((static_cast<DMAFDWrapper *>(frame1->data())), (static_cast<DMAFDWrapper *>(frame2->data())), (static_cast<DMAFDWrapper *>(frame3->data())), (static_cast<DMAFDWrapper *>(outFrame->data())));

    frames.insert(make_pair(mOutputPinId, outFrame));
    send(frames);

    return true;
}

// bool RandomDMA::processSOS(frame_sp &frame)
// {
//     LOG_ERROR << "<<<<  Process SOS is Triggered  >>>>";
// 	auto metadata = frame->getMetadata();
// 	auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
// 	if(rawMetadata->getImageType() == ImageMetadata::MONO){
// 		setMetadata(metadata);
// 	}else{
// 		mDetail->rgbaSetMetadata(metadata);
// 	}
// 	mFrameChecker++;

// 	return true;
// }

bool RandomDMA::processSOS(frame_sp &frame)
{
    LOG_ERROR << "<<<<  Process SOS is Triggered  >>>>";
	auto metadata = frame->getMetadata(); 
	auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
	setMetadata(metadata);
	mFrameChecker++;

	return true;
}

void RandomDMA::setMetadata(framemetadata_sp &metadata)
{
	auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
	auto width = rawMetadata->getWidth();
	auto height = rawMetadata->getHeight();
	auto type = rawMetadata->getType();
	auto depth = rawMetadata->getDepth();
	auto inputImageType = rawMetadata->getImageType();

	auto rawOutMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata);
	RawImageMetadata outputMetadata(width * 2, height * 2, inputImageType, rawMetadata->getType(), 512, rawMetadata->getDepth(), FrameMetadata::DMABUF, true);
	rawOutMetadata->setData(outputMetadata);
	mDetail->setMetadata(metadata, mOutputMetadata);
}

bool RandomDMA::shouldTriggerSOS()
{
	return (mFrameChecker == 0 || mFrameChecker == 1 || mFrameChecker == 2);
}

bool RandomDMA::processEOS(string& pinId)
{
	mFrameChecker = 0;
	return true;
}  