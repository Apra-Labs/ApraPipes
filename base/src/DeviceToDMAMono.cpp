#include "DeviceToDMAMono.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "AIPExceptions.h"
#include "DMAFDWrapper.h"

DeviceToDMAMono::DeviceToDMAMono(DeviceToDMAMonoProps _props) : Module(TRANSFORM, "DeviceToDMAMono", _props), props(_props), mFrameLength(0)
{
}

DeviceToDMAMono::~DeviceToDMAMono() {}

bool DeviceToDMAMono::validateInputPins()
{
    if (getNumberOfInputPins() != 1)
    {
        LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
        return false;
    }

    return true;
}

bool DeviceToDMAMono::validateOutputPins()
{
    if (getNumberOfOutputPins() != 1)
    {
        LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
        return false;
    }

    framemetadata_sp metadata = getFirstOutputMetadata();
    mOutputFrameType = metadata->getFrameType();

    FrameMetadata::MemType memType = metadata->getMemType();
    if (memType != FrameMetadata::MemType::DMABUF)
    {
        LOG_ERROR << "<" << getId() << ">::validateOutputPins input memType is expected to be DMABUF. Actual<" << memType << ">";
        return false;
    }

    return true;
}

void DeviceToDMAMono::addInputPin(framemetadata_sp &metadata, string &pinId)
{
    Module::addInputPin(metadata, pinId);
    auto inputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
    mOutputMetadata = framemetadata_sp(new RawImageMetadata(FrameMetadata::MemType::DMABUF));
    mOutputPinId = addOutputPin(mOutputMetadata);
}

bool DeviceToDMAMono::init()
{
    if (!Module::init())
    {
        return false;
    }

    return true;
}

bool DeviceToDMAMono::term()
{
    return Module::term();
}

bool DeviceToDMAMono::process(frame_container &frames)
{
    auto frame = frames.cbegin()->second;
    auto outFrame = makeFrame(mOutputMetadata->getDataSize());
    uint8_t *outBuffer = static_cast<uint8_t *>(static_cast<DMAFDWrapper *>(outFrame->data())->getCudaPtr());

    auto src = static_cast<uint8_t *>(frame->data());
    auto dst = outBuffer;

    auto cudaStatus = cudaMemcpy2DAsync(dst, mDstPitch[0], src, mSrcPitch[0], mRowSize[0], mHeight[0], cudaMemcpyHostToDevice, props.stream);
    if (cudaStatus != cudaSuccess)
    {
        LOG_ERROR << "Cuda Operation Failed";
        
    }

    frames.insert(make_pair(mOutputPinId, outFrame));
    send(frames);
    return true;
}

bool DeviceToDMAMono::processSOS(frame_sp &frame)
{
    auto metadata = frame->getMetadata();
    setMetadata(metadata);
    return true;
}

void DeviceToDMAMono::setMetadata(framemetadata_sp &metadata)
{
    mInputFrameType = metadata->getFrameType();

    auto rawImageMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
    auto width = rawImageMetadata->getWidth();
    auto height = rawImageMetadata->getHeight();
    auto depth = rawImageMetadata->getDepth();
    auto inputImageType = rawImageMetadata->getImageType();

    auto rawOutMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata);
    RawImageMetadata outputMetadata(width, height, ImageMetadata::ImageType::MONO, CV_8UC1, 256, CV_8U, FrameMetadata::HOST, true);
    rawOutMetadata->setData(outputMetadata);

    mFrameLength = mOutputMetadata->getDataSize();

    for (auto i = 0; i < 1; i++)
    {
        mSrcPitch[i] = rawImageMetadata->getStep();
        mRowSize[i] = rawImageMetadata->getRowSize();
        mHeight[i] = rawImageMetadata->getHeight();
    }
    for (auto i = 0; i < 1; i++)
    {
        mDstPitch[i] = rawOutMetadata->getStep();
    }
}

bool DeviceToDMAMono::shouldTriggerSOS()
{
    return mFrameLength == 0;
}

bool DeviceToDMAMono::processEOS(string &pinId)
{
    mFrameLength = 0;
    return true;
}