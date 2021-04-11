#pragma once
#include "Allocators.h"
#include "DMAFDWrapper.h"
#include "nvbuf_utils.h"
#include "FrameMetadataFactory.h"
#include "ApraEGLDisplay.h"
#include <deque>

class DMAAllocator : public HostAllocator
{
private:
    std::vector<DMAFDWrapper*>  dmaFDWrapperArr;
    int freeDMACount;
    NvBufferColorFormat colorFormat;
    EGLDisplay eglDisplay;
    int height;
    int width;
    int count;

public:
    DMAAllocator(framemetadata_sp framemetadata) : freeDMACount(0), count(0)
    {
        if(!framemetadata->isSet())
        {
            return;
        }

        eglDisplay = ApraEGLDisplay::getEGLDisplay();

        auto imageType = ImageMetadata::RGBA;

        auto frameType = framemetadata->getFrameType();
        switch (frameType)
        {
        case FrameMetadata::FrameType::RAW_IMAGE:
        {
            auto inputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(framemetadata);
            width = inputRawMetadata->getWidth();
            height = inputRawMetadata->getHeight();
            imageType = inputRawMetadata->getImageType();
        }
        break;
        case FrameMetadata::FrameType::RAW_IMAGE_PLANAR:
        {
            auto inputRawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(framemetadata);
            width = inputRawMetadata->getWidth(0);
            height = inputRawMetadata->getHeight(0);
            imageType = inputRawMetadata->getImageType();
        }
        break;
        default:
            throw AIPException(AIP_FATAL, "Expected Raw Image or RAW_IMAGE_PLANAR. Actual<" + std::to_string(frameType) + ">");
            break;
        }

        switch (imageType)
        {
        case ImageMetadata::UYVY:
            colorFormat = NvBufferColorFormat_UYVY;
            break;
        case ImageMetadata::RGBA:
            colorFormat = NvBufferColorFormat_ARGB32;
            break;
        case ImageMetadata::YUV420:
            colorFormat = NvBufferColorFormat_YUV420;
            break;
        default:
            throw AIPException(AIP_FATAL, "Only Image Type accepted are UYVY or ARGB found " + std::to_string(imageType));
        }
    };

    ~DMAAllocator()
    {
        for(auto wrapper : dmaFDWrapperArr)
        {
            delete wrapper;
        }             
    }

    void *allocateChunks(size_t n)
    {
        if (freeDMACount == 0)
        {
            auto dmaFDWrapper = DMAFDWrapper::create(count++, width, height, colorFormat, NvBufferLayout_Pitch, eglDisplay);
            if (!dmaFDWrapper)
            {
                LOG_ERROR << "Failed to allocate dmaFDWrapper";
                throw AIPException(AIP_FATAL, "Memory Allocation Failed.");
            }
            dmaFDWrapperArr.push_back(dmaFDWrapper);
            dmaFDWrapper->tempFD = dmaFDWrapper->getFd();
            freeDMACount++;
        }

        auto wrapper = dmaFDWrapperArr.front();
        dmaFDWrapperArr.erase(dmaFDWrapperArr.begin());
        freeDMACount--;

        return static_cast<void *>(wrapper);
    }

    void freeChunks(void *MemPtr, size_t n)
    {
        dmaFDWrapperArr.push_back(static_cast<DMAFDWrapper *>(MemPtr));
        freeDMACount++;
    }

    size_t getChunkSize()
    {
        return 1;
    }
};