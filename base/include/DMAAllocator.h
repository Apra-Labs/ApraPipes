#pragma once
#include "Allocators.h"
#include "DMAFDWrapper.h"
#include "nvbuf_utils.h"
#include "FrameMetadataFactory.h"
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

        if(framemetadata->getFrameType() != FrameMetadata::FrameType::RAW_IMAGE){
            throw AIPException(AIP_FATAL, "Only Frame Type accepted are Raw Image");
        }

        eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if(eglDisplay == EGL_NO_DISPLAY)
        {
            throw AIPException(AIP_FATAL, "eglGetDisplay failed");
        } 

        if (!eglInitialize(eglDisplay, NULL, NULL))
        {
            throw AIPException(AIP_FATAL, "eglInitialize failed");
        } 

		auto inputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(framemetadata);
		width = inputRawMetadata->getWidth();
        height = inputRawMetadata->getHeight();
        switch(inputRawMetadata->getImageType()){
            case ImageMetadata::UYVY:
                colorFormat = NvBufferColorFormat_UYVY;
                break;
            case ImageMetadata::RGBA:
                colorFormat = NvBufferColorFormat_ARGB32;
                break;
            default:
                throw AIPException(AIP_FATAL, "Only Image Type accepted are UYVY or ARGB found " + std::to_string(inputRawMetadata->getImageType()));
        }
    };

    ~DMAAllocator()
    {
        for(auto wrapper : dmaFDWrapperArr)
        {
            delete wrapper;
        }

        eglTerminate(eglDisplay);
        if (!eglReleaseThread())
        {
            LOG_ERROR << "ERROR eglReleaseThread failed";
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