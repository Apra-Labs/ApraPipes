#pragma once
#include "Allocators.h"
#include "DMAFDWrapper.h"
#include "nvbuf_utils.h"
#include "FrameMetadataFactory.h"
#include "ApraEGLDisplay.h"
#include "Logger.h"
#include <deque>

class DMAAllocator : public HostAllocator
{
private:
    std::vector<DMAFDWrapper *> mDMAFDWrapperArr;
    int mFreeDMACount;
    NvBufferColorFormat mColorFormat;
    EGLDisplay mEglDisplay;
    int mHeight;
    int mWidth;
    int mCount;

    static NvBufferColorFormat getColorFormat(ImageMetadata::ImageType imageType)
    {
        NvBufferColorFormat colorFormat;
        switch (imageType)
        {
        case ImageMetadata::UYVY:
            colorFormat = NvBufferColorFormat_UYVY;
            break;
        case ImageMetadata::YUYV:
            colorFormat = NvBufferColorFormat_YUYV;
            break;      
        case ImageMetadata::RGBA:
            colorFormat = NvBufferColorFormat_ABGR32;
            break;
        case ImageMetadata::BGRA:
            colorFormat = NvBufferColorFormat_ARGB32;
            break;
        case ImageMetadata::YUV420:
            colorFormat = NvBufferColorFormat_YUV420;
            break;
        case ImageMetadata::NV12:
            colorFormat = NvBufferColorFormat_NV12;
            break;
        default:
            throw AIPException(AIP_FATAL, "Expected <RGBA/BGRA/UYVY/YUV420/NV12> Actual<" + std::to_string(imageType) + ">");
        }

        return colorFormat;
    }

public:
    DMAAllocator(framemetadata_sp framemetadata) : mFreeDMACount(0), mCount(0)
    {
        if (!framemetadata->isSet())
        {
            return;
        }

        mEglDisplay = ApraEGLDisplay::getEGLDisplay();

        auto imageType = ImageMetadata::RGBA;

        auto frameType = framemetadata->getFrameType();
        switch (frameType)
        {
        case FrameMetadata::FrameType::RAW_IMAGE:
        {
            auto inputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(framemetadata);
            mWidth = inputRawMetadata->getWidth();
            mHeight = inputRawMetadata->getHeight();
            imageType = inputRawMetadata->getImageType();
        }
        break;
        case FrameMetadata::FrameType::RAW_IMAGE_PLANAR:
        {
            auto inputRawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(framemetadata);
            mWidth = inputRawMetadata->getWidth(0);
            mHeight = inputRawMetadata->getHeight(0);
            imageType = inputRawMetadata->getImageType();
        }
        break;
        default:
            throw AIPException(AIP_FATAL, "Expected Raw Image or RAW_IMAGE_PLANAR. Actual<" + std::to_string(frameType) + ">");
            break;
        }

        mColorFormat = getColorFormat(imageType);
    };

    ~DMAAllocator()
    {
        for (auto wrapper : mDMAFDWrapperArr)
        {
            delete wrapper;
        }
    }

    static void setMetadata(framemetadata_sp &metadata, int width, int height, ImageMetadata::ImageType imageType)
    {
        auto eglDisplay = ApraEGLDisplay::getEGLDisplay();
        auto colorFormat = getColorFormat(imageType);

        auto dmaFDWrapper = DMAFDWrapper::create(0, width, height, colorFormat, NvBufferLayout_Pitch, eglDisplay);
        if (!dmaFDWrapper)
        {
            LOG_ERROR << "Failed to allocate dmaFDWrapper";
            throw AIPException(AIP_FATAL, "Memory Allocation Failed.");
        }

        NvBufferParams fdParams;
        if (NvBufferGetParams(dmaFDWrapper->getFd(), &fdParams))
        {
            throw AIPException(AIP_FATAL, "NvBufferGetParams Failed.");
        }

        LOG_DEBUG << "PixelFormat<" << fdParams.pixel_format << "> Planes<" << fdParams.num_planes << "> NvBufferSize<" << fdParams.nv_buffer_size << "> MemSize<" << fdParams.memsize << ">";
        for (auto i = 0; i < fdParams.num_planes; i++)
        {
            LOG_DEBUG << "Width<" << fdParams.width[i] << "> Height<" << fdParams.height[i] << "> Pitch<" << fdParams.pitch[i] << "> Offset<" << fdParams.offset[i] << "> PSize<" << fdParams.psize[i] << "> Layout<" << fdParams.layout[i] << ">";
        }

        auto frameType = metadata->getFrameType();
        switch (frameType)
        {
        case FrameMetadata::FrameType::RAW_IMAGE:
        {
            int type = CV_8UC4;
            switch (imageType)
            {
            case ImageMetadata::ImageType::RGBA:
            case ImageMetadata::ImageType::BGRA:
                type = CV_8UC4;
                break;
            case ImageMetadata::ImageType::UYVY:
                type = CV_8UC3;
                break;
            case ImageMetadata::ImageType::YUYV:
                type = CV_8UC3;
                break;    
            default:
                throw AIPException(AIP_FATAL, "Only Image Type accepted are UYVY or ARGB found " + std::to_string(imageType));
            }
            auto inputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
            RawImageMetadata rawMetadata(width, height, imageType, type, fdParams.pitch[0], CV_8U, FrameMetadata::MemType::DMABUF, false);
            inputRawMetadata->setData(rawMetadata);
        }
        break;
        case FrameMetadata::FrameType::RAW_IMAGE_PLANAR:
        {
            auto inputRawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
            size_t step[4] = {0, 0, 0, 0};
            for (auto i = 0; i < fdParams.num_planes; i++)
            {
                step[i] = fdParams.pitch[i];
            }
            RawImagePlanarMetadata rawMetadata(width, height, imageType, step, CV_8U, FrameMetadata::MemType::DMABUF);
            inputRawMetadata->setData(rawMetadata);
        }
        break;
        default:
            throw AIPException(AIP_FATAL, "Expected Raw Image or RAW_IMAGE_PLANAR. Actual<" + std::to_string(frameType) + ">");
            break;
        }

        delete dmaFDWrapper;
    }

    void *allocateChunks(size_t n)
    {
        if (mFreeDMACount == 0)
        {
            auto dmaFDWrapper = DMAFDWrapper::create(mCount++, mWidth, mHeight, mColorFormat, NvBufferLayout_Pitch, mEglDisplay);
            if (!dmaFDWrapper)
            {
                LOG_ERROR << "Failed to allocate dmaFDWrapper";
                throw AIPException(AIP_FATAL, "Memory Allocation Failed.");
            }
            mDMAFDWrapperArr.push_back(dmaFDWrapper);
            dmaFDWrapper->tempFD = dmaFDWrapper->getFd();
            mFreeDMACount++;
        }

        auto wrapper = mDMAFDWrapperArr.front();
        mDMAFDWrapperArr.erase(mDMAFDWrapperArr.begin());
        mFreeDMACount--;

        return static_cast<void *>(wrapper);
    }

    void freeChunks(void *MemPtr, size_t n)
    {
        mDMAFDWrapperArr.push_back(static_cast<DMAFDWrapper *>(MemPtr));
        mFreeDMACount++;
    }

    size_t getChunkSize()
    {
        return 1;
    }
};