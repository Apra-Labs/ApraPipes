#pragma once
#include "Allocators.h"
#include "DMAFDWrapper.h"
#include "nvbufsurface.h"
#include "ImageMetadata.h"
#include "RawImageMetadata.h"
#include "RawImagePlanarMetadata.h"
#include "FrameMetadataFactory.h"
#include "Logger.h"
#include <deque>

class DMAAllocator : public HostAllocator
{
private:
    std::vector<DMAFDWrapper *> mDMAFDWrapperArr;
    int mFreeDMACount;
    NvBufSurfaceColorFormat  mColorFormat;
    EGLDisplay mEglDisplay;
    int mHeight;
    int mWidth;
    int mCount;

    static NvBufSurfaceColorFormat  getColorFormat(ImageMetadata::ImageType imageType)
    {
        NvBufSurfaceColorFormat  colorFormat;
        switch (imageType)
        {
        case ImageMetadata::UYVY:
            colorFormat = NVBUF_COLOR_FORMAT_UYVY;
            break;
        case ImageMetadata::YUYV:
            colorFormat = NVBUF_COLOR_FORMAT_YUYV;
            break;      
        case ImageMetadata::RGBA:
            colorFormat = NVBUF_COLOR_FORMAT_RGBA;
            break;
        case ImageMetadata::BGRA:
            colorFormat = NVBUF_COLOR_FORMAT_BGRA;
            break;
        case ImageMetadata::YUV420:
            colorFormat = NVBUF_COLOR_FORMAT_YUV420;
            break;
        case ImageMetadata::NV12:
            colorFormat = NVBUF_COLOR_FORMAT_NV12;
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

    static void setMetadata(framemetadata_sp &metadata, int width, int height, ImageMetadata::ImageType imageType,size_t pitchValues[4] = nullptr, size_t offsetValues[4] = nullptr)
    {
        auto eglDisplay = ApraEGLDisplay::getEGLDisplay();
        auto colorFormat = getColorFormat(imageType);

        auto dmaFDWrapper = DMAFDWrapper::create(0, width, height, colorFormat, NVBUF_LAYOUT_PITCH, eglDisplay);
        if (!dmaFDWrapper)
        {
            LOG_ERROR << "Failed to allocate dmaFDWrapper";
            throw AIPException(AIP_FATAL, "Memory Allocation Failed.");
        }

        auto surf = dmaFDWrapper->getNvBufSurface();
        if (!surf)
        {
            throw AIPException(AIP_FATAL, "NvBufSurface is null.");
        }

        auto &fdParams = surf->surfaceList[0];
        LOG_DEBUG << "PixelFormat<" << fdParams.colorFormat << "> Layout<" << fdParams.layout << ">";
        LOG_DEBUG << "Width<" << fdParams.width << "> Height<" << fdParams.height << "> Pitch<" << fdParams.planeParams.pitch[0] << "> Offset<" << fdParams.planeParams.offset[0] << "> PSize<" << fdParams.planeParams.psize[0] << ">";

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
            case ImageMetadata::ImageType::YUYV:
                type = CV_8UC3;
                break;    
            default:
                throw AIPException(AIP_FATAL, "Only Image Type accepted are UYVY or ARGB found " + std::to_string(imageType));
            }
            auto inputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
            RawImageMetadata rawMetadata(width, height, imageType, type, fdParams.planeParams.pitch[0], CV_8U, FrameMetadata::MemType::DMABUF, false);
            inputRawMetadata->setData(rawMetadata);
            if(pitchValues != nullptr)
            {
              pitchValues[0] = fdParams.planeParams.pitch[0];
            }
        }
        break;
        case FrameMetadata::FrameType::RAW_IMAGE_PLANAR:
        {
            auto inputRawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
            size_t step[4] = {0, 0, 0, 0};
            // Populate pitch/offset for all available planes
            int num_planes = fdParams.planeParams.num_planes;
            for (int i = 0; i < num_planes && i < 4; i++)
            {
                step[i] = fdParams.planeParams.pitch[i];
                if (pitchValues != nullptr)
                {
                    pitchValues[i] = fdParams.planeParams.pitch[i];
                }
                if (offsetValues != nullptr)
                {
                    offsetValues[i] = fdParams.planeParams.offset[i];
                }
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
            auto dmaFDWrapper = DMAFDWrapper::create(mCount++, mWidth, mHeight, mColorFormat, NVBUF_LAYOUT_PITCH, mEglDisplay);
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