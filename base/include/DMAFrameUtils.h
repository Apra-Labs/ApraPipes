#pragma once

#include "Frame.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "ImagePlaneData.h"
#include "AIPExceptions.h"
#include "DMAFDWrapper.h"
#include "DMAAllocator.h"
#include <memory>

class DMAFrameUtils
{
public:
    typedef std::function<void(frame_sp &, ImagePlanes &)> GetImagePlanes;

public:
    static GetImagePlanes getImagePlanesFunction(framemetadata_sp &metadata, ImagePlanes &imagePlanes)
    {

        auto frameType = metadata->getFrameType();
        switch (frameType)
        {
        case FrameMetadata::FrameType::RAW_IMAGE:
        {
            size_t pitch[4] = {0, 0, 0, 0};
            auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
            auto imageType = rawMetadata->getImageType();
            FrameMetadata::MemType inputMemType = metadata->getMemType();
            if (inputMemType == FrameMetadata::MemType::DMABUF)
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
                auto imageType = rawMetadata->getImageType();
                auto metadata = framemetadata_sp(new RawImageMetadata(rawMetadata->getWidth(), rawMetadata->getHeight(), imageType, type, size_t(0), CV_8U, FrameMetadata::MemType::DMABUF, true));
                DMAAllocator::setMetadata(metadata, rawMetadata->getWidth(), rawMetadata->getHeight(), imageType, pitch);

                imagePlanes.push_back(std::make_shared<ImagePlaneData>(metadata->getDataSize(),
                                                                       pitch[0],
                                                                       rawMetadata->getRowSize(),
                                                                       rawMetadata->getWidth(),
                                                                       rawMetadata->getHeight()));
            }

            else
            {

                imagePlanes.push_back(std::make_shared<ImagePlaneData>(metadata->getDataSize(),
                                                                       rawMetadata->getStep(),
                                                                       rawMetadata->getRowSize(),
                                                                       rawMetadata->getWidth(),
                                                                       rawMetadata->getHeight()));
            }

            return getDMAFDHostImagePlanes;
        }
        case FrameMetadata::FrameType::RAW_IMAGE_PLANAR:
        {
            auto rawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
            auto imageType = rawMetadata->getImageType();
            auto channels = rawMetadata->getChannels();
            FrameMetadata::MemType inputMemType = metadata->getMemType();
            size_t pitch[4] = {0, 0, 0, 0};
            size_t offset[4] = {0, 0, 0, 0};
            size_t width[4];
            size_t height[4];
            if (inputMemType == FrameMetadata::MemType::DMABUF)
            {
                for (auto i = 0; i < channels; i++)
                {
                    width[i] = rawMetadata->getWidth(i);
                    height[i] = rawMetadata->getHeight(i);
                    pitch[i]=rawMetadata->getStep(i);
                }
                auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width[0], height[0], imageType, size_t(0), CV_8U, FrameMetadata::MemType::DMABUF));
                DMAAllocator::setMetadata(metadata, width[0], height[0], imageType, pitch, offset);

                for (auto i = 0; i < channels; i++)
                {
                    imagePlanes.push_back(std::make_shared<ImagePlaneData>(rawMetadata->getDataSizeByChannel(i),
                                                                           pitch[i],
                                                                           rawMetadata->getRowSize(i),
                                                                           width[i],
                                                                           height[i]));
                }
            }

            else
            {
                for (auto i = 0; i < channels; i++)
                {
                    imagePlanes.push_back(std::make_shared<ImagePlaneData>(rawMetadata->getDataSizeByChannel(i),
                                                                           rawMetadata->getStep(i),
                                                                           rawMetadata->getRowSize(i),
                                                                           rawMetadata->getWidth(i),
                                                                           rawMetadata->getHeight(i)));
                }
            }

            switch (imageType)
            {
            case ImageMetadata::ImageType::YUV420:
                return getDMAFDYUV420HostImagePlanes;
            case ImageMetadata::ImageType::NV12:
                return getDMAFDNV12HostImagePlanes;
            default:
                throw AIPException(AIP_FATAL, "Unsupported ImageType<" + std::to_string(imageType) + ">");
            }
        }
        default:
            throw AIPException(AIP_FATAL, "Unsupported FrameType<" + std::to_string(frameType) + ">");
        }
    }

    static void getDMAFDHostImagePlanes(frame_sp &frame, ImagePlanes &imagePlanes)
    {
        auto inputMetadata = frame->getMetadata();
        FrameMetadata::MemType mInputMemType = inputMetadata->getMemType();
        if (mInputMemType == FrameMetadata::MemType::DMABUF)
        {

            auto ptr = static_cast<DMAFDWrapper *>(frame->data());
            imagePlanes[0]->data = ptr->getHostPtr();
        }
        else
        {
            auto ptr = static_cast<uint8_t *>(frame->data());
            imagePlanes[0]->data = ptr;
        }
    }

    static void getDMAFDYUV420HostImagePlanes(frame_sp &frame, ImagePlanes &imagePlanes)
    {
        auto ptr = static_cast<DMAFDWrapper *>(frame->data());
        imagePlanes[0]->data = ptr->getHostPtrY();
        imagePlanes[1]->data = ptr->getHostPtrU();
        imagePlanes[2]->data = ptr->getHostPtrV();
    }

    static void getDMAFDNV12HostImagePlanes(frame_sp &frame, ImagePlanes &imagePlanes)
    {
        auto ptr = static_cast<DMAFDWrapper *>(frame->data());
        imagePlanes[0]->data = ptr->getHostPtrY();
        imagePlanes[1]->data = ptr->getHostPtrUV();
    }
};