#pragma once

#include "Frame.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "ImagePlaneData.h"
#include "AIPExceptions.h"
#include "DMAFDWrapper.h"
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
            auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
            imagePlanes.push_back(std::make_shared<ImagePlaneData>(metadata->getDataSize(),
                                                                   rawMetadata->getStep(),
                                                                   rawMetadata->getRowSize(),
                                                                   rawMetadata->getWidth(),
                                                                   rawMetadata->getHeight()));
            return getDMAFDHostImagePlanes;
        }
        case FrameMetadata::FrameType::RAW_IMAGE_PLANAR:
        {
            auto rawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
            auto imageType = rawMetadata->getImageType();

            auto channels = rawMetadata->getChannels();
            for (auto i = 0; i < channels; i++)
            {
                imagePlanes.push_back(std::make_shared<ImagePlaneData>(rawMetadata->getDataSizeByChannel(i),
                                                                       rawMetadata->getStep(i),
                                                                       rawMetadata->getRowSize(i),
                                                                       rawMetadata->getWidth(i),
                                                                       rawMetadata->getHeight(i)));
            }

            switch (imageType)
            {
            case ImageMetadata::ImageType::YUV420:
            case ImageMetadata::ImageType::YUV444:
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
        auto ptr = static_cast<DMAFDWrapper *>(frame->data());
        imagePlanes[0]->data = ptr->getHostPtr();
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