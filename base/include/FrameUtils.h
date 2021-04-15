#pragma once

#include "Frame.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Dataset.h"
#include "AIPExceptions.h"
#ifdef ARM64
#include "DMAFDWrapper.h"
#endif

class FrameUtils
{
public:
    typedef std::function<void (frame_sp &, Dataset &dataset)> GetDataset;

public:
    static GetDataset getDatasetFunction(framemetadata_sp &metadata, Dataset &dataset)
    {
        auto memType = metadata->getMemType();
        switch (memType)
        {
#ifdef ARM64
        case FrameMetadata::MemType::DMABUF:
        {
            auto frameType = metadata->getFrameType();
            switch (frameType)
            {
            case FrameMetadata::FrameType::RAW_IMAGE:
                dataset.data.push_back(nullptr);
                dataset.size.push_back(metadata->getDataSize());
                return getDMAFDHostDataset;
            case FrameMetadata::FrameType::RAW_IMAGE_PLANAR:
            {
                auto rawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
                auto imageType = rawMetadata->getImageType();
                switch (imageType)
                {
                case ImageMetadata::ImageType::YUV420:
                    dataset.data.push_back(nullptr);
                    dataset.size.push_back(rawMetadata->getDataSizeByChannel(0));
                    dataset.data.push_back(nullptr);
                    dataset.size.push_back(rawMetadata->getDataSizeByChannel(1));
                    dataset.data.push_back(nullptr);
                    dataset.size.push_back(rawMetadata->getDataSizeByChannel(2));
                    return getDMAFDYUV420HostDataset;
                case ImageMetadata::ImageType::NV12:
                    dataset.data.push_back(nullptr);
                    dataset.size.push_back(rawMetadata->getDataSizeByChannel(0));
                    dataset.data.push_back(nullptr);
                    dataset.size.push_back(rawMetadata->getDataSizeByChannel(1));
                    return getDMAFDNV12HostDataset;
                default:
                    throw AIPException(AIP_FATAL, "Unsupported ImageType<" + std::to_string(imageType) + ">");
                }
            }
            default:
                throw AIPException(AIP_FATAL, "Unsupported FrameType<" + std::to_string(frameType) + ">");
            }
            break;
        }
#endif
        default:
            dataset.data.push_back(nullptr);
            dataset.size.push_back(0);
            return getHostDataset;
        }
    }

    static void getHostDataset(frame_sp &frame, Dataset &dataset)
    {
        dataset.data[0] = frame->data();
        dataset.size[0] = frame->size();
    }

#ifdef ARM64
    static void getDMAFDHostDataset(frame_sp &frame, Dataset &dataset)
    {
        auto ptr = static_cast<DMAFDWrapper *>(frame->data());
        dataset.data[0] = ptr->getHostPtr();
    }

    static void getDMAFDYUV420HostDataset(frame_sp &frame, Dataset &dataset)
    {
        auto ptr = static_cast<DMAFDWrapper *>(frame->data());
        dataset.data[0] = ptr->getHostPtrY();
        dataset.data[1] = ptr->getHostPtrU();
        dataset.data[2] = ptr->getHostPtrV();
    }

    static void getDMAFDNV12HostDataset(frame_sp &frame, Dataset &dataset)
    {
        auto ptr = static_cast<DMAFDWrapper *>(frame->data());
        dataset.data[0] = ptr->getHostPtrY();
        dataset.data[1] = ptr->getHostPtrUV();
    }
#endif
};