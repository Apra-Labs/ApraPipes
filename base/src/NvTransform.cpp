#include "NvTransform.h"
//#include "nvbuf_utils.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "AIPExceptions.h"
#include "DMAFDWrapper.h"
#include "DMAAllocator.h"
#include "npp.h"
#include <nvbufsurface.h> 
#include <nvbufsurftransform.h> 

class NvTransform::Detail
{
public:
    Detail(NvTransformProps &_props) : props(_props)
{
    src_rect.top = _props.top;
    src_rect.left = _props.left;
    src_rect.width = _props.width;
    src_rect.height = _props.height;


    memset(&transParams, 0, sizeof(transParams));
    transParams.transform_filter = NvBufSurfTransformInter_Default;

    transParams.transform_flag = NVBUFSURF_TRANSFORM_FILTER;

    if (src_rect.width != 0 && src_rect.height != 0)
    {
        transParams.src_rect = &src_rect;
        transParams.transform_flag |= NVBUFSURF_TRANSFORM_CROP_SRC | NVBUFSURF_TRANSFORM_ALLOW_ODD_CROP;
    }

    transParams.transform_flip = NvBufSurfTransform_None;
    if (_props.rotation != (NvTransformProps::NvRotation)0)
        {
            transParams.transform_flag |= NVBUFSURF_TRANSFORM_FLIP;

            switch (_props.rotation)
            {
            case NvTransformProps::NvRotation::Rotate90:
                transParams.transform_flip = NvBufSurfTransform_Rotate90;
                break;
            case NvTransformProps::NvRotation::Rotate180:
                transParams.transform_flip = NvBufSurfTransform_Rotate180;
                break;
            case NvTransformProps::NvRotation::Rotate270:
                transParams.transform_flip = NvBufSurfTransform_Rotate270;
                break;
            default:
                LOG_ERROR << "Invalid rotation angle. Supported: 0, 90, 180, 270.";
                transParams.transform_flip = NvBufSurfTransform_None;
                break;
            }
        }
        if (_props.flip != (NvTransformProps::NvFlip)0)
        {
            transParams.transform_flag |= NVBUFSURF_TRANSFORM_FLIP;

            switch (_props.flip)
            {
            case NvTransformProps::NvFlip::FlipX:
                transParams.transform_flip = NvBufSurfTransform_FlipX;
                break;
            case NvTransformProps::NvFlip::FlipY:
                transParams.transform_flip = NvBufSurfTransform_FlipY;
                break;
            default:
                LOG_ERROR << "Invalid flip value. Supported: None, FlipX, FlipY.";
                transParams.transform_flip = NvBufSurfTransform_None;
                break;
            }
        }
}


    bool compute(frame_sp &frame, int outFD)
    {
        auto dmaFDWrapper = static_cast<DMAFDWrapper *>(frame->data());
        NvBufSurface *in_surf = nullptr;
        NvBufSurface *out_surf = nullptr;

        if (NvBufSurfaceFromFd(dmaFDWrapper->getFd(), (void**)&in_surf) != 0) {
            LOG_INFO << "Failed to create input surface";
            return false;
        }

        if (NvBufSurfaceFromFd(outFD, (void**)&out_surf) != 0) {
            LOG_INFO << "Failed to create output surface";
            return false;
        }

        int in_planes  = in_surf->surfaceList[0].planeParams.num_planes;
        int out_planes = out_surf->surfaceList[0].planeParams.num_planes;

        for (int p = 0; p < in_planes; ++p) {
            NvBufSurfaceSyncForDevice(in_surf, 0, p);
        }

        NvBufSurfTransform_Error err = NvBufSurfTransform(in_surf, out_surf, &transParams);
        if (err != NvBufSurfTransformError_Success) {
            LOG_INFO << "Transform failed";
        }

        for (int p = 0; p < out_planes; ++p) {
            NvBufSurfaceSyncForCpu(out_surf, 0, p);
        }

        return true;
    }


public:
    NvBufSurfTransformRect src_rect;
    framemetadata_sp outputMetadata;
    std::string outputPinId;
    NvTransformProps props;

private:
    NvBufSurfTransformParams transParams;
};

NvTransform::NvTransform(NvTransformProps props) : Module(TRANSFORM, "NvTransform", props)
{
    mDetail.reset(new Detail(props));
}

NvTransform::~NvTransform() {}

bool NvTransform::validateInputPins()
{
    if (getNumberOfInputPins() != 1)
    {
        LOG_INFO << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
        return false;
    }

    framemetadata_sp metadata = getFirstInputMetadata();
    FrameMetadata::FrameType frameType = metadata->getFrameType();
    if (frameType != FrameMetadata::RAW_IMAGE && frameType != FrameMetadata::RAW_IMAGE_PLANAR)
    {
		LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be RAW_IMAGE or RAW_IMAGE_PLANAR. Actual<" << frameType << ">";
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

bool NvTransform::validateOutputPins()
{
    if (getNumberOfOutputPins() != 1)
    {
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
        return false;
    }

    framemetadata_sp metadata = getFirstOutputMetadata();
    auto frameType = metadata->getFrameType();
    if (frameType != FrameMetadata::RAW_IMAGE && frameType != FrameMetadata::RAW_IMAGE_PLANAR)
    {
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input frameType is expected to be RAW_IMAGE or RAW_IMAGE_PLANAR. Actual<" << frameType << ">";
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

void NvTransform::addInputPin(framemetadata_sp &metadata, string &pinId)
{
    Module::addInputPin(metadata, pinId);
    switch (mDetail->props.imageType)
    {
    case ImageMetadata::BGRA:
    case ImageMetadata::RGBA:
        mDetail->outputMetadata = framemetadata_sp(new RawImageMetadata(FrameMetadata::MemType::DMABUF));
        break;
    case ImageMetadata::NV12:
    case ImageMetadata::YUV420:
    case ImageMetadata::YUV444:
        mDetail->outputMetadata = framemetadata_sp(new RawImagePlanarMetadata(FrameMetadata::MemType::DMABUF));
        break;
    default:
        throw AIPException(AIP_FATAL, "Unsupported Image Type<" + std::to_string(mDetail->props.imageType) + ">");
    }

    mDetail->outputMetadata->copyHint(*metadata.get());
    mDetail->outputPinId = addOutputPin(mDetail->outputMetadata);
}

bool NvTransform::init()
{
    if (!Module::init())
    {
        return false;
    }

    return true;
}

bool NvTransform::term()
{
    return Module::term();
}

bool NvTransform::process(frame_container &frames)
{
    try
    {
        auto frame = frames.cbegin()->second;
    if(isFrameEmpty(frame))
    {
        LOG_INFO << "Found Empty Frame ";
        return true;
    }
    if(!mDetail->outputMetadata->getDataSize())
    {
        return true;
    }
    auto outFrame = makeFrame(mDetail->outputMetadata->getDataSize(), mDetail->outputPinId);
    if (!outFrame.get())
    {
			LOG_ERROR << "FAILED TO GET BUFFER";
        return false;
    }

    auto dmaFdWrapper = static_cast<DMAFDWrapper *>(outFrame->data());
    dmaFdWrapper->tempFD = dmaFdWrapper->getFd();

    mDetail->compute(frame, dmaFdWrapper->tempFD);

    frames.insert(make_pair(mDetail->outputPinId, outFrame));
    send(frames);
}
	
    return true;
}

bool NvTransform::processSOS(frame_sp &frame)
{
    auto metadata = frame->getMetadata();
    setMetadata(metadata);

    return true;
}

void NvTransform::setMetadata(framemetadata_sp &metadata)
{
    auto frameType = metadata->getFrameType();
    int width = 0;
    int height = 0;
    int depth = CV_8U;
    ImageMetadata::ImageType inputImageType = ImageMetadata::ImageType::MONO;

    switch (frameType)
    {
    case FrameMetadata::FrameType::RAW_IMAGE:
    {
        auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
        width = rawMetadata->getWidth();
        height = rawMetadata->getHeight();
        depth = rawMetadata->getDepth();
        inputImageType = rawMetadata->getImageType();
    }
    break;
    case FrameMetadata::FrameType::RAW_IMAGE_PLANAR:
    {
        auto rawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
        width = rawMetadata->getWidth(0);
        height = rawMetadata->getHeight(0);
        depth = rawMetadata->getDepth();
        inputImageType = rawMetadata->getImageType();
    }
    break;
    default:
        throw AIPException(AIP_NOTIMPLEMENTED, "Unsupported FrameType<" + std::to_string(frameType) + ">");
    }

    if (mDetail->props.width != 0)
    {
        width = mDetail->props.width;
        height = mDetail->props.height;
    }

    DMAAllocator::setMetadata(mDetail->outputMetadata, width, height, mDetail->props.imageType);
}

bool NvTransform::processEOS(string &pinId)
{
    LOG_DEBUG<< "Resetting Output Metadata";
    // mDetail->outputMetadata.reset();
    return true;
}