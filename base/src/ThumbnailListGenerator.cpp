#include "ThumbnailListGenerator.h"
#include "FrameMetadata.h"
#include "ImageMetadata.h"
#include "RawImageMetadata.h"
#include "RawImagePlanarMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include "Utils.h"
#include <vector>
#include <fstream>
#include <jpeglib.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cstdio>

#include "DMAFDWrapper.h"
#include "DMAFrameUtils.h"

class ThumbnailListGenerator::Detail
{

public:
    Detail(ThumbnailListGeneratorProps &_props) : mProps(_props)
    {
        mOutSize = cv::Size(mProps.thumbnailWidth, mProps.thumbnailHeight);
        enableSOS = true;
        flags.push_back(cv::IMWRITE_JPEG_QUALITY);
        flags.push_back(90);
    }

    ~Detail() {}

    void initMatImages(framemetadata_sp &input)
    {
        mIImg = Utils::getMatHeader(FrameMetadataFactory::downcast<RawImageMetadata>(input));
    }

    void setProps(ThumbnailListGeneratorProps &props)
    {
        mProps = props;
    }

    cv::Mat mIImg;
    cv::Size mOutSize;
    bool enableSOS;
    ThumbnailListGeneratorProps mProps;
    int m_width;
    int m_height;
    int m_step;
    cv::Mat m_tempImage;
    int count = 0;
    vector<int> flags;
};

ThumbnailListGenerator::ThumbnailListGenerator(ThumbnailListGeneratorProps _props) : Module(SINK, "ThumbnailListGenerator", _props)
{
    mDetail.reset(new Detail(_props));
}

ThumbnailListGenerator::~ThumbnailListGenerator() {}

bool ThumbnailListGenerator::validateInputPins()
{
    // if (getNumberOfInputPins() != 1)
    // {
    //     LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
    //     return false;
    // }

    framemetadata_sp metadata = getFirstInputMetadata();
    FrameMetadata::FrameType frameType = metadata->getFrameType();
    if (frameType != FrameMetadata::RAW_IMAGE_PLANAR)
    {
        LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be RAW_IMAGE. Actual<" << frameType << ">";
        return false;
    }

    return true;
}

bool ThumbnailListGenerator::init()
{
    if (!Module::init())
    {
        return false;
    }
    return true;
}

bool ThumbnailListGenerator::term()
{
    return Module::term();
}

bool ThumbnailListGenerator::process(frame_container &frames)
{
    auto frame = getFrameByType(frames, FrameMetadata::RAW_IMAGE_PLANAR);
    if (isFrameEmpty(frame))
    {
        LOG_ERROR << "Got Empty Frames will return from here ";
        return true;
    }

    // ImagePlanes mImagePlanes;
    // DMAFrameUtils::GetImagePlanes mGetImagePlanes;
    // int mNumPlanes = 0;

     framemetadata_sp frameMeta = frame->getMetadata();

    // mGetImagePlanes = DMAFrameUtils::getImagePlanesFunction(frameMeta, mImagePlanes);
	// mNumPlanes = static_cast<int>(mImagePlanes.size());

    // mGetImagePlanes(frame, mImagePlanes);

    // uint8_t* dstPtr = (uint8_t*) malloc(frameMeta->getDataSize());
    // for (auto i = 0; i < mNumPlanes; i++)
	// {
	// 	mImagePlanes[i]->mCopyToData(mImagePlanes[i].get(), dstPtr);
	// 	dstPtr += mImagePlanes[i]->imageSize;
	// }

    // FrameMetadata::FrameType fType = frameMeta->getFrameType();

    // uint8_t* dstPtr = (uint8_t*) malloc(frame->size());
    // auto frameSize = frame->size();

    // dstPtr = (uint8_t*)(static_cast<DMAFDWrapper *>(frame->data()))->getHostPtrY();
    // dstPtr += frameSize / 2;
    // dstPtr = (uint8_t*)(static_cast<DMAFDWrapper *>(frame->data()))->getHostPtrU();
    // dstPtr += frameSize / 4;
    // dstPtr = (uint8_t*)(static_cast<DMAFDWrapper *>(frame->data()))->getHostPtrV();
    // dstPtr += frameSize / 4;
    // dstPtr -= frameSize;

    auto dstPtr = (uint8_t*)(static_cast<DMAFDWrapper *>(frame->data()))->getHostPtr();

    auto rawPlanarMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(frameMeta);
    auto height = rawPlanarMetadata->getHeight(0);
    auto width = rawPlanarMetadata->getWidth(0);
    LOG_ERROR << "width = "<< width;
     LOG_ERROR << "height = "<< height;
    auto st = rawPlanarMetadata->getStep(0);
    uint8_t data = 0;
    cv::Mat bgrImage;
    auto yuvImage = cv::Mat(height * 1.5, width, CV_8UC1, static_cast<void*>(&data));
    yuvImage.data = static_cast<uint8_t*>(dstPtr);
    cv::cvtColor(yuvImage, bgrImage, cv::COLOR_YUV2BGRA_NV12);

    cv::Mat bgrImageResized;
    auto newSize = cv::Size(1000, 1000);

    cv::resize(bgrImage, bgrImageResized, newSize);

    unsigned char* frame_buffer = (unsigned char*)bgrImageResized.data;
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    JSAMPROW row_pointer[1];
    FILE* outfile = fopen(mDetail->mProps.fileToStore.c_str(), "wb");
    if (!outfile)
    {
        LOG_ERROR << "Couldn't open file" << mDetail->mProps.fileToStore.c_str();
        return false;
    }
    mDetail->count = mDetail->count + 1;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, outfile);

    // Set the image dimensions and color space
    cinfo.image_width = 1000;
    cinfo.image_height = 1000;
    cinfo.input_components = 4;
    cinfo.in_color_space = JCS_EXT_BGRA;

    // Set the JPEG compression parameters
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 80, TRUE);

    // Start the compression process
    jpeg_start_compress(&cinfo, TRUE);
    // Loop over the image rows
    while (cinfo.next_scanline < cinfo.image_height)
    {
        // Get a pointer to the current row
        row_pointer[0] = &frame_buffer[cinfo.next_scanline * 1000 * 4];
        if (row_pointer && &cinfo)
        {
            // Compress the row
            jpeg_write_scanlines(&cinfo, row_pointer, 1);
        }
        else
        {
            LOG_ERROR << "COULDN'T WRITE .......................................";
        }
    }

    // Finish the compression process
    jpeg_finish_compress(&cinfo);

    // Clean up the JPEG compression object and close the output file
    jpeg_destroy_compress(&cinfo);
    fclose(outfile);
    LOG_ERROR << "wrote thumbail";
    return true;
}

void ThumbnailListGenerator::setProps(ThumbnailListGeneratorProps &props)
{
    Module::addPropsToQueue(props);
}

ThumbnailListGeneratorProps ThumbnailListGenerator::getProps()
{
    fillProps(mDetail->mProps);
    return mDetail->mProps;
}

bool ThumbnailListGenerator::handlePropsChange(frame_sp &frame)
{
    ThumbnailListGeneratorProps props(0, 0, "s");
    bool ret = Module::handlePropsChange(frame, props);
    mDetail->setProps(props);
    return ret;
}