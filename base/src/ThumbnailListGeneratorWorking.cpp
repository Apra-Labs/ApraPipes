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

        // auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(input);
        // m_height = rawMetadata->getHeight();
        // m_width = rawMetadata->getWidth();
        // m_step = rawMetadata->getStep();
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
    if (getNumberOfInputPins() != 1)
    {
        LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
        return false;
    }

    framemetadata_sp metadata = getFirstInputMetadata();
    FrameMetadata::FrameType frameType = metadata->getFrameType();
    if (frameType != FrameMetadata::RAW_IMAGE)
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

// void ThumbnailListGenerator::addInputPin(framemetadata_sp &metadata, string &pinId)
// {
//     // Module::addInputPin(metadata, pinId);
//     mDetail->initMatImages(metadata); // should do inside SOS
// }

bool ThumbnailListGenerator::process(frame_container &frames)
{
    auto frame = getFrameByType(frames, FrameMetadata::RAW_IMAGE);
    if (isFrameEmpty(frame))
    {
        LOG_ERROR << "Got Empty Frames will return from here ";
        return true;
    }
    LOG_ERROR << "Size Of Frame in Thumbnail Generator is " << frame->size();

    framemetadata_sp frameMeta = frame->getMetadata();
    FrameMetadata::FrameType fType = frameMeta->getFrameType();

    auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(frameMeta);
    auto height = rawMetadata->getHeight();
    auto width = rawMetadata->getWidth();
    auto st = rawMetadata->getStep();

    LOG_ERROR << "Width of a frame is " << width << "height of a frame is " << height << "Channels " << rawMetadata->getChannels();
    LOG_ERROR << "Image type of a frame is "<< rawMetadata->getImageType();

    unsigned char *frame_buffer = (unsigned char *)frame->data();
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    JSAMPROW row_pointer[1];
    std::string filename = "/home/developer/data/2023-04-04/DOCTOR/PATIENT/" + std::to_string(mDetail->count) + "_i_t.jpeg";
    FILE *outfile = fopen(mDetail->mProps.fileToStore.c_str(), "wb");
    LOG_ERROR << "FILE NAME IS =====================================>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << mDetail->mProps.fileToStore;
    // FILE* outfile = fopen(mDetail->mProps.fileToStore.c_str(), "wb");
    // /home/developer/data/2023-04-04/DOCTOR/PATIENT/.th/
    if (!outfile)
    {
        // fprintf(stderr, "Error: could not open \n", filename);
        LOG_ERROR << "Couldn't open file";
        return false;
    }
    mDetail->count = mDetail->count + 1;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, outfile);

    // Set the image dimensions and color space
    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 4;
    cinfo.in_color_space = JCS_EXT_RGBA;

    // Set the JPEG compression parameters
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 80, TRUE);

    // Start the compression process
    jpeg_start_compress(&cinfo, TRUE);
    // fclose(outfile);

    // return true;

    // Loop over the image rows
    while (cinfo.next_scanline < cinfo.image_height)
    {
        // Get a pointer to the current row
        row_pointer[0] = &frame_buffer[cinfo.next_scanline * width * 4];
        if(row_pointer && &cinfo)
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