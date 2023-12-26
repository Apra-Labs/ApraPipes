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
        return true;
    }

    framemetadata_sp frameMeta = frame->getMetadata();
    FrameMetadata::FrameType fType = frameMeta->getFrameType();

    auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(frameMeta);
    auto height = rawMetadata->getHeight();
    auto width = rawMetadata->getWidth();
    auto st = rawMetadata->getStep();

    unsigned char* rgba_buffer = (unsigned char*)frame->data(); 

    unsigned char* rgb_buffer = new unsigned char[width * height * 3];
    for (int i = 0; i < width * height; i++) {
        rgb_buffer[3 * i + 0] = rgba_buffer[4 * i + 0];
        rgb_buffer[3 * i + 1] = rgba_buffer[4 * i + 1];
        rgb_buffer[3 * i + 2] = rgba_buffer[4 * i + 2];
    }

    jpeg_compress_struct cinfo;
    jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    std::ofstream outfile("/home/developer/" + std::to_string(mDetail->count) + ".jpeg", std::ios::out | std::ios::binary);
    if (!outfile) {
        std::cerr << "Error: could not open output file" << std::endl;
        return 1;
    }
    mDetail->count = mDetail->count + 1;

    // jpeg_stdio_dest(&cinfo, outfile.rdbuf());
    // jpeg_stdio_dest(&cinfo, std::FILE*(outfile));
    // jpeg_stdio_dest(&cinfo, fdopen(outfile.rdbuf()->filedesc(), "wb"));
    // int fd = outfile.rdbuf()->native_handle();
    // FILE *file = fdopen(fd, "wb");
    // int fd = fileno(outfile);
    // FILE* file = fdopen(fd, "wb");
    // if (file == nullptr)
    // {
    //     std::cerr << "Error: could not obtain FILE* from file descriptor" << std::endl;
    //     return 1;
    // }


    ///fjgf
    int output_fd = outfile.rdbuf()->fd();

    // Convert the file descriptor to a FILE* pointer
    FILE* output_file = fdopen(output_fd, "wb");
    jpeg_stdio_dest(&cinfo, output_file);

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 80, TRUE);

    // Compress the RGB frame to JPEG format
    jpeg_start_compress(&cinfo, TRUE);

    JSAMPROW row_pointer[1];
    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer[0] = &rgb_buffer[cinfo.next_scanline * width * 3];
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);

    // Cleanup
    jpeg_destroy_compress(&cinfo);
    delete[] rgb_buffer;
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