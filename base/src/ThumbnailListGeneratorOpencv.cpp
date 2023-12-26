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

    cv::Mat tempResizedImage;
    auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(frameMeta);
    auto height = rawMetadata->getHeight();
    auto width = rawMetadata->getWidth();
    auto st = rawMetadata->getStep();

    mDetail->count = mDetail->count + 1;
    tempResizedImage = cv::Mat(height, width, CV_8UC4, frame->data(), st); //// uncomment line here
    vector<uchar> buf;
    cv::imencode(".jpg", tempResizedImage, buf, mDetail->flags);
    std::ofstream file("/home/developer/" + std::to_string(mDetail->count) + ".jpeg", std::ios::binary);
    if (file.is_open())
    {
        file.write((const char *)&buf[0], buf.size());
        file.close();
        std::cout << "Frame data saved to file: " << std::endl;
    }
    else
    {
        LOG_ERROR << "Failed to Save file ";
    }

    // cv::imwrite("/home/developer/" + std::to_string(mDetail->count) + ".jpeg", tempResizedImage);

    // mDetail->mIImg.data = static_cast<uint8_t *>(frame->data());
    // LOG_ERROR << "Size Of Frame is ===================>>>>>>> " << mDetail->mIImg.size();
    // LOG_ERROR << "Width & Height is  " << mDetail->mIImg.size().width << " === " << mDetail->mIImg.size().height;

    // if(mDetail->mOutSize.width == -1 || mDetail->mOutSize.height == -1)
    // {
    //     mDetail->m_tempImage = cv::Mat(mDetail->m_height, mDetail->m_width, CV_8UC4, frame->data(), mDetail->m_step); //// uncomment line here
    //     cv::imwrite(mDetail->mProps.fileToStore, mDetail->m_tempImage);  //// uncomment line here
    // }
    // else
    // {
    //     cv::Mat dstMat2(mDetail->mOutSize.width, mDetail->mOutSize.height, CV_8UC4);
    //     cv::resize(mDetail->mIImg, dstMat2, mDetail->mOutSize);
    //     cv::imwrite(mDetail->mProps.fileToStore, dstMat2);
    // }
    // framemetadata_sp frameMeta = frame->getMetadata();
    // FrameMetadata::FrameType fType=frameMeta->getFrameType();

    return true;

    //// this works
    // cv::Mat img;
    // framemetadata_sp frameMeta = frame->getMetadata();

    // auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(frameMeta);
    // auto height = rawMetadata->getHeight();
    // auto width = rawMetadata->getWidth();
    // auto st = rawMetadata->getStep();
    // img =cv::Mat(height, width, CV_8UC4, frame->data(), st); //// uncomment line here
    // cv::imwrite("/home/developer/abc_res.jpeg" ,img);  //// uncomment line her
    // return true;
}

// bool ThumbnailListGenerator::processSOS(frame_sp &frame)
// {
//     LOG_ERROR << "COming Inside process SOS";
//     auto metadata = frame->getMetadata();
//     // mDetail->initMatImages(metadata);
// }

// bool ThumbnailListGenerator::shouldTriggerSOS()
// {
//     if(mDetail->enableSOS == true)
//     {
//         mDetail->enableSOS = false;
//         return true;
//     }
//     else
//     {
//         return false;
//     }
// }

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