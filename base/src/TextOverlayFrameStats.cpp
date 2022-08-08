#include "TextOverlayFrameStats.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "AIPExceptions.h"
#include "DMAFDWrapper.h"
#include "DMAUtils.h"
#include "nvbuf_utils.h"
#include "cudaEGL.h"
#include <Argus/Argus.h>
#include <deque>
#include "StatsResult.h"
#include "npp.h"
#include "string"

#define WIDTH 800
#define HEIGHT 800
#define CAMERA_HINT1 "Camera1"
#define CAMERA_HINT2 "Camera2"
#define CAMERA_HINT3 "Camera3"
class TextOverlayFrameStats::Detail
{
public:
    Detail(TextOverlayFrameStatsProps &_props) : props(_props)
    {
    }

    ~Detail()
    {
    }

    // bool assignValues(frame_sp frame1, frame_sp frame2, frame_sp frame3)
    // {
    //     StatsResult res1;
    //     StatsResult::deSerialize(res1, frame1);
    //     cam1min = res1.min;
    //     cam1max = res1.max;
    //     cam1saturatedPixel = res1.saturationPercentage;
    //     StatsResult res2;
    //     StatsResult::deSerialize(res2, frame2);
    //     cam2min = res2.min;
    //     cam2max = res2.max;
    //     cam2saturatedPixel = res2.saturationPercentage;
    //     StatsResult res3;
    //     StatsResult::deSerialize(res3, frame3);
    //     cam3min = res3.min;
    //     cam3max = res3.max;
    //     cam3saturatedPixel = res3.saturationPercentage;
    //     LOG_ERROR << "Min Cam1 " << res1.min << "Min Cam2 " << res2.min << "Min Cam3" << res3.min;
    //     return true;
    // }

    void initMatImages()
    {
        mOutputImg = Utils::getMatHeader(FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata));
    }

    void compute()
    {
        cv::putText(mOutputImg, "Camera2", cv::Point(400, 0), cv::FONT_HERSHEY_COMPLEX_SMALL, 2.0, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
        cv::putText(mOutputImg, "Camera1", cv::Point(200, 0), cv::FONT_HERSHEY_COMPLEX_SMALL, 2.0, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
        cv::putText(mOutputImg, "Camera3", cv::Point(600, 0), cv::FONT_HERSHEY_COMPLEX_SMALL, 2.0, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
        cv::putText(mOutputImg, "Min", cv::Point(0, 300), cv::FONT_HERSHEY_COMPLEX_SMALL, 2.0, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
        cv::putText(mOutputImg, "Max", cv::Point(0, 600), cv::FONT_HERSHEY_COMPLEX_SMALL, 2.0, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
        cv::putText(mOutputImg, "Saturated_Pixel", cv::Point(0, 900), cv::FONT_HERSHEY_COMPLEX_SMALL, 2.0, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
        // cv::putText(mDetail->mOutputImg, "dyfgyfg", cv::Point(0, 0), cv::FONT_HERSHEY_COMPLEX_SMALL, 2.0, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        // cv::putText(mDetail->mOutputImg, "dyfgyfg", cv::Point(0, 0), cv::FONT_HERSHEY_COMPLEX_SMALL, 2.0, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        // cv::putText(mDetail->mOutputImg, "dyfgyfg", cv::Point(0, 0), cv::FONT_HERSHEY_COMPLEX_SMALL, 2.0, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        // cv::putText(mDetail->mOutputImg, "dyfgyfg", cv::Point(0, 0), cv::FONT_HERSHEY_COMPLEX_SMALL, 2.0, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        // cv::putText(mDetail->mOutputImg, "dyfgyfg", cv::Point(0, 0), cv::FONT_HERSHEY_COMPLEX_SMALL, 2.0, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        // cv::putText(mDetail->mOutputImg, "dyfgyfg", cv::Point(0, 0), cv::FONT_HERSHEY_COMPLEX_SMALL, 2.0, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        // cv::putText(mDetail->mOutputImg, "dyfgyfg", cv::Point(0, 0), cv::FONT_HERSHEY_COMPLEX_SMALL, 2.0, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        // cv::putText(mDetail->mOutputImg, "dyfgyfg", cv::Point(0, 0), cv::FONT_HERSHEY_COMPLEX_SMALL, 2.0, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        // cv::putText(mDetail->mOutputImg, "dyfgyfg", cv::Point(0, 0), cv::FONT_HERSHEY_COMPLEX_SMALL, 2.0, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        // cv::putText(mDetail->mOutputImg, "dyfgyfg", cv::Point(0, 0), cv::FONT_HERSHEY_COMPLEX_SMALL, 2.0, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    }

private:
    int height;
    size_t rowSize;
    ImageMetadata::ImageType inputImageType;
    ImageMetadata::ImageType outputImageType;
    int inputChannels;
    int outputChannels;
    Npp32f *src[4];
    NppiSize srcSize[4];
    int srcPitch[4];
    NppiSize dstSize[4];
    int dstPitch[4];

    size_t dstNextPtrOffset[4];
    size_t srcNextPtrOffset[4];
    size_t mHeight[4];
    size_t mRowSize[4];

    

    TextOverlayFrameStatsProps props;
    
    // StatsResult res1;
    // StatsResult res2;
    // StatsResult res3;

public:
    bool isSet = false;
    FrameMetadata::FrameType mFrameType;
    framemetadata_sp mOutputMetadata;
    std::string mOutputPinId;
    cv::Mat mOutputImg;
    uint8_t cam1min;
    uint8_t cam1max;
    uint8_t cam1saturatedPixel;

    uint8_t cam2min;
    uint8_t cam2max;
    uint8_t cam2saturatedPixel;

    uint8_t cam3min;
    uint8_t cam3max;
    uint8_t cam3saturatedPixel;

};

TextOverlayFrameStats::TextOverlayFrameStats(TextOverlayFrameStatsProps _props) : Module(TRANSFORM, "TextOverlayFrameStats", _props), props(_props), mFrameChecker(0)
{
    mDetail.reset(new Detail(_props));
}

TextOverlayFrameStats::~TextOverlayFrameStats() {}

bool TextOverlayFrameStats::validateInputPins()
{
    framemetadata_sp metadata = getFirstInputMetadata();

    FrameMetadata::FrameType frameType = metadata->getFrameType();
    // if (frameType != FrameMetadata::STATS)
    // {
    //     LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be STATS. Actual<" << frameType << ">";
    //     return false;
    // }
    return true;
}

bool TextOverlayFrameStats::validateOutputPins()
{
    if (getNumberOfOutputPins() != 1)
    {
        LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
        return false;
    }

    framemetadata_sp metadata = getFirstOutputMetadata();
    auto mOutputFrameType = metadata->getFrameType();
    if (mOutputFrameType != FrameMetadata::RAW_IMAGE && mOutputFrameType != FrameMetadata::RAW_IMAGE_PLANAR)
    {
        LOG_ERROR << "<" << getId() << ">::validateOutputPins input frameType is expected to be RAW_IMAGE or RAW_IMAGE_PLANAR. Actual<" << mOutputFrameType << ">";
        return false;
    }

    FrameMetadata::MemType memType = metadata->getMemType();
    if (memType != FrameMetadata::MemType::HOST)
    {
        LOG_ERROR << "<" << getId() << ">::validateOutputPins input memType is expected to be DMABUF. Actual<" << memType << ">";
        return false;
    }

    return true;
}

void TextOverlayFrameStats::addInputPin(framemetadata_sp &metadata, string &pinId)
{
    Module::addInputPin(metadata, pinId);
    if (!mDetail->isSet)
    {
        // LOG_ERROR << "Setting Output Metadata";
        mDetail->mOutputMetadata = boost::shared_ptr<FrameMetadata>(new RawImageMetadata(WIDTH, HEIGHT, ImageMetadata::ImageType::MONO, CV_8UC1, 0, CV_8U, FrameMetadata::HOST, true));
        // mOutputMetadata = framemetadata_sp(new RawImageMetadata(FrameMetadata::MemType::DMABUF));
        mDetail->mOutputPinId = addOutputPin(mDetail->mOutputMetadata);
        mDetail->isSet = true;
        // LOG_ERROR << "Done Output Metadata";
    }
}
bool TextOverlayFrameStats::init()
{
    if (!Module::init())
    {
        return false;
    }

    return true;
}

bool TextOverlayFrameStats::term()
{
    return Module::term();
}

bool TextOverlayFrameStats::process(frame_container &frames)
{
    frame_sp frame1, frame2, frame3;

    for (auto const &element : frames)
    {
        auto frame = element.second;
        if (frame->getMetadata()->getHint() == CAMERA_HINT1)
        {
            frame1 = frame; // contain stats
        }
        else if (frame->getMetadata()->getHint() == CAMERA_HINT2)
        {
            frame2 = frame;
        }
        else if (frame->getMetadata()->getHint() == CAMERA_HINT3)
        {
            frame3 = frame;
        }
    }
    // LOG_ERROR << "Before Deserialize";
    StatsResult res1;
    StatsResult::deSerialize(res1, frame1);
    // LOG_ERROR << "Adter Deserialize0";
    mDetail->cam1min = res1.min;
    mDetail->cam1max = res1.max;
    mDetail->cam1saturatedPixel = res1.saturationPercentage;


    StatsResult res2;
    StatsResult::deSerialize(res2, frame2);
    // LOG_ERROR << "Adter Deserialize0";
    mDetail->cam2min = res2.min;
    mDetail->cam2max = res2.max;
    mDetail->cam2saturatedPixel = res2.saturationPercentage;


    StatsResult res3;
    StatsResult::deSerialize(res3, frame3);
    // LOG_ERROR << "Adter Deserialize0";
    mDetail->cam3min = res3.min;
    mDetail->cam3max = res3.max;
    mDetail->cam3saturatedPixel = res3.saturationPercentage;


    // LOG_ERROR << "Camera1 Min" << mDetail->cam1min << "Camera2 Max " << mDetail->cam1max;
    
    // mDetail->assignValues(frame1, frame2, frame3);
    // LOG_ERROR << "Frame Size is " << mDetail->mOutputMetadata->getDataSize();
    auto outFrame = makeFrame(mDetail->mOutputMetadata->getDataSize(), mDetail->mOutputPinId);
    if(isFrameEmpty(outFrame))
    {
        LOG_ERROR << "outFrame Size is 0";
        return true;
    }
    memset(outFrame->data(), 0, outFrame->size());

    mDetail->mOutputImg.data = static_cast<uint8_t *>(outFrame->data());

    // mDetail->mOutputImg.data = static_cast<uint8_t *>(static_cast<DMAFDWrapper *>(outFrame->data())->getHostPtr());
    // mDetail->mOutputImg.data = static_cast<uint8_t *>(static_cast<DMAFDWrapper *>(outFrame->data())->getHostPtr());
    // mDetail->compute();
    cv::putText(mDetail->mOutputImg, "Min", cv::Point(200, 100), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
    cv::putText(mDetail->mOutputImg, "Max", cv::Point(400, 100), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
    cv::putText(mDetail->mOutputImg, "SatPix", cv::Point(600, 100), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);

    cv::putText(mDetail->mOutputImg, "Cam1", cv::Point(10, 300), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
    cv::putText(mDetail->mOutputImg, "Cam2", cv::Point(10, 500), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
    cv::putText(mDetail->mOutputImg, "Cam3", cv::Point(10, 700), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);

    cv::putText(mDetail->mOutputImg, to_string(mDetail->cam1min), cv::Point(200, 300), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
    cv::putText(mDetail->mOutputImg, to_string(mDetail->cam1max), cv::Point(400, 300), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
    cv::putText(mDetail->mOutputImg, to_string(mDetail->cam1saturatedPixel) + " %", cv::Point(600, 300), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);

    cv::putText(mDetail->mOutputImg, to_string(mDetail->cam2min), cv::Point(200, 500), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
    cv::putText(mDetail->mOutputImg, to_string(mDetail->cam2max), cv::Point(400, 500), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
    cv::putText(mDetail->mOutputImg, to_string(mDetail->cam2saturatedPixel) + " %", cv::Point(600, 500), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);


    cv::putText(mDetail->mOutputImg, to_string(mDetail->cam3min), cv::Point(200, 700), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
    cv::putText(mDetail->mOutputImg, to_string(mDetail->cam3max), cv::Point(400, 700), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
    cv::putText(mDetail->mOutputImg, to_string(mDetail->cam3saturatedPixel) + " %", cv::Point(600, 700), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);


    // LOG_ERROR << "After Compute";
    frames.insert(make_pair(mDetail->mOutputPinId, outFrame));
    send(frames);
    // boost::this_thread::sleep_for(boost::chrono::seconds(1));
    return true;
}

bool TextOverlayFrameStats::processSOS(frame_sp &frame)
{
    auto metadata = frame->getMetadata();
    setMetadata(metadata);
    mFrameChecker++;

    return true;
}

void TextOverlayFrameStats::setMetadata(framemetadata_sp &metadata)
{
    if (!metadata->isSet())
    {
        return;
    }
    mDetail->mOutputMetadata = boost::shared_ptr<FrameMetadata>(new RawImageMetadata(WIDTH, HEIGHT, ImageMetadata::ImageType::MONO, CV_8UC1, 0, CV_8U, FrameMetadata::HOST, true));
    mDetail->initMatImages();
}

bool TextOverlayFrameStats::shouldTriggerSOS()
{
    return (mFrameChecker == 0 || mFrameChecker == 1 || mFrameChecker == 2);
}

bool TextOverlayFrameStats::processEOS(string &pinId)
{
    mFrameChecker = 0;
    return true;
}