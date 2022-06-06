#include "ArgusStats.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include <opencv2/imgcodecs.hpp>
#include "Utils.h"
#include "StatsResult.h"
#include "DMAFDWrapper.h"
#include <iostream>
#include <fstream>

#define WIDTH 800
#define HEIGHT 800

class ArgusStats::Detail
{
public:
    Detail(ArgusStatsProps &_props) : props(_props)
    {
    }
    ~Detail() {}

public:
    size_t mFrameLength;
    framemetadata_sp mOutputMetadata;
    std::string mOutputPinId;
    int currPixVal = 0;
    uint8_t imgCount = 0;
    bool saveImage = false;

private:
    ArgusStatsProps props;
};

ArgusStats::ArgusStats(ArgusStatsProps _props) : Module(TRANSFORM, "ArgusStats", _props), props(_props), mFrameType(FrameMetadata::GENERAL) // yash GENERAL
{
    mDetail.reset(new Detail(_props));
}

ArgusStats::~ArgusStats() {}

bool ArgusStats::validateInputPins()
{
    if (getNumberOfInputPins() != 1)
    {
        LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
        return false;
    }

    framemetadata_sp metadata = getFirstInputMetadata();
    // FrameMetadata::FrameType frameType = metadata->getFrameType();
    // if (frameType != FrameMetadata::RAW_IMAGE)
    // {
    //     LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be RAW_IMAGE. Actual<" << frameType << ">";
    //     return false;
    // }

    return true;
}

bool ArgusStats::validateOutputPins()
{
    if (getNumberOfOutputPins() != 1)
    {
        LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
        return false;
    }
    return true;
}

void ArgusStats::addInputPin(framemetadata_sp &metadata, string &pinId)
{
    LOG_ERROR << "Adding Input Pins";
    Module::addInputPin(metadata, pinId);
    mDetail->mOutputMetadata = framemetadata_sp(new StatsResult());
    mDetail->mOutputMetadata->copyHint(*metadata.get());
    mDetail->mOutputPinId = addOutputPin(mDetail->mOutputMetadata);
    LOG_ERROR << "Done Adding Output Pins";
}

std::string ArgusStats::addOutputPin(framemetadata_sp &metadata)
{
    return Module::addOutputPin(metadata);
}

bool ArgusStats::init()
{
    return Module::init();
}

bool ArgusStats::term()
{
    return Module::term();
}

bool ArgusStats::process(frame_container &frames)
{
    uint8_t* saveFrame = NULL,  *saveFrame1 = NULL;
    // LOG_ERROR << "Coming Inside Process ";
    auto frame = frames.cbegin()->second;
    auto inpMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(frame->getMetadata());
    // LOG_ERROR << "Step is " << inpMetadata->getStep(0);
    // LOG_ERROR << "Hint of the Camera " << frame->getMetadata()->getHint();
    // LOG_ERROR << "Frame Size is" << frame->size();
    if (isFrameEmpty(frame))
    {
        LOG_ERROR << "Frame is Empty";
        return true;
    }
    // if(!frame){
    //     LOG_ERROR << "Frame is deleted";
    //     return true;
    // }
    // if(frame->data() == NULL){
    //     LOG_ERROR << "Frame data is deleted";
    //     return true;
    // }
    // void* frameCopy = malloc(frame->size());
    // if(frameCopy == NULL) {
    //     LOG_ERROR << "Memory allocation failed";
    //     return true;
    // }
    // // auto frameCopy = makeFrame(frame->size());
    // memcpy(frameCopy, frame->data(), frame->size());

    uint8_t min = 255;
    uint8_t max = 0;
    uint64_t saturatedPixel = 0;
    uint8_t saturatedValue = 250;
    if(mDetail->imgCount < 100){
        mDetail->imgCount++;
    }
    if(mDetail->imgCount == 100 && !mDetail->saveImage){
        saveFrame = (uint8_t*) malloc(HEIGHT*WIDTH);
        saveFrame1 = saveFrame;
    }
    

    // if (isFrameEmpty(frame))
    // {
    //     LOG_ERROR << "Frame is Empty";
    //     return true;
    // }    -
    auto inpPtr = static_cast<uint8_t *>(static_cast<DMAFDWrapper *>(frame->data())->getHostPtr());
    // auto inpPtr = static_cast<uint8_t *>(frame->data());
    // printf("InputPtr is ================> %x == %x\n", inpPtr,frame->data());
    // printf("i:%04d j:%04d size:%lu\n",i,j,frame->size());
    // LOG_ERROR << "Frame Data is " << frame->data();
    for (auto i = 0; i < HEIGHT; i++)
    {
        auto inpPtr1 = inpPtr + i * inpMetadata->getStep(0);
        for (auto j = 0; j < WIDTH; j++)
        {
            // LOG_ERROR << *inpPtr1;
            // printf("Input Ptr1 is ====================>%08x\n", inpPtr1);
            // printf("Input Ptr1 Value is ====================>%02x: %02x: %0d\n", *inpPtr1, min, (*inpPtr1 < min));

            
            // LOG_ERROR <<"Value Of input ptr 1 is " <<inpPtr1;
            // LOG_ERROR << "Value of inputPtr is  " <<inpPtr;
            // printf("i:%04d j:%04d size:%lu\n",i,j,frame->size());
            uint8_t pixelData = (*inpPtr1);
            if(saveFrame != NULL){
                *saveFrame1++ = pixelData;
            }
            if (pixelData < min)
            {
                min = pixelData;
            }
            if (pixelData > max)
            {
                max = pixelData;
            }
            if (pixelData >= props.saturatedPixel)
            {
                saturatedPixel++;
            }
            inpPtr1++;
        }
    }
    if(saveFrame != NULL){
        mDetail->saveImage = true;
        auto mode = std::ios::out | std::ios::binary;

        auto res = true;

        std::ofstream file("image_stats.raw", mode);
        // LOG_ERROR << "File Name " << fileName.c_str() << "Data Size is " << dataSize;

        if (file.is_open())
        {
            file.write((const char *)saveFrame, (WIDTH*HEIGHT));

            res = !file.bad() && !file.eof() && !file.fail();

            file.close();
        }
        free(saveFrame);
        saveFrame=NULL;
        saveFrame1=NULL;
        LOG_ERROR << "Frame Saved = " << res;
    }
    // free(frameCopy);
    // frameCopy = NULL;
    // LOG_ERROR << "Calculation Done";
    // LOG_ERROR << "Min Value " << min << "Max Value " << max ;

    LOG_ERROR << "Saturated PIXEL COUNT ======================================>" << to_string(saturatedPixel);
    uint8_t SaturatedPercentage = (saturatedPixel * 100.0) / (WIDTH * HEIGHT);
    // LOG_ERROR << "Saturated PERCENTAGE ===============================>" << to_string(SaturatedPercentage);

    auto out = StatsResult(min, max, SaturatedPercentage);
    auto statFrame = makeFrame(out.getSerializeSize());
    if(isFrameEmpty(statFrame))
    {
        LOG_ERROR << "statFrame Size is  0";
        return true;
    }
    StatsResult::serialize(out, statFrame);
    frames.insert(make_pair(mDetail->mOutputPinId, statFrame));
    send(frames);
    return true;
}

void ArgusStats::setMetadata(framemetadata_sp &metadata)
{
    if (!metadata->isSet())
    {
        return;
    }
    mDetail->mOutputMetadata = framemetadata_sp(new StatsResult());
}

bool ArgusStats::processSOS(frame_sp &frame)
{
    auto metadata = frame->getMetadata();
    setMetadata(metadata);
    return true;
}
