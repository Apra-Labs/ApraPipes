#include "TestSignalGeneratorSrc.h"
#include "Module.h"
#include <cstdlib>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

class TestSignalGenerator::Detail
{
public:
    Detail(TestSignalGeneratorProps &_props)
        : mProps(_props), start_shade(0), end_shade(255), current_shade(start_shade), frameCounter(0) {}

    ~Detail() {}

    // Calculate optimal font size based on image dimensions
    double calculateFontSize()
    {
        if (mProps.overlayFontSize > 0)
        {
            return mProps.overlayFontSize;
        }
        // Auto-size: use 1/20th of image height as base
        return mProps.height / 600.0;  // Scale factor for readable text
    }

    // Format current timestamp with milliseconds
    std::string formatTimestamp()
    {
        auto now = std::chrono::system_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::tm tm;
        localtime_s(&tm, &t);  
        char buffer[64];
        std::strftime(buffer, sizeof(buffer), mProps.timestampFormat.c_str(), &tm);
        
        // Append milliseconds if configured to do so
        if (mProps.timestampAppendMilliseconds)
        {
            char finalBuffer[128];
            sprintf_s(finalBuffer, sizeof(finalBuffer), "%s.%03d", buffer, (int)ms.count());
            return std::string(finalBuffer);
        }
        
        return std::string(buffer);
    }

    // Render overlay text on the frame
    void renderOverlay(uint8_t* frameData)
    {
        if (mProps.overlayType == OverlayType::NONE)
        {
            return;
        }

        // Convert YUV420 to BGR for OpenCV processing
        cv::Mat yuvImg(mProps.height * 3 / 2, mProps.width, CV_8UC1, frameData);
        cv::Mat bgrImg;
        cv::cvtColor(yuvImg, bgrImg, cv::COLOR_YUV2BGR_I420);

        // Parse colors
        int r, g, b, backR, backG, backB;
        sscanf_s(mProps.overlayFgColor.c_str(), "%02x%02x%02x", &r, &g, &b);
        sscanf_s(mProps.overlayBgColor.c_str(), "%02x%02x%02x", &backR, &backG, &backB);

        double fontSize = calculateFontSize();
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        int thickness = std::max(1, (int)(fontSize * 2));

        std::vector<std::string> lines;
        std::vector<std::string> refLines; // For stable layout calculation

        // Prepare text and reference lines
        if (mProps.overlayType == OverlayType::FRAME_INDEX)
        {
            char buffer[32];
            sprintf_s(buffer, sizeof(buffer), "Frame:%llu", frameCounter);
            lines.push_back(std::string(buffer));
            refLines.push_back("Frame: 0000000000"); // Max width ref
        }
        else if (mProps.overlayType == OverlayType::TIMESTAMP)
        {
            lines.push_back(formatTimestamp());
            refLines.push_back("00:00:00.000"); // Max width ref
        }
        else if (mProps.overlayType == OverlayType::BOTH)
        {
            char buffer[32];
            sprintf_s(buffer, sizeof(buffer), "Frame:%llu", frameCounter);
            lines.push_back(std::string(buffer));
            lines.push_back(formatTimestamp());
            
            refLines.push_back("Frame: 0000000000");
            refLines.push_back("00:00:00.000");
        }

        int maxWidth = 0;
        int lineHeight = 0;
        int baseline = 0;
        
        for (const auto& line : refLines)
        {
            cv::Size textSize = cv::getTextSize(line, fontFace, fontSize, thickness, &baseline);
            maxWidth = std::max(maxWidth, textSize.width);
            lineHeight = std::max(lineHeight, textSize.height);
        }

        int lineSpacing = 20;
        int totalHeight = (lineHeight * (int)lines.size()) + (lineSpacing * ((int)lines.size() - 1));

        
        int x = mProps.overlayX >= 0 ? mProps.overlayX : (mProps.width - maxWidth) / 2;
        
        
        int startY;
        if (mProps.overlayY >= 0)
        {
            startY = mProps.overlayY;
        }
        else
        {
            startY = (mProps.height - totalHeight) / 2 + lineHeight;
        }

    
        int padding = 10;
        int boxTop = startY - lineHeight;
        int boxBottom = boxTop + totalHeight + padding + baseline; 
        
        cv::Point topLeft(x - padding, boxTop - padding);
        cv::Point bottomRight(x + maxWidth + padding, boxBottom);
        
        cv::rectangle(bgrImg, topLeft, bottomRight, cv::Scalar(backB, backG, backR), cv::FILLED);

        int currentY = startY;
        for (const auto& line : lines)
        {
            cv::putText(bgrImg, line, cv::Point(x, currentY), 
                       fontFace, fontSize, cv::Scalar(b, g, r), thickness, cv::LINE_AA);
            currentY += lineHeight + lineSpacing;
        }

        // Convert back to YUV420
        cv::cvtColor(bgrImg, yuvImg, cv::COLOR_BGR2YUV_I420);
    }

    bool generate(frame_sp &frame)
    {
        auto frame_ptr = frame->data();
        uint8_t* x = static_cast<uint8_t*>(frame_ptr);

        // Generate gradient pattern
        for (int height = 0; height < mProps.height * 1.5; height++)
        {
            memset(x, current_shade, mProps.width);
            x += mProps.width;
            current_shade += 1;
            if (current_shade > end_shade)
            {
                current_shade = start_shade;
            }
        }

        renderOverlay(static_cast<uint8_t*>(frame_ptr));
        frameCounter++;

        return true;
    }

    void setProps(const TestSignalGeneratorProps &_props)
    {
        mProps = _props;
        reset();
    }
    
    void reset()
    {
        current_shade = start_shade;
        frameCounter = 0; 
    }

    TestSignalGeneratorProps mProps;
    uint8_t start_shade = 0;
    uint8_t end_shade = 255;
    uint8_t current_shade = 0;
    uint64_t frameCounter = 0;  
};

TestSignalGenerator::TestSignalGenerator(TestSignalGeneratorProps _props)
    : Module(SOURCE, "TestSignalGenerator", _props), outputFrameSize(0)
{
    mDetail.reset(new Detail(_props));
    mOutputMetadata = framemetadata_sp(new RawImagePlanarMetadata(_props.width, _props.height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));
    mOutputPinId = addOutputPin(mOutputMetadata);
}

TestSignalGenerator::~TestSignalGenerator()
{
    mDetail->~Detail();
}

bool TestSignalGenerator::validateOutputPins()
{
    if (getNumberOfOutputPins() != 1)
    {
        LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
        return false;
    }
    framemetadata_sp metadata = getFirstOutputMetadata();
    auto frameType = metadata->getFrameType();
    if (frameType != FrameMetadata::RAW_IMAGE_PLANAR)
    {
        LOG_ERROR << "<" << getId() << ">::validateOutputPins output frameType should be RAW_IMAGE_PLANAR. Actual<" << frameType << ">";
        return false;
    }

    return true;
}

bool TestSignalGenerator::init()
{
    if (!Module::init())
    {
        return false;
    }
    outputFrameSize = (getProps().width * getProps().height * 3) >> 1;

    return true;
}

bool TestSignalGenerator::produce()
{
    auto mPinId = getOutputPinIdByType(FrameMetadata::RAW_IMAGE_PLANAR);
    frame_container frames;
    frame_sp frame = makeFrame(outputFrameSize);
    mDetail->generate(frame);
    frames.insert(make_pair(mPinId, frame));
    send(frames);
    return true;
}

bool TestSignalGenerator::term()
{
    return Module::term();
}

void TestSignalGenerator::setMetadata(framemetadata_sp &metadata)
{
    if (!metadata->isSet())
    {
        return;
    }
}

bool TestSignalGenerator::handlePropsChange(frame_sp &frame)
{
    TestSignalGeneratorProps props;
    bool ret = Module::handlePropsChange(frame, props);
    mDetail->setProps(props);
    outputFrameSize = (props.width * props.height * 3) >> 1;
    return ret;
}

void TestSignalGenerator::setProps(TestSignalGeneratorProps &props)
{
    Module::addPropsToQueue(props);
}

TestSignalGeneratorProps TestSignalGenerator::getProps()
{
    return mDetail->mProps;
}
