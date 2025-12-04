#pragma once
#include "Module.h"
#include <string>

enum class OverlayType
{
    NONE = 0,
    FRAME_INDEX = 1,
    TIMESTAMP = 2,
    BOTH = 3
};

class TestSignalGeneratorProps : public ModuleProps
{
public:
    TestSignalGeneratorProps() {}
    TestSignalGeneratorProps(int _width, int _height)
        : width(_width), height(_height) {}

    ~TestSignalGeneratorProps() {}

    int width = 0;
    int height = 0;
    
    // Overlay configuration
    OverlayType overlayType = OverlayType::NONE;
    std::string overlayFgColor = "00FF00";  // Green
    std::string overlayBgColor = "000000";  // Black
    std::string timestampFormat = "%H:%M:%S";  // hh:mm:ss
    bool timestampAppendMilliseconds = true;  // Append .nnn to timestamp
    int overlayX = -1;  // -1 for auto-center
    int overlayY = -1;  // -1 for vertical middle
    double overlayFontSize = -1.0;  // -1 for auto-sizing

private:
    friend class boost::serialization::access;

    template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar &boost::serialization::base_object<ModuleProps>(*this);
        ar &width;
        ar &height;
        ar &overlayType;
        ar &overlayFgColor;
        ar &overlayBgColor;
        ar &timestampFormat;
        ar &timestampAppendMilliseconds;
        ar &overlayX;
        ar &overlayY;
        ar &overlayFontSize;
    }
};

class TestSignalGenerator : public Module
{
public:
    TestSignalGenerator(TestSignalGeneratorProps _props);
    ~TestSignalGenerator();

    bool init();
    bool term();
    void setProps(TestSignalGeneratorProps &props);
    TestSignalGeneratorProps getProps();
 
protected:
    bool produce();
    bool validateOutputPins();
    void setMetadata(framemetadata_sp &metadata);
    bool handlePropsChange(frame_sp &frame);


private:
    class Detail;
    boost::shared_ptr<Detail> mDetail;
    size_t outputFrameSize;
    framemetadata_sp mOutputMetadata;
    std::string mOutputPinId;
};
