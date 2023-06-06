#pragma once
#include "Module.h"

class TestSignalGeneratorProps : public ModuleProps
{
public:
    TestSignalGeneratorProps() {}
    TestSignalGeneratorProps(int _width, int _height)
        : width(_width), height(_height) {}

    ~TestSignalGeneratorProps() {}

    int width;
    int height;

private:
    friend class boost::serialization::access;

    template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar &boost::serialization::base_object<ModuleProps>(*this);
        ar &width;
        ar &height;
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
