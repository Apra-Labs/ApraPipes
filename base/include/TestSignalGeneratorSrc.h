#pragma once
#include "Module.h"

class TestSignalGeneratorProps : public ModuleProps
{
public:
    TestSignalGeneratorProps()
    {
    }
    TestSignalGeneratorProps(int _width, int _height)
    {
        width = _width;
        height = _height;
    }
    ~TestSignalGeneratorProps()
    {
    }
    int width;
    int height;
};

class TestSignalGenerator : public Module
{
public:
    TestSignalGenerator(TestSignalGeneratorProps _props);
    ~TestSignalGenerator();
    bool init();
    bool term();

protected:
    bool produce();
    bool validateOutputPins();
    void setMetadata(framemetadata_sp &metadata);

private:
    class Detail;
    boost::shared_ptr<Detail> mDetail;
};