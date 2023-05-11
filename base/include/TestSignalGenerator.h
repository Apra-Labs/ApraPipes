#pragma once
#include "Module.h"

class TestSignalGeneratorProps : public ModuleProps
{
public:
    TestSignalGeneratorProps()
    {
    }
    ~TestSignalGeneratorProps()
    {
    }

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
    void setMetadata(framemetadata_sp& metadata);
private:
	class Detail;
	boost::shared_ptr<Detail> mDetail;
    unsigned char* tempBuffer;
    
};