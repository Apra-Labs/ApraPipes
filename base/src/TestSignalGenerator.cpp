#include "TestSignalGenerator.h"
#include "Module.h"
#include <cstdlib>
#include <stdint.h>

class TestSignalGenerator::Detail
{
public:
    Detail(TestSignalGeneratorProps& _props)  : mProps(_props)
    {
        
    }

    ~Detail()
    {
    }
public:
    TestSignalGeneratorProps mProps;

};

TestSignalGenerator::TestSignalGenerator(TestSignalGeneratorProps _props)
:Module(SOURCE,"TestSignalGenerator",_props)
{
    mDetail.reset(new Detail(_props));
}

TestSignalGenerator::~TestSignalGenerator(){};

bool TestSignalGenerator::validateOutputPins()
{
    return true;
}

bool TestSignalGenerator::init()
{
    if (!Module::init())
	{
		return false;
	}
	 
    size_t buffer_size = (640*360*3) >> 1;
    tempBuffer = (unsigned char*)malloc(buffer_size);

    return true;
}



bool TestSignalGenerator::produce()
{
    size_t read_size = (640*360*3)>>1;
    unsigned char* Y = tempBuffer;

//    for(int height = 0; height < 120; height++)
//    {
//        //Loop of rows
//        uint8_t shade = (uint8_t)(height%256);
//        memset(Y, shade, 640);
//        Y+=640;
//    }

        int start_shade = 0;
        int end_shade = 255;
        int steps = 120;
        int step = (end_shade - start_shade) / steps;
        int current_shade = start_shade;

        for(int height = 0; height < 120; height++)
        {
        //Loop of rows
        memset(Y, (uint8_t)current_shade, 640);
        Y += 640;

        // Update the shade value
        current_shade += step;
        if (current_shade > end_shade) {
            current_shade = start_shade;
        }
        }
        auto mPinId = getOutputPinIdByType(FrameMetadata::RAW_IMAGE_PLANAR);
        frame_container frames;
        auto frame = makeFrame(read_size);
        memcpy(frame->data(),tempBuffer,read_size);
        frames.insert(make_pair(mPinId, frame));
        send(frames);

        return true;  
}
bool TestSignalGenerator::term()
{
    return Module::term();
}
void TestSignalGenerator::setMetadata(framemetadata_sp& metadata)
{
    if (!metadata->isSet())
    {
        return;
    }
}