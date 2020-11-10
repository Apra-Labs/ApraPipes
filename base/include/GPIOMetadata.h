#pragma once

#include "FrameMetadata.h"

class GPIOMetadata: public FrameMetadata
{
public:   
	GPIOMetadata(): FrameMetadata(FrameType::GPIO) 
    {
        dataSize = 1;
    }

};