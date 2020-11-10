#pragma once

#include "FrameMetadata.h"

class PausePlayMetadata: public FrameMetadata
{
public:
    PausePlayMetadata(): FrameMetadata(FrameType::PAUSE_PLAY)
    {
        dataSize = 1;
    }
};