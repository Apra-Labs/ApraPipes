#pragma once

#include "Frame.h"

class ExtFrame : public Frame
{
public:
	ExtFrame(void* data, size_t size) : Frame(data, size, std::shared_ptr<FrameFactory>())
	{

	}

	virtual ~ExtFrame()
	{
		
	}
};