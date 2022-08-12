#pragma once

#include "FrameMetadata.h"

class H264Metadata : public FrameMetadata
{
public:
	H264Metadata() : FrameMetadata(FrameType::H264_DATA) {}
	H264Metadata(int _width, int _height) : FrameMetadata(FrameType::H264_DATA) , width(_width), height(_height)
	{
		width = _width;
		height = _height;
	}

	void reset()
	{
		FrameMetadata::reset();
		width = NOT_SET_NUM;
		height = NOT_SET_NUM;
	}

	bool isSet()
	{
		return width != NOT_SET_NUM;
	}

	int getWidth()
	{
		return width;
	}

	int getHeight()
	{
		return height;
	}

protected:
	// https://docs.opencv.org/4.1.1/d3/d63/classcv_1_1Mat.html
	int width = NOT_SET_NUM;
	int height = NOT_SET_NUM;
};