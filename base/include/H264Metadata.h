#pragma once

#include "FrameMetadata.h"

class H264Metadata : public FrameMetadata
{
public:
	H264Metadata() : FrameMetadata(FrameType::H264_DATA) {}
	H264Metadata(int _width, int _height) : FrameMetadata(FrameType::H264_DATA) , width(_width), height(_height)
	{
	}
	H264Metadata(int _width, int _height, int _gop_size, int _max_b_frames) : FrameMetadata(FrameType::H264_DATA), width(_width), height(_height), gop_size(_gop_size),max_b_frames(_max_b_frames)
	{
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
	void setData(H264Metadata& metadata)
	{
		FrameMetadata::setData(metadata);

		width = metadata.width;
		height = metadata.height;
		direction = metadata.direction;
		mp4Seek = metadata.mp4Seek;
		//setDataSize();
	}
	bool direction = true;
	bool mp4Seek = false;
protected:
	void initData(int _width, int _height, MemType _memType = MemType::HOST)
	{
		width = _width;
		height = _height;
	}
	// https://docs.opencv.org/4.1.1/d3/d63/classcv_1_1Mat.html
	int width = NOT_SET_NUM;
	int height = NOT_SET_NUM;
	int gop_size = NOT_SET_NUM;
	int max_b_frames = 0;

};