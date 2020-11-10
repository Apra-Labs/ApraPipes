#pragma once

#include "FrameMetadata.h"
#include "opencv2/opencv.hpp"

class LineMetadata : public FrameMetadata
{
public:
	LineMetadata() : FrameMetadata(FrameType::LINE) {}

	static cv::Vec4i &deserialize(void *buffer)
	{
		return *(reinterpret_cast<cv::Vec4i *>(buffer));
	}

	static void copyLine(cv::Vec4i& line, void* dst)
	{
		auto& dstLine = deserialize(dst);
		dstLine[0] = line[0];
		dstLine[1] = line[1];
		dstLine[2] = line[2];
		dstLine[3] = line[3];
	}

	static bool isValid(cv::Vec4i& line)
	{
		return !(line[0] == 0 && line[1] == 0 && line[2] == 0 && line[3] == 0);
	}

	size_t getDataSize()
	{
		return 16;
	}

};