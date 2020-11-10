#pragma once

#include "FrameMetadata.h"
#include "opencv2/opencv.hpp"

class ROIMetadata : public FrameMetadata
{
public:
	ROIMetadata() : FrameMetadata(FrameType::ROI) {}

	static cv::Rect &deserialize(void *buffer)
	{
		return *(reinterpret_cast<cv::Rect *>(buffer));
	}

	static void copyROI(cv::Rect& roi, void* dst)
	{
		auto& dstROI = deserialize(dst);
		dstROI.x = roi.x;
		dstROI.y = roi.y;
		dstROI.width = roi.width;
		dstROI.height = roi.height;
	}

	static bool isValid(cv::Rect& roi)
	{
		return !(roi.x == 0 && roi.y == 0 && roi.width == 0 && roi.height == 0);
	}

	size_t getDataSize()
	{
		return 16;
	}

};