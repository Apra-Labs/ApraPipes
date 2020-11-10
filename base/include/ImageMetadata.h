#pragma once

#include "opencv2/core/cvdef.h"
#include <string>
#include "AIPExceptions.h"

class ImageMetadata
{
public:
	enum ImageType
	{
		UNSET = 0,
		MONO = 1,
		BGR, // Interleaved
		BGRA, // Interleaved
		RGB, // Interleaved
		RGBA, // Interleaved
		YUV411_I, // Interleaved
        YUV444, // Planar
		YUV420 // Planar		
	};

	static size_t getElemSize(int depth)
	{
		size_t elemSize = 1;
		switch (depth)
		{
		case CV_8U:
		case CV_8S:
			elemSize = 1;
			break;
		case CV_16U:
		case CV_16S:
			elemSize = 2;
			break;
		case CV_32S:
		case CV_32F:
			elemSize = 4;
			break;
		case CV_64F:
			elemSize = 8;
			break;
		default:
			auto msg = "Unknown depth type<" + std::to_string(depth) + ">";
			throw AIPException(AIP_NOTIMPLEMENTED, msg);
		}

		return elemSize;
	}
};