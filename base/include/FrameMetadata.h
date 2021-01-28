#pragma once

#include <boost/shared_ptr.hpp>
#include <opencv2/opencv.hpp>
#include "CommonDefs.h"
#include "AIPExceptions.h"

#define NOT_SET_NUM 999999

class RawImageMetadata;

class FrameMetadata {
public:	
	static size_t getPaddingLength(size_t length, size_t alignLength)
	{
		if (!alignLength)
		{
			return 0;
		}

		auto rem = length % alignLength;
		if (rem == 0)
		{
			return 0;
		}

		return (alignLength - rem);
	}

public:
	enum FrameType {
		GENERAL = 0,
		ENCODED_IMAGE,
		RAW_IMAGE,
		RAW_IMAGE_PLANAR,
		AUDIO,
		ARRAY,
		CHANGE_DETECTION,
		EDGEDEFECT_ANALYSIS_INFO, 
		PROPS_CHANGE,
		PAUSE_PLAY,
		COMMAND,
		H264_DATA,
		GPIO,
		APRA_LINES,
		LINE,
		ROI,
		DEFECTS_INFO,
		BMP_IMAGE
	};

	enum MemType
	{
		HOST = 1,
#ifdef APRA_CUDA_ENABLED
		HOST_PINNED = 2,
		CUDA_DEVICE = 3,
		DMABUF = 4
#endif
	};
		
	FrameMetadata(FrameType _frameType)
	{
		frameType = _frameType;
		memType = MemType::HOST;
		hint = "";
	}

	FrameMetadata(FrameType _frameType, std::string _hint)
	{
		frameType = _frameType;
		memType = MemType::HOST;
		hint = _hint;
	}

	FrameMetadata(FrameType _frameType, MemType _memType)
	{
		frameType = _frameType;
		memType = _memType;
		hint = "";
	}

	virtual ~FrameMetadata() {	}

	virtual void reset()
	{
		dataSize = NOT_SET_NUM;
	}

	virtual bool isSet() 
	{
		return true;
	}		

	FrameType getFrameType()
	{
		return frameType;
	}

	MemType getMemType()
	{
		return memType;
	}

	virtual size_t getDataSize()
	{
		return dataSize;
	}

	std::string getHint() { return hint; }	

	void setHint(std::string _hint) { hint = _hint; }
	void copyHint(FrameMetadata& metadata)
	{
		if(!hint.empty())
		{
			return;
		}

		auto _hint = metadata.getHint();
		if(_hint.empty())
		{
			return;
		}

		setHint(_hint);
	}

	void setData(FrameMetadata& metadata)
	{
		// dont set memType
		// assuming frameType is same so no need to set 

		// hint I am still undecided whether to copy or not
	}

protected:
	FrameType frameType;
	MemType memType;
	std::string hint;
	
	size_t dataSize = NOT_SET_NUM;
};
