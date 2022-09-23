#pragma once

#include "FrameMetadata.h"

class Mp4VideoMetadata : public FrameMetadata
{
public:
	Mp4VideoMetadata() : FrameMetadata(FrameType::MP4_VIDEO_METADATA) {}
	Mp4VideoMetadata(MemType _memType) : FrameMetadata(FrameType::MP4_VIDEO_METADATA, _memType) {}

	Mp4VideoMetadata(std::string _version) : FrameMetadata(FrameType::MP4_VIDEO_METADATA, FrameMetadata::HOST)
	{
		version = _version;
	}

	void reset()
	{
		FrameMetadata::reset();
	}

	bool isSet()
	{
		return !version.empty();
	}

	void setData(std::string& _version)
	{
		version = _version;
	}

	std::string getVersion()
	{
		return version;
	}

protected:

	void initData(std::string _version, MemType _memType = MemType::HOST)
	{
		version = _version;
	}

	std::string version = "";
};