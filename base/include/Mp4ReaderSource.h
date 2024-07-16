#pragma once

#include "Module.h"
#include <boost/filesystem.hpp>
class Mp4ReaderDetailAbs;
class Mp4ReaderDetailJpeg;
class Mp4ReaderDetailH264;

class Mp4ReaderSourceProps : public ModuleProps
{
public:
	Mp4ReaderSourceProps() : ModuleProps()
	{

	}

	Mp4ReaderSourceProps(std::string _videoPath, bool _parseFS, uint16_t _reInitInterval, bool _direction, bool _readLoop, bool _giveLiveTS, int _parseFSTimeoutDuration = 15, bool _bFramesEnabled = false) : ModuleProps()
	{
		/* About props:
			- videoPath - Path of a video from where the reading will start.
			- reInitInterval - Live Mode if reInitInterval > 0 i.e. reading a file as it gets written. We recheck the last file every reInitInterval time to see if it has been updated.
			- direction - Playback direction (fwd/bwd).
			- parseFS - Read the NVR format till infinity, if true. Else we read only one file.
			- readLoop - Read a single video in loop. It can not be used in conjuction with live mode (reInitInterval > 0) or NVR mode (parseFS = true) mode.
			- giveLiveTS - If enabled, gives live timestamps instead of recorded timestamps in the video files.
		*/

		if (reInitInterval < 0)
		{
			auto errMsg = "incorrect prop reInitInterval <" + std::to_string(reInitInterval) + ">";
			throw AIPException(AIP_FATAL, errMsg);
		}

		if (_readLoop && (_reInitInterval || _parseFS))
		{
			auto errMsg = "Incorrect parameters. Looping can not be coupled with retry feature or disk parsing. loop <" + std::to_string(_readLoop) +
				"> reInitInterval <" + std::to_string(reInitInterval) + "> parseFS <" + std::to_string(_parseFS) + ">";
			throw AIPException(AIP_FATAL, errMsg);
		}
		auto canonicalVideoPath = boost::filesystem::canonical(_videoPath);
		videoPath = canonicalVideoPath.string();
		parseFS = _parseFS;
		bFramesEnabled = _bFramesEnabled;
		direction = _direction;
		giveLiveTS = _giveLiveTS;
		if (_reInitInterval < 0)
		{
			throw AIPException(AIP_FATAL, "reInitInterval must be 0 or more seconds");
		}
		parseFSTimeoutDuration = _parseFSTimeoutDuration;
		readLoop = _readLoop;
		reInitInterval = _reInitInterval;

		//If the input file path is the full video path then its root dir will be skipDir else if the input path is only root dir path then it is directly assigned to skipDir.
		if (parseFS && boost::filesystem::path(videoPath).extension() == ".mp4")
		{
			skipDir = boost::filesystem::path(videoPath).parent_path().parent_path().parent_path().string();
		}
		else
		{
			skipDir = boost::filesystem::path(canonicalVideoPath).string();
		}
	}

	void setMaxFrameSizes(size_t _maxImgFrameSize, size_t _maxMetadataFrameSize)
	{
		biggerFrameSize = _maxImgFrameSize;
		biggerMetadataFrameSize = _maxMetadataFrameSize;
	}

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(videoPath) + sizeof(parseFS) + sizeof(skipDir) + sizeof(direction) + sizeof(parseFSTimeoutDuration) + sizeof(biggerFrameSize) + sizeof(biggerMetadataFrameSize) + sizeof(bFramesEnabled);
	}

	std::string skipDir = "./data/Mp4_videos";
	std::string videoPath = "";
	size_t biggerFrameSize = 600000;
	size_t biggerMetadataFrameSize = 60000;
	bool parseFS = true;
	bool direction = true;
	bool bFramesEnabled = false;
	uint16_t reInitInterval = 0;
	int parseFSTimeoutDuration = 15;
	bool readLoop = false;
	bool giveLiveTS = false;
private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar& boost::serialization::base_object<ModuleProps>(*this);
		ar& videoPath;
		ar& parseFS;
		ar& skipDir;
		ar& biggerFrameSize;
		ar& biggerMetadataFrameSize;
		ar& bFramesEnabled;
		ar& parseFSTimeoutDuration;
		ar& direction;
		ar& readLoop;
		ar& giveLiveTS;
	}
};

class Mp4ReaderSource : public Module
{
public:
	Mp4ReaderSource(Mp4ReaderSourceProps _props);
	virtual ~Mp4ReaderSource();
	bool init();
	bool term();
	Mp4ReaderSourceProps getProps();
	void setProps(Mp4ReaderSourceProps& props);
	std::string getOpenVideoPath();
	void setImageMetadata(std::string& pinId, framemetadata_sp& metadata);
	std::string addOutPutPin(framemetadata_sp& metadata);
	bool changePlayback(float speed, bool direction);
	bool getVideoRangeFromCache(std::string videoPath, uint64_t& start_ts, uint64_t& end_ts);
	bool randomSeek(uint64_t skipTS, bool forceReopen = false);
	bool refreshCache();
	std::map<std::string, std::pair<uint64_t, uint64_t>> getCacheSnapShot(); // to be used for debugging only
	double getOpenVideoFPS();
	double getOpenVideoDurationInSecs();
	int32_t getOpenVideoFrameCount();
	void setPlaybackSpeed(float _playbckSpeed);
	void getResolution(uint32_t& width, uint32_t& height)
	{
		width = mWidth;
		height = mHeight;
	}
protected:
	bool produce();
	bool validateOutputPins();
	bool handleCommand(Command::CommandType type, frame_sp& fame);
	bool handlePropsChange(frame_sp& frame);
	bool handlePausePlay(float speed, bool direction) override;
private:
	std::string h264ImagePinId;
	std::string encodedImagePinId;
	uint32_t mWidth = 0;
	uint32_t mHeight = 0;
	std::string metadataFramePinId;
	boost::shared_ptr<Mp4ReaderDetailAbs> mDetail;
	Mp4ReaderSourceProps props;
};