#pragma once

#include "Module.h"
#include <boost/filesystem.hpp>
class Mp4readerDetailAbs;
class Mp4readerDetailJpeg;
class Mp4readerDetailH264;

class Mp4ReaderSourceProps : public ModuleProps
{
public:
	Mp4ReaderSourceProps() : ModuleProps()
	{

	}

	Mp4ReaderSourceProps(std::string _videoPath, bool _parseFS, size_t _biggerFrameSize, size_t _biggerMetadataFrameSize, int _parseFSTimeoutDuration = 15, bool _bFramesEnabled = false) : ModuleProps()
	{
		biggerFrameSize = _biggerFrameSize;
		biggerMetadataFrameSize = _biggerMetadataFrameSize;
		videoPath = _videoPath;
		parseFS = _parseFS;
		bFramesEnabled = _bFramesEnabled;
		parseFSTimeoutDuration = _parseFSTimeoutDuration;
		if (parseFS)
		{
			skipDir = boost::filesystem::path(videoPath).parent_path().parent_path().parent_path().string();
		}

	}

	Mp4ReaderSourceProps(std::string _videoPath, bool _parseFS, int _parseFSTimeoutDuration = 15, bool _bFramesEnabled = false) : ModuleProps()
	{
		videoPath = _videoPath;
		parseFS = _parseFS;
		bFramesEnabled = _bFramesEnabled;
		parseFSTimeoutDuration = _parseFSTimeoutDuration;
		if (parseFS)
		{
			skipDir = boost::filesystem::path(videoPath).parent_path().parent_path().parent_path().string();
		}

	}

	Mp4ReaderSourceProps(std::string _videoPath, bool _parseFS, uint16_t _reInitInterval, bool _direction, bool _readLoop, int _parseFSTimeoutDuration = 15, bool _bFramesEnabled = false) : ModuleProps()
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
		skipDir = boost::filesystem::path(canonicalVideoPath).parent_path().parent_path().parent_path().string();
		bFramesEnabled = _bFramesEnabled;
		direction = _direction;
		if (_reInitInterval < 0)
		{
			throw AIPException(AIP_FATAL, "reInitInterval must be 0 or more seconds");
		}
		parseFSTimeoutDuration = _parseFSTimeoutDuration;
		readLoop = _readLoop;
		reInitInterval = _reInitInterval;
		if (parseFS)
		{
			skipDir = boost::filesystem::path(videoPath).parent_path().parent_path().parent_path().string();
		}

	}

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(videoPath) + sizeof(parseFS) + sizeof(skipDir) + + sizeof(parseFSTimeoutDuration);
	}

	std::string skipDir = "./data/mp4_videos";
	std::string videoPath = "";
	size_t biggerFrameSize = 600000;
	size_t biggerMetadataFrameSize = 60000;
	bool parseFS = true;
	bool direction = true;
	bool bFramesEnabled = false;
	uint16_t reInitInterval = 0;
	int parseFSTimeoutDuration = 15;
	bool readLoop = false;
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
	void setMetadata(framemetadata_sp metadata);
	std::string getOpenVideoPath();
	std::string addOutPutPin(framemetadata_sp& metadata);
	bool randomSeek(uint64_t seekStartTS, uint64_t seekEndTS);
	bool changePlayback(float speed, bool direction);
	bool getVideoRangeFromCache(std::string videoPath, uint64_t& start_ts, uint64_t& end_ts);
	bool randomSeek(uint64_t seekStartTS);
	bool Mp4ReaderSource::randomSeek(uint64_t skipTS, bool forceReopen);
	bool refreshCache();
	std::map<std::string, std::pair<uint64_t, uint64_t>> getCacheSnapShot();
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
	std::string mp4FramePinId;
	framemetadata_sp encodedImageMetadata;
	int outImageFrameType;
	boost::shared_ptr<Mp4readerDetailAbs> mDetail;
	Mp4ReaderSourceProps props;
	std::function<frame_sp(size_t size)> _makeFrame;
	std::function<framemetadata_sp(int type)> _getOutputMetadataByType;
};