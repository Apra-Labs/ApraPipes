#include "Mp4ReaderSource.h"
#include "FrameMetadata.h"
#include "Mp4VideoMetadata.h"
#include "Mp4ReaderSourceUtils.h"
#include "EncodedImageMetadata.h"
#include "H264Metadata.h"
#include "Frame.h"
#include "Command.h"
#include "libmp4.h"
#include "H264Utils.h"

class Mp4readerDetailAbs
{
public:
	Mp4readerDetailAbs(Mp4ReaderSourceProps& props, std::function<frame_sp(size_t size, string& pinId)> _makeFrame,
		std::function<frame_sp(frame_sp& bigFrame, size_t& size, string& pinId)> _makeframe, std::function<void()> _sendEOS)
	{
		setProps(props);
		makeFrame = _makeFrame;
		makeframe = _makeframe;
		sendEOS = _sendEOS;
		mFSParser = boost::shared_ptr<FileStructureParser>(new FileStructureParser());
	}

	~Mp4readerDetailAbs()
	{
	}
	virtual void setMetadata() = 0;
	virtual void sendEndOfStream() = 0;
	virtual bool produceFrames(frame_container& frames) = 0;
	virtual void prependSpsPps(boost::asio::mutable_buffer& iFrameBuffer) = 0;
	virtual int mp4Seek(mp4_demux* demux,uint64_t time_offset_usec, mp4_seek_method syncType) = 0;

	void setProps(Mp4ReaderSourceProps& props)
	{
		mProps = props;
		mState.mVideoPath = mProps.videoPath;
	}

	std::string getSerFormatVersion()
	{
		return mState.mSerFormatVersion;
	}

	void Init()
	{
		initNewVideo();
	}

	bool parseFS()
	{
		/*
			raise error if init fails
			return 0 if no relevant files left on disk
			relevant - files with timestamp after the mVideoPath
		*/

		bool includeStarting = (mState.mVideoCounter ? false : true) | (mState.randomSeekParseFlag);
		mState.mParsedVideoFiles.clear();
		bool flag = mFSParser->init(mState.mVideoPath, mState.mParsedVideoFiles, includeStarting, mProps.parseFS);
		if (!flag)
		{
			LOG_ERROR << "File Structure Parsing Failed. Check logs.";
			throw AIPException(AIP_FATAL, "Parsing File Structure failed");
		}
		mState.mVideoCounter = 0;
		mState.randomSeekParseFlag = false;
		mState.mParsedFilesCount = mState.mParsedVideoFiles.size();
		return mState.mParsedFilesCount == 0;
	}

	/* initNewVideo responsible for setting demux-ing files in correct order
	   Opens the mp4 file at mVideoCounter index in mParsedVideoFiles
	   If mParsedVideoFiles are exhausted - it performs a fresh fs parse for mp4 files */

	bool initNewVideo()
	{
		/*  parseFS() is called:
			only if parseFS is set AND (it is the first time OR if parse file limit is reached)
			returns false if no relevant mp4 file left on disk. */
		if (mProps.parseFS && (mState.mParsedVideoFiles.empty() || mState.mVideoCounter == mState.mParsedFilesCount))
		{
			mState.end = parseFS();
		}

		// no files left to read
		if (mState.end)
		{
			mState.mVideoPath = "";
			return false;
		}

		if (mState.mVideoCounter < mState.mParsedFilesCount) //just for safety
		{
			mState.mVideoPath = mState.mParsedVideoFiles[mState.mVideoCounter];
			++mState.mVideoCounter;
		}

		LOG_INFO << "InitNewVideo <" << mState.mVideoPath << ">";

		/* libmp4 stuff */
		// open the mp4 file here
		if (mState.demux)
		{
			mp4_demux_close(mState.demux);
			mState.videotrack = -1;
			mState.metatrack = -1;
			mState.mFrameCounter = 0;
		}

		ret = mp4_demux_open(mState.mVideoPath.c_str(), &mState.demux);
		if (ret < 0)
		{
			throw AIPException(AIP_FATAL, "Failed to open the file <" + mState.mVideoPath + ">");
		}

		unsigned int count = 0;
		char** keys = NULL;
		char** values = NULL;
		ret = mp4_demux_get_metadata_strings(mState.demux, &count, &keys, &values);
		if (ret < 0)
		{
			LOG_ERROR << "mp4_demux_get_metadata_strings <" << -ret;
		}

		auto boostVideoTS = boost::filesystem::path(mState.mVideoPath).stem().string();

		if (count > 0) {
			LOG_DEBUG << "Reading User Metadata Key-Values\n";
			for (auto i = 0; i < count; i++) {
				if ((keys[i]) && (values[i]))
				{
					if (!strcmp(keys[i], "\251too"))
					{
						LOG_DEBUG << "key <" << keys[i] << ",<" << values[i] << ">";
						mState.mSerFormatVersion.assign(values[i]);
					}
					if (!strcmp(keys[i], "\251sts"))
					{
						LOG_DEBUG << "key <" << keys[i] << ",<" << values[i] << ">";
						mState.startTimeStamp = std::stoull(values[i]);
					}
				}
			}
		}

		mState.ntracks = mp4_demux_get_track_count(mState.demux);
		for (auto i = 0; i < mState.ntracks; i++)
		{

			ret = mp4_demux_get_track_info(mState.demux, i, &mState.info);
			if (ret < 0) {
				LOG_ERROR << "mp4 track info fetch failed <" << i << "> ret<" << ret << ">";
				continue;
			}

			if (mState.info.type == MP4_TRACK_TYPE_VIDEO && mState.videotrack == -1)
			{
				mState.video = mState.info;
				mState.has_more_video = mState.info.sample_count > 0;
				mState.videotrack = 1;
				mState.mFramesInVideo = mState.info.sample_count;
				width = mState.info.video_width;
				height = mState.info.video_height;
			}
		}

		if (mState.videotrack == -1)
		{
			LOG_ERROR << "No Videotrack found in the video <" << mState.mVideoPath << " Stopping.";
			throw AIPException(AIP_FATAL, "No video track found");
		}

		try
		{
			openVideoStartingTS = std::stoull(boostVideoTS);
			mState.startTimeStamp = std::stoull(boostVideoTS);
		}
		catch (std::invalid_argument)
		{
			if (!mState.startTimeStamp)
			{
				throw AIPException(AIP_FATAL, "unexpected state - starting ts not found in video name or metadata");
			}
			openVideoStartingTS = mState.startTimeStamp;
		}

		setMetadata();
		return true;
	}

	bool randomSeek(uint64_t& skipTS, uint64_t _seekEndTS)
	{
		/* Takes a timestamp and sets proper mVideoFile and mParsedFilesCount (in case new parse is required) and initNewVideo().
		* Also, seeks to correct frame in the mVideoFile. If seek within in the videoFile fails, moving to next available video is attempted.
		* If all ways to seek fails, the read state is reset.
		*/
		seekEndTS = _seekEndTS;

		if (!mProps.parseFS)
		{
			int seekedToFrame = -1;
			uint64_t skipMsecsInFile = 0;

			if (!mState.startTimeStamp)
			{
				LOG_ERROR << "Start timestamp is not saved in the file. Can't support seeking with timestamps.";
				return false;
			}
			if (skipTS < mState.startTimeStamp)
			{
				LOG_INFO << "seek time outside range. Seeking to start of video.";
				skipMsecsInFile = 0;
			}
			else
			{
				skipMsecsInFile = skipTS - mState.startTimeStamp;
			}

			LOG_DEBUG << "Attempting seek <" << mState.mVideoPath << "> @skipMsecsInFile <" << skipMsecsInFile << ">";

			uint64_t time_offset_usec = skipMsecsInFile * 1000;
			int returnCode = mp4Seek(mState.demux, time_offset_usec, mp4_seek_method::MP4_SEEK_METHOD_NEAREST_SYNC);
			mState.mFrameCounter = seekedToFrame;
			if (returnCode == -ENFILE)
			{
				LOG_INFO << "Query time beyond the EOF. Resuming...";
				return true;
			}
			if (returnCode < 0)
			{
				LOG_ERROR << "Seek failed. Unexpected error.";
				return false;
			}
			return true;
		}

		DemuxAndParserState tempState = mState;
		std::string skipVideoFile;
		uint64_t skipMsecsInFile;
		int ret = mFSParser->randomSeek(skipTS, mProps.skipDir, skipVideoFile, skipMsecsInFile);
		LOG_INFO << "Attempting seek <" << skipVideoFile << "> @skipMsecsInFile <" << skipMsecsInFile << ">";
		if (ret < 0)
		{
			LOG_ERROR << "Skip to ts <" << skipTS << "> failed. Please check skip dir <" << mProps.skipDir << ">";
			return false;
		}
		bool found = false;
		for (auto i = 0; i < mState.mParsedVideoFiles.size(); ++i)
		{
			if (mState.mParsedVideoFiles[i] == skipVideoFile)
			{
				mState.mVideoCounter = i;
				found = true;
				break;
			}
		}
		if (!found)
		{
			// do fresh fs parse
			mState.mVideoPath = skipVideoFile;
			mState.mVideoCounter = mState.mParsedFilesCount;
			mState.randomSeekParseFlag = true;
		}
		initNewVideo();
		if (skipMsecsInFile)
		{
			uint64_t time_offset_usec = skipMsecsInFile * 1000;
			int returnCode = mp4Seek(mState.demux, time_offset_usec, mp4_seek_method::MP4_SEEK_METHOD_NEAREST_SYNC);
			// to determine the end of video
			mState.mFrameCounter = seekedToFrame;

			if (returnCode < 0)
			{
				LOG_ERROR << "Error while skipping to ts <" << skipTS << "> failed. File <" << skipVideoFile << "> @ time <" << skipMsecsInFile << "> ms errorCode <" << returnCode << ">";
				LOG_ERROR << "Trying to seek to next available file...";
				auto nextFlag = mFSParser->getNextToVideoFileFlag();
				if (nextFlag < 0)
				{
					// reset the state of mux to before seek op
					LOG_ERROR << "No next file found <" << nextFlag << ">" << "Resetting the reader state to before seek op";
					mState = tempState;
					LOG_ERROR << "Switching back to video <" << mState.mVideoPath << ">";
					// hot fix to avoid demux close attempt
					mState.demux = nullptr;
					mState.mVideoCounter -= 1;
					initNewVideo();
					return false;
				}
				mState.mVideoPath = mFSParser->getNextVideoFile();
				mState.mVideoCounter = mState.mParsedFilesCount;
				mState.randomSeekParseFlag = true;
				initNewVideo();
			}
		}
		return true;
	}

	void readNextFrame(uint8_t* sampleFrame, uint8_t* sampleMetadataFrame, size_t& imageFrameSize, size_t& metadataFrameSize)
	{

		// all frames of the open video are already read and end has not reached
		if (mState.mFrameCounter == mState.mFramesInVideo && !mState.end)
		{
			sendEndOfStream();
			// if parseFS is unset, it is the end
			LOG_ERROR << "frames number" << mState.mFrameCounter;
			mState.end = !mProps.parseFS;
			initNewVideo();
		}

		if (mState.end) // no files left to be parsed
		{
			sampleFrame = nullptr;
			imageFrameSize = 0;
			return;
		}

		if (mState.has_more_video)
		{
			ret = mp4_demux_get_track_sample(mState.demux,
				mState.video.id,
				1,
				sampleFrame,
				mProps.biggerFrameSize,
				sampleMetadataFrame,
				mProps.biggerMetadataFrameSize,
				&mState.sample);
			imageFrameSize = imageFrameSize + mState.sample.size;
			metadataFrameSize = mState.sample.metadata_size;
			/* To get only info about the frames
			ret = mp4_demux_get_track_sample(
				demux, id, 1, NULL, 0, NULL, 0, &sample);
			*/

			if (ret != 0 || mState.sample.size == 0)
			{
				LOG_INFO << "<" << ret << "," << mState.sample.size << "," << mState.sample.metadata_size << ">";
				mState.has_more_video = 0;
				sampleFrame = nullptr;
				imageFrameSize = 0;
				return;
			}
			++mState.mFrameCounter;
		}
		
		return;
	}
	Mp4ReaderSourceProps mProps;
protected:

	struct DemuxAndParserState
	{
		mp4_demux* demux = nullptr;
		mp4_track_info info, video;
		mp4_track_sample sample;
		std::string mSerFormatVersion = "";
		int has_more_video;
		int videotrack = -1;
		int metatrack = -1;
		int ntracks = -1;
		uint64_t startTimeStamp = 0;
		uint32_t mParsedFilesCount = 0;
		uint32_t mVideoCounter = 0;
		uint32_t mFrameCounter = 0;
		uint32_t mFramesInVideo = 0;
		std::vector<std::string> mParsedVideoFiles;
		std::string mVideoPath = "";
		bool randomSeekParseFlag = false;
		bool end = false;
		Mp4ReaderSourceProps props;
	} mState;
	uint64_t openVideoStartingTS = 0;
	uint64_t seekEndTS = 9999999999999;
	int seekedToFrame = -1;
	/*
		mState.end = true is possible only in two cases:
		- if parseFS found no more relevant files on the disk
		- parseFS is disabled and intial video has finished playing
	*/
public:
	int width = 0;
	int height = 0;
	int ret;
	std::function<frame_sp(size_t size, string& pinId)> makeFrame;
	std::function<void()> sendEOS;
	std::function<frame_sp(frame_sp& bigFrame, size_t& size, string& pinId)> makeframe;
	boost::shared_ptr<FileStructureParser> mFSParser;
	std::string h264ImagePinId;
	std::string encodedImagePinId;
	std::string mp4FramePinId;
};

class Mp4readerDetailJpeg : public Mp4readerDetailAbs
{
public:
	Mp4readerDetailJpeg(Mp4ReaderSourceProps& props, std::function<frame_sp(size_t size, std::string& pinId)> _makeFrame,
		std::function<frame_sp(frame_sp& bigFrame, size_t& size, string& pinId)> _makeframe, std::function<void()> _sendEOS) : Mp4readerDetailAbs(props, _makeFrame, _makeframe, _sendEOS)
	{}
	~Mp4readerDetailJpeg() { mp4_demux_close(mState.demux); }
	void setMetadata();
	bool produceFrames(frame_container& frames);
	void prependSpsPps(boost::asio::mutable_buffer& iFrameBuffer) {}
	void sendEndOfStream() {}
	int mp4Seek(mp4_demux* demux, uint64_t time_offset_usec, mp4_seek_method syncType);
};

class Mp4readerDetailH264 : public Mp4readerDetailAbs
{
public:
	Mp4readerDetailH264(Mp4ReaderSourceProps& props, std::function<frame_sp(size_t size, string& pinId)> _makeFrame,
		std::function<frame_sp(frame_sp& bigFrame, size_t& size, string& pinId)> _makeframe, std::function<void()> _sendEOS) : Mp4readerDetailAbs(props, _makeFrame, _makeframe, _sendEOS)
	{}
	~Mp4readerDetailH264() { mp4_demux_close(mState.demux); }
	void setMetadata();
	bool produceFrames(frame_container& frames);
	void prependSpsPps(boost::asio::mutable_buffer& iFrameBuffer);
	void sendEndOfStream();
	int mp4Seek(mp4_demux* demux, uint64_t time_offset_usec, mp4_seek_method syncType);
private:
	uint8_t* sps = nullptr;
	uint8_t* pps = nullptr;
	size_t spsSize = 0;
	size_t ppsSize = 0;
	bool seekedToEndTS = false;
	bool isRandomSeek = true;
};

void Mp4readerDetailJpeg::setMetadata()
{
	auto metadata = framemetadata_sp(new EncodedImageMetadata(width, height));
	if (!metadata->isSet())
	{
		return;
	}
	auto encodedMetadata = FrameMetadataFactory::downcast<EncodedImageMetadata>(metadata);
	encodedMetadata->setData(*encodedMetadata);

	auto mp4FrameMetadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	//// set proto version in mp4videometadata
	auto serFormatVersion = getSerFormatVersion();
	auto mp4VideoMetadata = FrameMetadataFactory::downcast<Mp4VideoMetadata>(mp4FrameMetadata);
	mp4VideoMetadata->setData(serFormatVersion);
	return;
}

int Mp4readerDetailJpeg::mp4Seek(mp4_demux* demux, uint64_t time_offset_usec, mp4_seek_method syncType) 
{
	auto ret = mp4_demux_seek_jpeg(demux, time_offset_usec, syncType, &seekedToFrame);
	return ret;
}

bool Mp4readerDetailJpeg::produceFrames(frame_container& frames)
{
	frame_sp imgFrame = makeFrame(mProps.biggerFrameSize, encodedImagePinId);
	frame_sp metadataFrame = makeFrame(mProps.biggerMetadataFrameSize, mp4FramePinId);
	uint8_t* sampleFrame = reinterpret_cast<uint8_t*>(imgFrame->data());
	uint8_t* sampleMetadataFrame = reinterpret_cast<uint8_t*>(metadataFrame->data());
	size_t imageActualSize = 0;
	size_t metadataActualSize = 0;
	readNextFrame(sampleFrame, sampleMetadataFrame, imageActualSize, metadataActualSize);

	if (!imageActualSize || !sampleFrame)
	{
		return true;
	}
	
	auto trimmedImgFrame = makeframe(imgFrame, imageActualSize, encodedImagePinId);

	uint64_t sample_ts_usec = mp4_sample_time_to_usec(mState.sample.dts, mState.video.timescale);
	auto frameTSInMsecs = openVideoStartingTS + (sample_ts_usec / 1000);

	trimmedImgFrame->timestamp = frameTSInMsecs;

	if (seekEndTS <= frameTSInMsecs)
	{
		return true;
	}

	frames.insert(make_pair(encodedImagePinId, trimmedImgFrame));
	if (metadataActualSize)
	{
		auto metadataSizeFrame = makeframe(metadataFrame, metadataActualSize, mp4FramePinId);
		metadataSizeFrame->timestamp = frameTSInMsecs;
		frames.insert(make_pair(mp4FramePinId, metadataSizeFrame));
	}
	return true;
}

void Mp4readerDetailH264::setMetadata()
{
	auto metadata = framemetadata_sp(new H264Metadata(width, height));
	if (!metadata->isSet())
	{
		return;
	}
	auto h264Metadata = FrameMetadataFactory::downcast<H264Metadata>(metadata);
	h264Metadata->setData(*h264Metadata);

	auto mp4FrameMetadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	//// set proto version in mp4videometadata
	auto serFormatVersion = getSerFormatVersion();
	auto mp4VideoMetadata = FrameMetadataFactory::downcast<Mp4VideoMetadata>(mp4FrameMetadata);
	mp4VideoMetadata->setData(serFormatVersion);

	struct mp4_video_decoder_config* vdc =
		(mp4_video_decoder_config*)malloc(
			sizeof(mp4_video_decoder_config));
	unsigned int track_id = 1;
	mp4_demux_get_track_video_decoder_config(
		mState.demux, track_id, vdc);
	sps = vdc->avc.sps;
	pps = vdc->avc.pps;
	spsSize = vdc->avc.sps_size;
	ppsSize = vdc->avc.pps_size;
	return;
}

int Mp4readerDetailH264::mp4Seek(mp4_demux* demux, uint64_t time_offset_usec, mp4_seek_method syncType)
{
	auto ret = mp4_demux_seek(demux, time_offset_usec, syncType, &seekedToFrame);
	return ret;
}

void Mp4readerDetailH264::sendEndOfStream()
{
	sendEOS();
}

void Mp4readerDetailH264::prependSpsPps(boost::asio::mutable_buffer& iFrameBuffer)
{
	//1a write sps on tmpBuffer.data()
	//1b tmpBuffer+=sizeof_sps
	//2a write NALU on tmpBuffer.data()
	//2b tmpBuffer+=4
	//1a write pps on tmpBuffer.data()
	//1b tmpBuffer+=sizeof_pps
	// Now pass tmpBuffer.data() and tmpBuffer.size() to libmp4
	char NaluSeprator[4] = { 00 ,00, 00 ,01 };
	auto nalu = reinterpret_cast<uint8_t*>(NaluSeprator);
	memcpy(iFrameBuffer.data(), nalu, 4);
	iFrameBuffer += 4;
	memcpy(iFrameBuffer.data(), sps, spsSize);
	iFrameBuffer += spsSize;
	memcpy(iFrameBuffer.data(), nalu, 4);
	iFrameBuffer += 4;
	memcpy(iFrameBuffer.data(), pps, ppsSize);
	iFrameBuffer += ppsSize;
}

bool Mp4readerDetailH264::produceFrames(frame_container& frames)
{
	frame_sp imgFrame = makeFrame(mProps.biggerFrameSize, h264ImagePinId);
	boost::asio::mutable_buffer tmpBuffer(imgFrame->data(), imgFrame->size());
	size_t imageActualSize = 0;

	if (mState.randomSeekParseFlag && isRandomSeek)
	{
		prependSpsPps(tmpBuffer);
		imageActualSize = spsSize + ppsSize + 8;
		isRandomSeek = false;
	}
	frame_sp metadataFrame = makeFrame(mProps.biggerMetadataFrameSize,mp4FramePinId);
	uint8_t* sampleFrame = reinterpret_cast<uint8_t*>(tmpBuffer.data());
	uint8_t* sampleMetadataFrame = reinterpret_cast<uint8_t*>(metadataFrame->data());
	size_t metadataActualSize = 0;
	readNextFrame(sampleFrame, sampleMetadataFrame, imageActualSize, metadataActualSize);

	if (!imageActualSize || !sampleFrame)
	{
		return true;
	}

	auto trimmedImgFrame = makeframe(imgFrame, imageActualSize, h264ImagePinId);
	auto tempBuffer = const_buffer(trimmedImgFrame->data(), trimmedImgFrame->size());
	auto ret = H264Utils::parseNalu(tempBuffer);
	short typeFound;
	const_buffer spsBuff, ppsBuff;
	tie(typeFound, spsBuff, ppsBuff) = ret; 

	uint64_t sample_ts_usec = mp4_sample_time_to_usec(mState.sample.dts, mState.video.timescale);
	auto frameTSInMsecs = openVideoStartingTS + (sample_ts_usec / 1000);

	trimmedImgFrame->timestamp = frameTSInMsecs;

	if (seekedToEndTS)
	{
		return true;
	}

	if (seekEndTS <= frameTSInMsecs && !mProps.bFramesEnabled)
	{
		return true;
	}

	if (seekEndTS <= frameTSInMsecs && mProps.bFramesEnabled)
	{
		if (typeFound == H264Utils::H264_NAL_TYPE::H264_NAL_TYPE_IDR_SLICE)
		{
			frames.insert(make_pair(h264ImagePinId, trimmedImgFrame));
			seekedToEndTS = true;
			return true;
		}
	}

	frames.insert(make_pair(h264ImagePinId, trimmedImgFrame));
	if (metadataActualSize)
	{
		auto metadataSizeFrame = makeframe(metadataFrame, metadataActualSize, mp4FramePinId);
		metadataSizeFrame->timestamp = frameTSInMsecs;
		frames.insert(make_pair(mp4FramePinId, metadataSizeFrame));
	}
	return true;
}

Mp4ReaderSource::Mp4ReaderSource(Mp4ReaderSourceProps _props)
	: Module(SOURCE, "Mp4ReaderSource", _props), props(_props)
{
}

Mp4ReaderSource::~Mp4ReaderSource() {}

bool Mp4ReaderSource::init()
{
	if (!Module::init())
	{
		return false;
	}
	auto inputPinIdMetadataMap = getFirstOutputMetadata();
	auto  mFrameType = inputPinIdMetadataMap->getFrameType();
	if (mFrameType == FrameMetadata::FrameType::ENCODED_IMAGE)
	{
		mDetail.reset(new Mp4readerDetailJpeg(
			props,
			[&](size_t size, string& pinId)
			{ return makeFrame(size, pinId); },
			[&](frame_sp& frame, size_t& size, string& pinId)
			{ return makeFrame(frame, size, pinId); },
			[&](void)
			{return sendEOS(); }));
	}
	else if (mFrameType == FrameMetadata::FrameType::H264_DATA)
	{
		mDetail.reset(new Mp4readerDetailH264(props,
			[&](size_t size, string& pinId)
			{ return makeFrame(size, pinId); },
			[&](frame_sp& frame, size_t& size, string& pinId)
			{ return makeFrame(frame, size, pinId); },
			[&](void)
			{return sendEOS(); }));
	}

	mDetail->encodedImagePinId = encodedImagePinId;
	mDetail->h264ImagePinId = h264ImagePinId;
	mDetail->mp4FramePinId = mp4FramePinId;

	mDetail->Init();
	return true;
}

bool Mp4ReaderSource::term()
{
	auto moduleRet = Module::term();

	return moduleRet;
}

std::string Mp4ReaderSource::addOutPutPin(framemetadata_sp& metadata)
{
	auto outFrameType = metadata->getFrameType();

	if (outFrameType == FrameMetadata::FrameType::ENCODED_IMAGE)
	{
		encodedImagePinId = Module::addOutputPin(metadata);
		return encodedImagePinId;
	}
	else if (outFrameType == FrameMetadata::FrameType::H264_DATA)
	{
		h264ImagePinId = Module::addOutputPin(metadata);
		return h264ImagePinId;
	}
	else
	{
		mp4FramePinId = Module::addOutputPin(metadata);
		return mp4FramePinId;
	}
}

bool Mp4ReaderSource::produce()
{
	frame_container frames;
	mDetail->produceFrames(frames);
	send(frames);
	return true;
}

bool Mp4ReaderSource::validateOutputPins()
{
	if (getNumberOfOutputPins() > 2)
	{
		return false;
	}

	auto outputMetadataByPin = getFirstOutputMetadata();

	FrameMetadata::FrameType frameType = outputMetadataByPin->getFrameType();

	if (frameType != FrameMetadata::MP4_VIDEO_METADATA && frameType != FrameMetadata::ENCODED_IMAGE && frameType != FrameMetadata::H264_DATA)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input frameType is expected to be MP4_VIDEO_METADATA or ENCODED_IMAGE. Actual<" << frameType << ">";
		return false;
	}

	auto memType = outputMetadataByPin->getMemType();
	if (memType != FrameMetadata::MemType::HOST)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input memType is expected to be HOST. Actual<" << memType << ">";
		return false;
	}

	return true;
}

Mp4ReaderSourceProps Mp4ReaderSource::getProps()
{
	return mDetail->mProps;
}

bool Mp4ReaderSource::handlePropsChange(frame_sp& frame)
{
	Mp4ReaderSourceProps props(mDetail->mProps.videoPath, mDetail->mProps.parseFS, mDetail->mProps.biggerFrameSize, mDetail->mProps.biggerMetadataFrameSize);
	bool ret = Module::handlePropsChange(frame, props);
	mDetail->setProps(props);
	mDetail->Init();
	return ret;
}

void Mp4ReaderSource::setProps(Mp4ReaderSourceProps& props)
{
	Module::addPropsToQueue(props);
}

bool Mp4ReaderSource::handleCommand(Command::CommandType type, frame_sp& frame)
{
	if (type == Command::CommandType::Seek)
	{
		Mp4SeekCommand seekCmd;
		getCommand(seekCmd, frame);
		return mDetail->randomSeek(seekCmd.seekStartTS,seekCmd.seekEndTS);
	}
	else
	{
		return Module::handleCommand(type, frame);
	}
}

bool Mp4ReaderSource::randomSeek(uint64_t seekStartTS, uint64_t seekEndTS)
{
	Mp4SeekCommand cmd(seekStartTS, seekEndTS);
	return queueCommand(cmd);
}
