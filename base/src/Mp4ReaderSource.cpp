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
#include "OrderedCacheOfFiles.h"
#include "AIPExceptions.h"
#include "Mp4ErrorFrame.h"
#include <iostream>

class Mp4readerDetailAbs
{
public:
	Mp4readerDetailAbs(Mp4ReaderSourceProps& props, std::function<frame_sp(size_t size, string& pinId)> _makeFrame,
		std::function<frame_sp(frame_sp& bigFrame, size_t& size, string& pinId)> _makeframe, std::function<void(frame_sp frame)> _sendEOS, std::function<void(framemetadata_sp metadata)> _setMetadata, std::function<void(frame_sp& errorFrame)> _sendMp4ErrorFrame)
	{
		setProps(props);
		makeFrame = _makeFrame;
		makeframe = _makeframe;
		sendEOS = _sendEOS;
		mSetMetadata = _setMetadata;
		sendMp4ErrorFrame = _sendMp4ErrorFrame;
		mFSParser = boost::shared_ptr<FileStructureParser>(new FileStructureParser());
		cof = boost::shared_ptr<OrderedCacheOfFiles>(new OrderedCacheOfFiles(mProps.skipDir));
	}

	~Mp4readerDetailAbs()
	{
	}
	virtual void setMetadata() = 0;
	virtual void sendEndOfStream() = 0;
	virtual bool produceFrames(frame_container& frames) = 0;
	virtual void prependSpsPps(boost::asio::mutable_buffer& iFrameBuffer) = 0;
	virtual int mp4Seek(mp4_demux* demux,uint64_t time_offset_usec, mp4_seek_method syncType, int &seekedToFrame) = 0;
	
	void Init()
	{
		// TODO - setProps + reset the mState + Init() call can force a fresh disk parse and do a new playback.
		// #Dec_27_Review - redundant - use makeBuffer from produce
		sentEOSSignal = false;

		if (mProps.parseFS)
		{
			auto boostVideoTS = boost::filesystem::path(mState.mVideoPath).stem().string();
			uint64_t start_parsing_ts = 0;
			try
			{
				start_parsing_ts = std::stoull(boostVideoTS);
			}
			catch (std::invalid_argument)
			{
				auto msg = "Video File name not in proper format.Check the filename sent as props. \
					If you want to read a file with custom name instead, please disable parseFS flag.";
				LOG_ERROR << msg;
				throw AIPException(AIP_FATAL, msg);
			}
			cof->parseFiles(start_parsing_ts, mState.direction, true, false); // enable exactMatch, dont disable disableBatchSizeCheck
		}
		initNewVideo(true); // enable firstOpenAfterInit
	}

	void setProps(Mp4ReaderSourceProps& props)
	{
		mProps = props;
		mState.mVideoPath = mProps.videoPath;

		mState.direction = mProps.direction;

		std::chrono::time_point<std::chrono::system_clock> t = std::chrono::system_clock::now();
		auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(t.time_since_epoch());
		uint64_t nowTS = dur.count();
		reloadFileAfter = calcReloadFileAfter();
	}

	std::string getOpenVideoPath()
	{
		return mState.mVideoPath;
	}

	std::map<std::string, std::pair<uint64_t, uint64_t>> getSnapShot()
	{
		return cof->getSnapShot();
	}

	bool refreshCache()
	{
		return cof->refreshCache();
	}

	std::string getSerFormatVersion()
	{
		return mState.mSerFormatVersion;
	}

	void setPlayback(float _speed, bool _direction)
	{
		if (_speed != mState.speed)
		{
			mState.speed = _speed;
		}
		// only if direction changes
		if (mState.direction != _direction)
		{
			mState.direction = _direction;
			/* using the new direction, check if new video has to be opened
			setting proper mFrameCounterIdx will ensure new video is init at right time.*/
			if (mState.direction)
			{
				// bwd -> fwd
				if (mState.mFrameCounterIdx > mState.mFramesInVideo - 1)
				{
					// before direction change: we were going to read last frame
					mState.mFrameCounterIdx = mState.mFramesInVideo;
				}
				else
				{
					// if we were at EOF before dir change - so end/wait/sendEOS flags need to be reset
					if (mState.mFrameCounterIdx == -1)
					{
						mState.end = false;
						waitFlag = false;
						sentEOSSignal = false;
					}
					mState.mFrameCounterIdx++;
				}
			}
			else if (!mState.direction)
			{
				// fwd -> bwd
				if (mState.mFrameCounterIdx <= 0)
				{
					// before direction change: we were going to read first frame
					mState.mFrameCounterIdx = -1;
				}
				else
				{
					// if we were at EOF before dir change - so end/wait/sendEOS flags need to be reset
					if (mState.mFrameCounterIdx == mState.mFramesInVideo)
					{
						mState.end = false;
						waitFlag = false;
						sentEOSSignal = false;
					}
					mState.mFrameCounterIdx--;
				}
			}
			LOG_TRACE << "changed direction frameIdx <" << mState.mFrameCounterIdx << "> totalFrames <" << mState.mFramesInVideo << ">";
			mp4_demux_toggle_playback(mState.demux, mState.video.id);
		}
	}

	bool getVideoRangeFromCache(std::string& videoPath, uint64_t& start_ts, uint64_t& end_ts)
	{
		return cof->fetchFromCache(videoPath, start_ts, end_ts);
	}

	bool attemptFileClose()
	{
		LOG_INFO << "attemptFileClose called";
		try
		{
			if (mState.demux)
			{
				mp4_demux_close(mState.demux);
			}
		}
		catch (...)
		{
			auto msg = "Error occured while closing the video file <" + mState.mVideoPath + ">";
			LOG_ERROR << msg;
			throw Mp4Exception(MP4_FILE_CLOSE_FAILED, msg);
		}
		return true;
	}

	bool parseFS()
	{
		/*
			parseFS() asks OrderedCacheOfFiles to parse disk with timestamp of current file and playback direction.
			return 0, if no relevant files left on disk (end state i.e. EOF)
			else return 1
		*/

		bool foundRelevantFilesOnDisk = cof->parseFiles(mState.resolvedStartingTS, mState.direction);
		return foundRelevantFilesOnDisk;
	}

	/* initNewVideo responsible for setting demux-ing files in correct order
	   Opens the mp4 file at mVideoCounter index in mParsedVideoFiles
	   If mParsedVideoFiles are exhausted - it performs a fresh fs parse for mp4 files */

	bool initNewVideo(bool firstOpenAfterInit = false)
	{
		/*  parseFS() is called:
			only if parseFS is set AND (it is the first time OR if parse file limit is reached)
			returns false if no relevant mp4 file left on disk. */

		auto filePath = boost::filesystem::path(mState.mVideoPath);
		if (filePath.extension() != ".mp4")
		{
			if (!mFSParser->parseDir(filePath, mState.mVideoPath))
			{
				LOG_DEBUG << "Mp4 file is not present" << ">";
				isVideoFileFound = false;
				return false;
			}
			isVideoFileFound = true;
		}

		auto nextFilePath = mState.mVideoPath;

		if (mProps.parseFS)
		{
			if (!firstOpenAfterInit)
			{
				try
				{
					nextFilePath = cof->getNextFileAfter(mState.mVideoPath, mState.direction);
				}
				catch (AIP_Exception& exception)
				{
					if (exception.getCode() == MP4_OCOF_END) // EOC
					{
						LOG_INFO << "End OF Cache. Parsing disk again.";
						nextFilePath = "";
					}
					else
					{
						auto msg = "Failed to find next file to <" + mState.mVideoPath + ">";
						LOG_ERROR << msg;
						throw Mp4Exception(MP4_UNEXPECTED_STATE, msg);
					}
				}
			}
			if (nextFilePath.empty()) // we had reached EOC
			{
				mState.end = !parseFS();
				if (!mState.end)
				{
					try
					{
						nextFilePath = cof->getNextFileAfter(mState.mVideoPath, mState.direction); // from updated cache
					}
					catch (Mp4_Exception& ex)
					{
						if (ex.getCode() == MP4_OCOF_END)
						{
							LOG_ERROR << "parse found new files but getNextFileAfter hit EOC while looking for a potential file.";
							mState.end = true;
						}
						else
						{
							auto msg = "unexpected state while getting next file after successful parse <" + ex.getError() + ">";
							LOG_ERROR << msg;
							throw Mp4Exception(MP4_UNEXPECTED_STATE, msg);
						}
					}
				}
			}
		}

		// no files left to read OR no new files even after fresh parse OR empty folder
		if (mState.end)
		{
			LOG_INFO << "Reached EOF end state in playback.";
			if (mProps.readLoop)
			{
				openVideoSetPointer(mState.mVideoPath);
				mState.end = false;
				return true;
			}
			// reload the current file
			if (waitFlag)
			{
				uint64_t tstart_ts, tend_ts;
				cof->readVideoStartEnd(mState.mVideoPath, tstart_ts, tend_ts);
				if (mProps.parseFS) // update cache
				{
					cof->updateCache(mState.mVideoPath, tstart_ts, tend_ts);
					cof->fetchAndUpdateFromDisk(mState.mVideoPath, tstart_ts, tend_ts);
				}
				// verify if file is updated
				LOG_TRACE << "old endTS <" << mState.endTS << "> new endTS <" << tend_ts << ">";
				if (mState.endTS >= tend_ts)
				{
					return true;
				}
				// open the video in reader and set pointer at correct place
				LOG_TRACE << "REOPEN THE FILE < " << mState.mVideoPath << ">";
				openVideoSetPointer(mState.mVideoPath);
				auto seekTS = mState.direction ? mState.frameTSInMsecs + 1 : mState.frameTSInMsecs - 1;
				auto ret = randomSeekInternal(seekTS); // also resets end
				if (!ret)
				{
					auto msg = "Unexpected issue occured while resuming playback after reloading the file <"
						+ mState.mVideoPath + "> @seekTS <" + std::to_string(seekTS) + ">";
					LOG_ERROR << msg;
					throw Mp4Exception(MP4_RELOAD_RESUME_FAILED, msg);
				}
				// disable reloading now
				waitFlag = false;
				// allow sending EOS frame again
				sentEOSSignal = false;
			}
			return true;
		}
		LOG_INFO << "Opening New Video <" << nextFilePath << ">";
		/* in case we had reached EOR -> waiting state -> but after reInitInterval new file has been written
		   the waitFlag will be reset in openVideoSetPointer
		*/
		openVideoSetPointer(nextFilePath);
		return true;
	}

	void openVideoSetPointer(std::string &filePath)
	{
		if (mState.demux)
		{
			termOpenVideo();
		}

		LOG_INFO << "opening video <" << filePath << ">";
		ret = mp4_demux_open(filePath.c_str(), &mState.demux);
		if (ret < 0)
		{
			// if (ret == -ENOENT) // if no such file exists
			// {
			// 	try
			// 	{
			// 		// check if file does not exist on disk, then delete from cache 
			// 		if (!boost::filesystem::exists(filePath))
			// 		{
			// 			LOG_ERROR << "Video File <" << filePath << "> does not exist on disk. Deleting lost entry from cache...";
			// 			cof->deleteLostEntry(filePath); // ENOENT (-2)
			// 		}
			// 	}
			// 	catch (...)
			// 	{
			// 		LOG_ERROR << "Failed to verify if file <" << filePath << "> is deleted from disk.";
			// 	}
			// }
			auto msg = "Failed to open the file <" + filePath + "> libmp4 errorcode<" + std::to_string(ret) + ">";
			LOG_ERROR << msg;
			throw Mp4Exception(MP4_OPEN_FILE_FAILED, msg);
		}

		/* read metadata string to get serializer format version & starting timestamp from the header */
		unsigned int count = 0;
		char** keys = NULL;
		char** values = NULL;
		ret = mp4_demux_get_metadata_strings(mState.demux, &count, &keys, &values);
		if (ret < 0)
		{
			LOG_ERROR << "mp4_demux_get_metadata_strings <" << -ret;
		}

		if (count > 0) {
			LOG_INFO << "Reading User Metadata Key-Values\n";
			for (unsigned int i = 0; i < count; i++) {
				if ((keys[i]) && (values[i]))
				{
					if (!strcmp(keys[i], "\251too"))
					{
						LOG_INFO << "key <" << keys[i] << ",<" << values[i] << ">";
						mState.mSerFormatVersion.assign(values[i]);
					}
					if (!strcmp(keys[i], "\251sts"))
					{
						LOG_INFO << "key <" << keys[i] << ",<" << values[i] << ">";
						mState.startTimeStampFromFile = std::stoull(values[i]);
					}
				}
			}
		}

		/* Update mState with relevant information */
		mState.mVideoPath = filePath;
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
				mWidth = mState.info.video_width;
				mHeight = mState.info.video_height;
			}
			setMetadata();
		}

		if (mState.videotrack == -1)
		{
			auto msg = "No Videotrack found in the video <" + mState.mVideoPath + ">";
			LOG_ERROR << msg;
			throw Mp4Exception(MP4_MISSING_VIDEOTRACK, msg);
		}

		// starting timestamp of the video will either come from the video name or the header
		auto boostVideoTS = boost::filesystem::path(mState.mVideoPath).stem().string();
		try
		{
			mState.resolvedStartingTS = std::stoull(boostVideoTS);
			mState.startTimeStampFromFile = std::stoull(boostVideoTS);
		}
		catch (std::invalid_argument)
		{
			if (!mState.startTimeStampFromFile)
			{
				auto msg = "unexpected: starting ts not found in video name or metadata";
				LOG_ERROR << msg;
				throw Mp4Exception(MP4_MISSING_START_TS, msg);
			}
			mState.resolvedStartingTS = mState.startTimeStampFromFile;
		}

		// update output encoded image metadata
		/*updatedEncodedImgMetadata = framemetadata_sp(new EncodedImageMetadata(mWidth, mHeight));
		mSetMetadata(updatedEncodedImgMetadata);*/

		// get the end_ts of the video and update the cache 
		uint64_t dummy_start_ts, duration;
		try
		{
			mp4_demux_time_range(mState.demux, &dummy_start_ts, &duration);
		}
		catch (...)
		{
			auto msg = "Failed to get time range of the video.";
			LOG_ERROR << msg;
			throw Mp4Exception(MP4_TIME_RANGE_FETCH_FAILED, msg);
		}
		mState.endTS = mState.resolvedStartingTS + duration;
		if (mProps.parseFS)
		{
			cof->updateCache(mState.mVideoPath, mState.resolvedStartingTS, mState.endTS);
		}

		// if playback direction is reverse, move the pointer to the last frame of the video
		if (!mState.direction)
		{
			try
			{
				mp4_set_reader_pos_lastframe(mState.demux, mState.videotrack, mState.direction);
			}
			catch (...)
			{
				auto msg = "Unexpected error while moving the read pointer to last frame of <" + mState.mVideoPath + ">";
				LOG_ERROR << msg;
				throw Mp4Exception(MP4_SET_POINTER_END_FAILED, msg);
			}
			mState.mFrameCounterIdx = mState.mFramesInVideo - 1;
		}
		// reset flags
		waitFlag = false;
		sentEOSSignal = false;
	}

	bool randomSeekInternal(uint64_t& skipTS, bool forceReopen = false)
	{
		/* Seeking in custom file i.e. parseFS is disabled */
		if (!mProps.parseFS)
		{
			int seekedToFrame = -1;
			uint64_t skipMsecsInFile = 0;

			if (!mState.startTimeStampFromFile)
			{
				LOG_ERROR << "Start timestamp is not saved in the file. Can't support seeking with timestamps.";
				return false;
			}
			if (skipTS < mState.startTimeStampFromFile)
			{
				LOG_INFO << "seek time outside range. Seeking to start of video.";
				skipMsecsInFile = 0;
			}
			else
			{
				skipMsecsInFile = skipTS - mState.startTimeStampFromFile;
			}

			LOG_INFO << "Attempting seek <" << mState.mVideoPath << "> @skipMsecsInFile <" << skipMsecsInFile << ">";
			uint64_t time_offset_usec = skipMsecsInFile * 1000;
			int returnCode = mp4Seek(mState.demux, time_offset_usec, mp4_seek_method::MP4_SEEK_METHOD_NEXT_SYNC, seekedToFrame);
			mState.mFrameCounterIdx = seekedToFrame;
			if (returnCode == -ENFILE)
			{
				LOG_INFO << "Query time beyond the EOF. Resuming...";
				auto frame = frame_sp(new EoSFrame(EoSFrame::EoSFrameType::MP4_SEEK_EOS, skipTS));
				sendEOS(frame);
				seekReachedEOF = true;
				return true;
			}
			if (returnCode < 0)
			{
				LOG_ERROR << "Seek failed. Unexpected error.";
				auto msg = "Unexpected error happened whie seeking inside the video file.";
				LOG_ERROR << msg;
				throw Mp4Exception(MP4_SEEK_INSIDE_FILE_FAILED, msg);
			}
			// continue reading file if seek is successful
			mState.has_more_video = 1;
			mState.end = false; // enable seeking after eof
			// reset flags
			waitFlag = false;
			sentEOSSignal = false;
			return true;
		}

		/* Regular seek
			1. bool cof.getRandomSeekFile(skipTS, skipVideoFile, skipMsec)
			2. That method returns false if randomSeek has failed. Then, do nothing.
			3. On success, we know which file to open and how many secs to skip in it.
			4. Do openVideoFor() this file & seek in this file & set the mFrameCountIdx to the seekedToFrame ?
		*/
		std::string skipVideoFile;
		uint64_t skipMsecsInFile;
		bool ret = cof->getRandomSeekFile(skipTS, mState.direction, skipMsecsInFile, skipVideoFile);
		if (!ret)
		{
			// send EOS signal
			auto frame = frame_sp(new EoSFrame(EoSFrame::EoSFrameType::MP4_SEEK_EOS, skipTS));
			sendEOS(frame);
			// skip the frame in the readNextFrame that happens in the same step
			seekReachedEOF = true;
			LOG_INFO << "Seek to skipTS <" << skipTS << "> failed. Resuming playback...";
			return false;
		}
		// check if the skipTS is in already opened file (if mState.end has not reached)
		bool skipTSInOpenFile = false;
		if (!mState.end)
		{
			skipTSInOpenFile = cof->isTimeStampInFile(mState.mVideoPath, skipTS);
		}
		// force reopen the video file if skipVideo is the last file in cache
		auto lastVideoInCache = boost::filesystem::canonical(cof->getLastVideoInCache());
		bool skipFileIsLastInCache = boost::filesystem::equivalent(lastVideoInCache, boost::filesystem::canonical(skipVideoFile));
		if (!skipTSInOpenFile || skipFileIsLastInCache)
		{
			// open skipVideoFile if mState.end has reached or skipTS not in currently open video
			openVideoSetPointer(skipVideoFile); // it is possible that this file has been deleted but not removed from cache
		}
		LOG_INFO << "Attempting seek <" << skipVideoFile << "> @skipMsecsInFile <" << skipMsecsInFile << ">";
		if (skipMsecsInFile)
		{
			uint64_t time_offset_usec = skipMsecsInFile * 1000;
			int seekedToFrame = 0;
			mp4_seek_method seekDirectionStrategy = mState.direction ? mp4_seek_method::MP4_SEEK_METHOD_NEXT_SYNC : mp4_seek_method::MP4_SEEK_METHOD_PREVIOUS_SYNC;
			int returnCode = mp4Seek(mState.demux, time_offset_usec, seekDirectionStrategy, seekedToFrame);
			if (returnCode < 0)
			{
				auto msg = "Unexpected error happened whie seeking inside the video file.";
				LOG_ERROR << msg;
				throw Mp4Exception(MP4_SEEK_INSIDE_FILE_FAILED, msg);
			}
			mState.mFrameCounterIdx = seekedToFrame;
			LOG_TRACE << "Time offset usec <" << time_offset_usec << ">, seekedToFrame <" << seekedToFrame << ">";
		}

		// seek successful
		mState.end = false; // enable seeking after eof
		// reset sentEOFSignal
		sentEOSSignal = false;
		// reset waitFlag
		waitFlag = false;
		return true;
	}

	uint64_t calcReloadFileAfter()
	{
		std::chrono::time_point<std::chrono::system_clock> t = std::chrono::system_clock::now();
		auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(t.time_since_epoch());
		uint64_t val = dur.count();
		uint64_t reloadTSVal = val + mProps.reInitInterval * 1000;
		LOG_INFO << "nowTS <" << val << "> reloadFileAfter <" << reloadTSVal << ">";
		return reloadTSVal;
	}

	void termOpenVideo()
	{
		try
		{
			mp4_demux_close(mState.demux);
		}
		catch (...)
		{
			auto msg = "Error occured while closing the video file <" + mState.mVideoPath + ">";
			LOG_ERROR << msg;
			throw Mp4Exception(MP4_FILE_CLOSE_FAILED, msg);
		}
		mState.videotrack = -1;
		mState.metatrack = -1;
		mState.mFrameCounterIdx = 0;
	}

	bool randomSeek(uint64_t& skipTS, bool forceReopen = false) noexcept
	{
		try
		{
			randomSeekInternal(skipTS, forceReopen);
		}
		catch (Mp4_Exception& ex)
		{
			makeAndSendMp4Error(Mp4ErrorFrame::MP4_SEEK, ex.getCode(), ex.getError(), ex.getOpenFileErrorCode(), skipTS);
			return false;
		}
		catch (...)
		{
			std::string msg = "unknown error while seeking";
			makeAndSendMp4Error(Mp4ErrorFrame::MP4_SEEK, MP4_UNEXPECTED_STATE, msg, 0, skipTS);
			return false;
		}
		return true;
	}

	void makeAndSendMp4Error(int errorType, int errorCode, std::string& errorMsg, int openErrorCode, uint64_t _errorMp4TS)
	{
		LOG_ERROR << "makeAndSendMp4Error <" << errorType << "," << errorCode << "," << errorMsg << "," << openErrorCode << "," << _errorMp4TS << ">";
		frame_sp errorFrame = boost::shared_ptr<Mp4ErrorFrame>(new Mp4ErrorFrame(errorType, errorCode, errorMsg, openErrorCode, _errorMp4TS));
		/*auto serSize = errorFrame.getSerializeSize(errorFrame);
		frame_sp frame = makeFrame(serSize);*/
		//Mp4ErrorFrame::serialize(frame->data(), frame->size(), errorFrame);
		sendMp4ErrorFrame(errorFrame);
	}

	//bool randomSeek(uint64_t& skipTS, uint64_t _seekEndTS)
	//{
	//	/* Takes a timestamp and sets proper mVideoFile and mParsedFilesCount (in case new parse is required) and initNewVideo().
	//	* Also, seeks to correct frame in the mVideoFile. If seek within in the videoFile fails, moving to next available video is attempted.
	//	* If all ways to seek fails, the read state is reset.
	//	*/
	//	seekEndTS = _seekEndTS;
	//	if (!mProps.parseFS)
	//	{
	//		int seekedToFrame = -1;
	//		uint64_t skipMsecsInFile = 0;
	//		if (!mState.startTimeStamp)
	//		{
	//			LOG_ERROR << "Start timestamp is not saved in the file. Can't support seeking with timestamps.";
	//			return false;
	//		}
	//		if (skipTS < mState.startTimeStamp)
	//		{
	//			LOG_INFO << "seek time outside range. Seeking to start of video.";
	//			skipMsecsInFile = 0;
	//		}
	//		else
	//		{
	//			skipMsecsInFile = skipTS - mState.startTimeStamp;
	//		}
	//		LOG_DEBUG << "Attempting seek <" << mState.mVideoPath << "> @skipMsecsInFile <" << skipMsecsInFile << ">";
	//		uint64_t time_offset_usec = skipMsecsInFile * 1000;
	//		int returnCode = mp4Seek(mState.demux, time_offset_usec, mp4_seek_method::MP4_SEEK_METHOD_NEAREST_SYNC);
	//		mState.mFrameCounter = seekedToFrame;
	//		if (returnCode == -ENFILE)
	//		{
	//			LOG_INFO << "Query time beyond the EOF. Resuming...";
	//			return true;
	//		}
	//		if (returnCode < 0)
	//		{
	//			LOG_ERROR << "Seek failed. Unexpected error.";
	//			return false;
	//		}
	//		return true;
	//	}
	//	DemuxAndParserState tempState = mState;
	//	std::string skipVideoFile;
	//	uint64_t skipMsecsInFile;
	//	int ret = mFSParser->randomSeek(skipTS, mProps.skipDir, skipVideoFile, skipMsecsInFile);
	//	LOG_INFO << "Attempting seek <" << skipVideoFile << "> @skipMsecsInFile <" << skipMsecsInFile << ">";
	//	if (ret < 0)
	//	{
	//		LOG_ERROR << "Skip to ts <" << skipTS << "> failed. Please check skip dir <" << mProps.skipDir << ">";
	//		return false;
	//	}
	//	bool found = false;
	//	for (auto i = 0; i < mState.mParsedVideoFiles.size(); ++i)
	//	{
	//		if (mState.mParsedVideoFiles[i] == skipVideoFile)
	//		{
	//			mState.mVideoCounter = i;
	//			found = true;
	//			break;
	//		}
	//	}
	//	if (!found)
	//	{
	//		// do fresh fs parse
	//		mState.mVideoPath = skipVideoFile;
	//		mState.mVideoCounter = mState.mParsedFilesCount;
	//		mState.randomSeekParseFlag = true;
	//	}
	//	initNewVideo();
	//	if (skipMsecsInFile)
	//	{
	//		uint64_t time_offset_usec = skipMsecsInFile * 1000;
	//		int returnCode = mp4Seek(mState.demux, time_offset_usec, mp4_seek_method::MP4_SEEK_METHOD_NEAREST_SYNC);
	//		// to determine the end of video
	//		mState.mFrameCounter = seekedToFrame;
	//		if (returnCode < 0)
	//		{
	//			LOG_ERROR << "Error while skipping to ts <" << skipTS << "> failed. File <" << skipVideoFile << "> @ time <" << skipMsecsInFile << "> ms errorCode <" << returnCode << ">";
	//			LOG_ERROR << "Trying to seek to next available file...";
	//			auto nextFlag = mFSParser->getNextToVideoFileFlag();
	//			if (nextFlag < 0)
	//			{
	//				// reset the state of mux to before seek op
	//				LOG_ERROR << "No next file found <" << nextFlag << ">" << "Resetting the reader state to before seek op";
	//				mState = tempState;
	//				LOG_ERROR << "Switching back to video <" << mState.mVideoPath << ">";
	//				// hot fix to avoid demux close attempt
	//				mState.demux = nullptr;
	//				mState.mVideoCounter -= 1;
	//				initNewVideo();
	//				return false;
	//			}
	//			mState.mVideoPath = mFSParser->getNextVideoFile();
	//			mState.mVideoCounter = mState.mParsedFilesCount;
	//			mState.randomSeekParseFlag = true;
	//			initNewVideo();
	//		}
	//	}
	//	return true;
	//}

	bool isOpenVideoFinished()
	{
		if (mState.direction && (mState.mFrameCounterIdx >= mState.mFramesInVideo))
		{
			std::cout <<  "COMING INSIDE IS OPEN VIDEO FINISHED " << mState.mFrameCounterIdx << "&&& " << mState.mFramesInVideo << "Direction " << mState.direction << endl;
			return true;
		}
		if (!mState.direction && mState.mFrameCounterIdx <= -1)
		{
			return true;
		}
		return false;
	}

	void readNextFrame(uint8_t* sampleFrame, uint8_t* sampleMetadataFrame, size_t& imageFrameSize, size_t& metadataFrameSize, uint64_t& frameTSInMsecs)
	{

		// isVideoFileFound is false when there is no video file in the given dir and now we check if the video file has been created.
		if (!isVideoFileFound)
		{
			currentTS = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
			if (currentTS >= recheckDiskTS)
			{
				if (!initNewVideo())
				{
					sampleFrame = nullptr;
					imageFrameSize = 0;
					recheckDiskTS = currentTS + mProps.parseFSTimeoutDuration;
					return;
				}
			}
			else
			{
				sampleFrame = nullptr;
				imageFrameSize = 0;
				return;
			}
		}
		// all frames of the open video are already read and end has not reached
		if (waitFlag)
		{
			LOG_TRACE << "readNextFrame: waitFlag <" << waitFlag << ">";
			std::chrono::time_point<std::chrono::system_clock> t = std::chrono::system_clock::now();
			auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(t.time_since_epoch());
			uint64_t nowTS = dur.count();
			LOG_INFO << "readNextFrameInternal: nowTS <" << nowTS << "> reloadFileAfter <" << reloadFileAfter << ">";
			if (reloadFileAfter > nowTS)
			{
				LOG_TRACE << "waiting....";
				return;
			}
			else // no new data on reload (wait state continues) so, re-calc a new reloadFileAfter 
			{
				reloadFileAfter = calcReloadFileAfter();
				LOG_INFO << "New reloadFileAfter <" << reloadFileAfter << "> WaitFlag <" << waitFlag << ">";
			}
		}

		// video is finished
		if (isOpenVideoFinished())
		{
			mState.end = !mProps.parseFS;
			initNewVideo(); // new video is init or mState.end is reached.
		}
		else
		{
			// if video is not finished, end and wait states are not possible. 
			// This can happen due to direction change after EOF.
			mState.end = false;
			waitFlag = false;
			sentEOSSignal = false;
		}

		if (mState.end) // no files left to be parsed
		{
			if (mProps.reInitInterval && !waitFlag) // ONLY the first time after EOR
			{
				reloadFileAfter = calcReloadFileAfter();
				waitFlag = true; // will be reset by openVideoSetPointer or randomSeek or setPlayback
				LOG_TRACE << "EOR reached in readNextFrame: waitFlag <" << waitFlag << ">";
				LOG_INFO << "Reload File After reloadFileAfter <" << reloadFileAfter << ">";
				//return nullptr;
			}
			if (!sentEOSSignal)
			{

				auto frame = frame_sp(new EoSFrame(EoSFrame::EoSFrameType::MP4_PLYB_EOS, mState.frameTSInMsecs));
				sendEOS(frame); // just send once
				sentEOSSignal = true;
			}
			return;
		}

		// skip sending one frame in same step when seek reaches EOS
		if (seekReachedEOF)
		{
			seekReachedEOF = false;
			return;
		}

		if (mState.has_more_video)
		{
			if (mState.direction)
			{
				ret = mp4_demux_get_track_sample(mState.demux,
					mState.video.id,
					1,
					sampleFrame,
					imageFrameSize,
					sampleMetadataFrame,
					metadataFrameSize,
					&mState.sample);
				++mState.mFrameCounterIdx;
			}
			else
			{
				ret = mp4_demux_get_track_sample_rev(mState.demux,
					mState.video.id,
					1,
					sampleFrame,
					imageFrameSize,
					sampleMetadataFrame,
					metadataFrameSize,
					&mState.sample);
				--mState.mFrameCounterIdx;
			}

			/* To get only info about the frames
			ret = mp4_demux_get_track_sample(
				demux, id, 1, NULL, 0, NULL, 0, &sample);
			*/

			if (ret != 0 || mState.sample.size == 0)
			{
				LOG_INFO << "<" << ret << "," << mState.sample.size << "," << mState.sample.metadata_size << ">";
				mState.has_more_video = 0;
				if (!mProps.parseFS)
				{
					if (!sentEOSSignal)
					{
						auto frame = frame_sp(new EoSFrame(EoSFrame::EoSFrameType::MP4_PLYB_EOS, mState.frameTSInMsecs));
						sendEOS(frame); // just send once
						sentEOSSignal = true;
					}
				}
				return;
			}

			// get the frame timestamp
			uint64_t sample_ts_usec = mp4_sample_time_to_usec(mState.sample.dts, mState.video.timescale);
			frameTSInMsecs = mState.resolvedStartingTS + (sample_ts_usec / 1000);
			mState.frameTSInMsecs = frameTSInMsecs;
			LOG_TRACE << "readNextFrame frameTS <" << frameTSInMsecs << ">";
			imageFrameSize = static_cast<size_t>(mState.sample.size);
			// for metadata to be ignored - we will have metadata_buffer = nullptr and size = 0
			return;
		}
		return;
	}
	Mp4ReaderSourceProps mProps;
protected:

	struct DemuxAndParserState
	{
		DemuxAndParserState()
		{
			resetState();
		}

		void resetState()
		{
			demux = nullptr;
			videotrack = -1;
			metatrack = -1;
			ntracks = -1;
			startTimeStampFromFile = 0;
			resolvedStartingTS = 0;
			endTS = 0;
			frameTSInMsecs = 0;
			mFrameCounterIdx = 0;
			mFramesInVideo = 0;
			mVideoPath = "";
			mSerFormatVersion = "";
			speed = 1;
			direction = true;
			end = false;
		}

		mp4_demux* demux = nullptr;
		mp4_track_info info, video;
		mp4_track_sample sample;
		std::string mSerFormatVersion = "";
		int has_more_video;
		int videotrack = -1;
		int metatrack = -1;
		int ntracks = -1;
		uint64_t endTS;
		uint64_t startTimeStamp = 0;
		uint32_t mParsedFilesCount = 0;
		uint32_t mVideoCounter = 0;
		uint32_t mFrameCounter = 0;
		int32_t mFramesInVideo = 0;
		uint64_t resolvedStartingTS;
		uint64_t frameTSInMsecs;
		std::vector<std::string> mParsedVideoFiles;
		uint64_t startTimeStampFromFile;
		std::string mVideoPath = "";
		int32_t mFrameCounterIdx;
		bool randomSeekParseFlag = false;
		bool end = false;
		Mp4ReaderSourceProps props;
		float speed;
		bool direction;
		//bool end;
	} mState;
	uint64_t openVideoStartingTS = 0;
	uint64_t reloadFileAfter = 0;
	uint64_t seekEndTS = 9999999999999;
	int seekedToFrame = -1;
	bool isVideoFileFound = true;
	uint64_t currentTS = 0;
	bool sentEOSSignal = false;
	bool seekReachedEOF = false;
	bool waitFlag = false;
	uint64_t recheckDiskTS = 0;
	boost::shared_ptr<OrderedCacheOfFiles> cof;
	framemetadata_sp updatedEncodedImgMetadata;
	/*
		mState.end = true is possible only in two cases:
		- if parseFS found no more relevant files on the disk
		- parseFS is disabled and intial video has finished playing
	*/
public:
	int mWidth = 0;
	int mHeight = 0;
	int ret;
	std::function<frame_sp(size_t size, string& pinId)> makeFrame;
	std::function<void(frame_sp frame)> sendEOS;
	std::function<frame_sp(frame_sp& bigFrame, size_t& size, string& pinId)> makeframe;
	boost::shared_ptr<FileStructureParser> mFSParser;
	std::function<void(frame_sp& errorFrame)> sendMp4ErrorFrame;
	std::function<void(framemetadata_sp metadata)> mSetMetadata;
	std::string h264ImagePinId;
	std::string encodedImagePinId;
	std::string mp4FramePinId;
};

class Mp4readerDetailJpeg : public Mp4readerDetailAbs
{
public:
	Mp4readerDetailJpeg(Mp4ReaderSourceProps& props, std::function<frame_sp(size_t size, std::string& pinId)> _makeFrame,
		std::function<frame_sp(frame_sp& bigFrame, size_t& size, string& pinId)> _makeframe, std::function<void(frame_sp frame)> _sendEOS, std::function<void(framemetadata_sp metadata)> _setMetadata, std::function<void(frame_sp& frame)> _sendMp4ErrorFrame) : Mp4readerDetailAbs(props, _makeFrame, _makeframe, _sendEOS, _setMetadata, _sendMp4ErrorFrame)
	{}
	~Mp4readerDetailJpeg() { mp4_demux_close(mState.demux); }
	void setMetadata();
	bool produceFrames(frame_container& frames);
	void prependSpsPps(boost::asio::mutable_buffer& iFrameBuffer) {}
	void sendEndOfStream() {}
	int mp4Seek(mp4_demux* demux, uint64_t time_offset_usec, mp4_seek_method syncType, int &seekedToFrame);
};

class Mp4readerDetailH264 : public Mp4readerDetailAbs
{
public:
	Mp4readerDetailH264(Mp4ReaderSourceProps& props, std::function<frame_sp(size_t size, string& pinId)> _makeFrame,
		std::function<frame_sp(frame_sp& bigFrame, size_t& size, string& pinId)> _makeframe, std::function<void(frame_sp frame)> _sendEOS, std::function<void(framemetadata_sp metadata)> _setMetadata, std::function<void(frame_sp& frame)> _sendMp4ErrorFrame) : Mp4readerDetailAbs(props, _makeFrame, _makeframe, _sendEOS, _setMetadata, _sendMp4ErrorFrame)
	{}
	~Mp4readerDetailH264() { mp4_demux_close(mState.demux); }
	void setMetadata();
	bool produceFrames(frame_container& frames);
	void prependSpsPps(boost::asio::mutable_buffer& iFrameBuffer);
	void sendEndOfStream();
	int mp4Seek(mp4_demux* demux, uint64_t time_offset_usec, mp4_seek_method syncType, int &seekedToFrame);
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
	auto metadata = framemetadata_sp(new EncodedImageMetadata(mWidth, mHeight));
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

int Mp4readerDetailJpeg::mp4Seek(mp4_demux* demux, uint64_t time_offset_usec, mp4_seek_method syncType, int &seekedToFrame)
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
	size_t imageActualSize = 5 * 1024 * 1024;
	size_t metadataActualSize = 0;
	uint64_t frameTSInMsecs;
	readNextFrame(sampleFrame, sampleMetadataFrame, imageActualSize, metadataActualSize, frameTSInMsecs);

	if (!sampleFrame || imageActualSize == 5 * 1024 * 1024)
	{
		return true;
	}
	
	auto trimmedImgFrame = makeframe(imgFrame, imageActualSize, encodedImagePinId);

	trimmedImgFrame->timestamp = frameTSInMsecs;

	/*if (seekEndTS <= frameTSInMsecs)
	{
		return true;
	}*/

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
	auto metadata = framemetadata_sp(new H264Metadata(mWidth, mHeight));
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

int Mp4readerDetailH264::mp4Seek(mp4_demux* demux, uint64_t time_offset_usec, mp4_seek_method syncType, int &seekedToFrame)
{
	auto ret = mp4_demux_seek(demux, time_offset_usec, syncType, &seekedToFrame);
	return ret;
}

void Mp4readerDetailH264::sendEndOfStream()
{
	auto frame = frame_sp(new EoSFrame(EoSFrame::EoSFrameType::MP4_SEEK_EOS, 0));
	sendEOS(frame);
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
	size_t imageActualSize = 5 * 1024 * 1024;

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
	uint64_t frameTSInMsecs;
	readNextFrame(sampleFrame, sampleMetadataFrame, imageActualSize, metadataActualSize, frameTSInMsecs);

	if (!sampleFrame || imageActualSize == 5 * 1024 * 1024)
	{
		return true;
	}

	auto trimmedImgFrame = makeframe(imgFrame, imageActualSize, h264ImagePinId);
	auto tempBuffer = const_buffer(trimmedImgFrame->data(), trimmedImgFrame->size());
	auto ret = H264Utils::parseNalu(tempBuffer);
	short typeFound;
	const_buffer spsBuff, ppsBuff;
	tie(typeFound, spsBuff, ppsBuff) = ret; 

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
			[&](frame_sp frame) 
			{return Module::sendEOS(frame); },
			[&](framemetadata_sp metadata)
			{ return setMetadata(metadata); },
			[&](frame_sp& frame) {return Module::sendMp4ErrorFrame(frame); }));
		encodedImageMetadata = boost::shared_ptr<EncodedImageMetadata>(new EncodedImageMetadata());
	}
	else if (mFrameType == FrameMetadata::FrameType::H264_DATA)
	{
		mDetail.reset(new Mp4readerDetailH264(props,
			[&](size_t size, string& pinId)
			{ return makeFrame(size, pinId); },
			[&](frame_sp& frame, size_t& size, string& pinId)
			{ return makeFrame(frame, size, pinId); },
			[&](frame_sp frame) 
			{return Module::sendEOS(frame); },
			[&](framemetadata_sp metadata)
			{ return setMetadata(metadata); },
			[&](frame_sp& frame) {return Module::sendMp4ErrorFrame(frame); }));
	}
	mDetail->Init();
	mDetail->encodedImagePinId = encodedImagePinId;
	mDetail->h264ImagePinId = h264ImagePinId;
	mDetail->mp4FramePinId = mp4FramePinId;

	return true;
}

std::map<std::string, std::pair<uint64_t, uint64_t>> Mp4ReaderSource::getCacheSnapShot()
{
	return mDetail->getSnapShot();
}

bool Mp4ReaderSource::refreshCache()
{
	return mDetail->refreshCache();
}

std::string Mp4ReaderSource::getOpenVideoPath()
{
	if (mDetail)
	{
		return mDetail->getOpenVideoPath();
	}
	return "";
}

bool Mp4ReaderSource::getVideoRangeFromCache(std::string videoFile, uint64_t& start_ts, uint64_t& end_ts)
{
	return mDetail->getVideoRangeFromCache(videoFile, start_ts, end_ts);
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
void Mp4ReaderSource::setMetadata(framemetadata_sp metadata)
{
	if (!metadata->isSet())
	{
		return;
	}
	auto newMetadata = FrameMetadataFactory::downcast<EncodedImageMetadata>(metadata);
	auto outMetadata = FrameMetadataFactory::downcast<EncodedImageMetadata>(encodedImageMetadata);
	outMetadata->setData(*newMetadata);
	mHeight = outMetadata->getHeight();
	mWidth = outMetadata->getWidth();
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

bool Mp4ReaderSource::changePlayback(float speed, bool direction)
{
	PlayPauseCommand ppc(speed, direction);
	return queuePlayPauseCommand(ppc);
}

bool Mp4ReaderSource::handleCommand(Command::CommandType type, frame_sp& frame)
{
	if (type == Command::CommandType::Seek)
	{
		Mp4SeekCommand seekCmd;
		getCommand(seekCmd, frame);
		return mDetail->randomSeek(seekCmd.seekStartTS, seekCmd.forceReopen);
	}
	else
	{
		return Module::handleCommand(type, frame);
	}
}

bool Mp4ReaderSource::handlePausePlay(float speed, bool direction)
{
	mDetail->setPlayback(speed, direction);
	return Module::handlePausePlay(speed, direction);
}

bool Mp4ReaderSource::randomSeek(uint64_t seekStartTS, uint64_t seekEndTS)
{
	Mp4SeekCommand cmd(seekStartTS, seekEndTS);
	return queueCommand(cmd);
}

bool Mp4ReaderSource::randomSeek(uint64_t seekStartTS)
{
	//Mp4SeekCommand cmd(seekStartTS);
	//return queueCommand(cmd);
	return true;
}

bool Mp4ReaderSource::randomSeek(uint64_t skipTS, bool forceReopen)
{
	Mp4SeekCommand cmd(skipTS, forceReopen);
	return queueCommand(cmd);
}
