#include "Mp4ReaderSource.h"
#include "FrameMetadata.h"
#include "Mp4VideoMetadata.h"
#include "EncodedImageMetadata.h"
#include "H264Metadata.h"
#include "Frame.h"
#include "Command.h"
#include "libmp4.h"
#include "H264Utils.h"
#include "OrderedCacheOfFiles.h"
#include "AIPExceptions.h"
#include "Mp4ErrorFrame.h"
#include "Module.h"
#include "AbsControlModule.h"


class Mp4ReaderDetailAbs
{
public:
	Mp4ReaderDetailAbs(Mp4ReaderSourceProps& props, std::function<frame_sp(size_t size, string& pinId)> _makeFrame,
		std::function<frame_sp(frame_sp& bigFrame, size_t& size, string& pinId)> _makeFrameTrim, std::function<void(frame_sp frame)> _sendEOS,
		std::function<void(std::string& pinId, framemetadata_sp& metadata)> _setMetadata, std::function<void(frame_sp& errorFrame)> _sendMp4ErrorFrame,
		std::function<void(Mp4ReaderSourceProps& props)> _setProps)
	{
		setProps(props);
		makeFrame = _makeFrame;
		makeFrameTrim = _makeFrameTrim;
		sendEOS = _sendEOS;
		mSetMetadata = _setMetadata;
		sendMp4ErrorFrame = _sendMp4ErrorFrame;
		setMp4ReaderProps = _setProps;
		cof = boost::shared_ptr<OrderedCacheOfFiles>(new OrderedCacheOfFiles(mProps.skipDir));
	}

	~Mp4ReaderDetailAbs()
	{
	}
	virtual void setMetadata()
	{
		auto mp4FrameMetadata = framemetadata_sp(new Mp4VideoMetadata("v_1_0"));
		auto serFormatVersion = getSerFormatVersion();
		auto mp4VideoMetadata = FrameMetadataFactory::downcast<Mp4VideoMetadata>(mp4FrameMetadata);
		mp4VideoMetadata->setData(serFormatVersion);
	}

	virtual void sendEndOfStream() = 0;
	virtual bool produceFrames(frame_container& frames) = 0;
	virtual int mp4Seek(mp4_demux* demux, uint64_t time_offset_usec, mp4_seek_method syncType, int& seekedToFrame) = 0;
	virtual int getGop() = 0;

	bool Init()
	{
		sentEOSSignal = false;
		auto filePath = boost::filesystem::path(mState.mVideoPath);
		if (filePath.extension() != ".mp4")
		{
			if (!cof->probe(filePath, mState.mVideoPath))
			{
				LOG_DEBUG << "Mp4 file is not present" << ">";
				isVideoFileFound = false;
				return true;
			}
			isVideoFileFound = true;
		}
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
		return initNewVideo(true); // enable firstOpenAfterinit
	}

	void updateMstate(Mp4ReaderSourceProps& props, std::string videoPath)
	{
		mState.direction = props.direction;
		mState.mVideoPath = videoPath;
		mProps = props;
		mState.end = false;
		if(boost::filesystem::path(videoPath).extension() == ".mp4")
		{
			isVideoFileFound = true;
		}
	}

	void setProps(Mp4ReaderSourceProps& props)
	{
		std::string tempVideoPath;
		std::chrono::time_point<std::chrono::system_clock> t = std::chrono::system_clock::now();
		auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(t.time_since_epoch());
		uint64_t nowTS = dur.count();
		reloadFileAfter = calcReloadFileAfter();
		// To check if the video file path is correct
		try
		{
			auto canonicalVideoPath = boost::filesystem::canonical(props.videoPath);
			tempVideoPath = canonicalVideoPath.string();
		}
		catch (...)
		{
			auto msg = "Video File name not in proper format.Check the filename sent as props. \
					If you want to read a file with custom name instead, please disable parseFS flag.";
			LOG_ERROR << msg;
			return;
		}
		
		// If the video path is a custom file - don't parse just open the video, cof check says that not to open video if setProps if called from module init(). 
		if (!props.parseFS && cof)
		{
			cof->clearCache();
			if (tempVideoPath == mState.mVideoPath)
			{
				updateMstate(props, tempVideoPath);
				return;
			}
			updateMstate(props, tempVideoPath);
			initNewVideo();
		}

		std::string tempSkipDir;
		if (boost::filesystem::path(tempVideoPath).extension() == ".mp4")
		{
			tempSkipDir = boost::filesystem::path(tempVideoPath).parent_path().parent_path().parent_path().string();
		}
		else
		{
			tempSkipDir = boost::filesystem::path(tempVideoPath).string();
		}
		if (props.parseFS && mProps.skipDir == tempSkipDir && mState.mVideoPath != "")
		{
			if (mProps.videoPath == props.videoPath)
				mProps = props;
			LOG_ERROR << "The root dir is same and only file path has changed, Please use SEEK functionality instead for this use case!, cannot change props";
			return;
		}
		
		if (props.parseFS && mProps.skipDir != tempSkipDir && mState.mVideoPath != "")
		{
			sentEOSSignal = false;

			if (boost::filesystem::path(tempVideoPath).extension() != ".mp4")
			{
				if (!cof->probe(tempVideoPath, mState.mVideoPath))
				{
					LOG_DEBUG << "Mp4 file is not present" << ">";
					isVideoFileFound = false;
					return;
				}
				isVideoFileFound = true;
				tempVideoPath = mState.mVideoPath;
			}

			auto boostVideoTS = boost::filesystem::path(tempVideoPath).stem().string();
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
			
			//check if root has changed
			cof = boost::shared_ptr<OrderedCacheOfFiles>(new OrderedCacheOfFiles(tempSkipDir));
			cof->clearCache();
			cof->parseFiles(start_parsing_ts, props.direction, true, false);
			//parse successful - update mState and skipDir with current root dir
			updateMstate(props, tempVideoPath);
			mProps.skipDir = tempSkipDir;
			initNewVideo(true);

			return;
		}

		// It comes here when setProps is called during Module startup
		updateMstate(props, tempVideoPath);
	}

	std::string getOpenVideoPath()
	{
		return mState.mVideoPath;
	}

	int32_t getOpenVideoFrameCount()
	{
		return mState.mFramesInVideo;
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
			mDirection = _direction;
			setMetadata();
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
				mState.demux = nullptr;
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

			// in case race conditions happen between writer and reader (videotrack not found etc) - use code will retry
		auto filePath = boost::filesystem::path(mState.mVideoPath);
		if (filePath.extension() != ".mp4")
		{
			if (!cof->probe(filePath, mState.mVideoPath))
			{
				LOG_DEBUG << "Mp4 file is not present" << ">";
				isVideoFileFound = false;
				return true;
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
						if(ex.getError() == "Reached End of Cache in fwd play.")
						{
							// send command
							if(!mState.sentCommandToControlModule && controlModule != nullptr)
							{
								bool goLive = true;
								bool priority = true;
								boost::shared_ptr<AbsControlModule>ctl = boost::dynamic_pointer_cast<AbsControlModule>(controlModule);
								ctl->handleGoLive(goLive, priority);
								LOG_TRACE<<"Sending command to mmq";
								mState.sentCommandToControlModule = true;
							}
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

	/*
	throws MP4_OPEN_FAILED_EXCEPTION, MP4_MISSING_VIDEOTRACK, MP4_MISSING_START_TS,
	MP4_TIME_RANGE_FETCH_FAILED, MP4_SET_POINTER_END_FAILED
	*/
	void openVideoSetPointer(std::string& filePath)
	{
		if (mState.demux)
		{
			termOpenVideo();
		}

		LOG_INFO << "opening video <" << filePath << ">";
		ret = mp4_demux_open(filePath.c_str(), &mState.demux);
		if (ret < 0)
		{
			//TODO: Behaviour yet to be decided in case a file is deleted while it is cached, generating a hole in the cache.
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
				mDirection = mState.direction;
				mDurationInSecs = mState.info.duration / mState.info.timescale;
				mFPS = mState.mFramesInVideo / mDurationInSecs;
				// todo: Implement a way for mp4reader to update FPS when opening a new video in parseFS enabled mode. Must not set parseFS disabled in a loop.
				mProps.fps = mFPS;
				auto gop = getGop();
				mProps.fps = mFPS * playbackSpeed;
				if(playbackSpeed == 8 || playbackSpeed == 16 || playbackSpeed == 32)
				{
					if (gop)
					{
						mProps.fps = mProps.fps / gop;
					}
				}
				setMp4ReaderProps(mProps);
				if (controlModule != nullptr)
				{
					DecoderPlaybackSpeed cmd;
					cmd.playbackSpeed = playbackSpeed;
					cmd.playbackFps = mFPS;
					cmd.gop = gop;
					bool priority = true;
					boost::shared_ptr<AbsControlModule>ctl = boost::dynamic_pointer_cast<AbsControlModule>(controlModule);
					ctl->handleDecoderSpeed(cmd, priority);
				}
			}
		}

		if (mState.videotrack == -1)
		{
			auto msg = "No Videotrack found in the video <" + mState.mVideoPath + ">";
			LOG_ERROR << msg;
			std::string previousFile;
			std::string nextFile;
			cof->getPreviousAndNextFile(mState.mVideoPath, previousFile, nextFile);
			throw Mp4ExceptionNoVideoTrack(MP4_MISSING_VIDEOTRACK, msg, previousFile, nextFile);
		}

		// starting timestamp of the video will either come from the video name or the header
		if (mState.startTimeStampFromFile)
		{
			mState.resolvedStartingTS = mState.startTimeStampFromFile;
		}
		else
		{
			auto boostVideoTS = boost::filesystem::path(mState.mVideoPath).stem().string();
			try
			{
				mState.resolvedStartingTS = std::stoull(boostVideoTS);
			}
			catch (std::invalid_argument)
			{
				auto msg = "unexpected: starting ts not found in video name or metadata";
				LOG_ERROR << msg;
				throw Mp4Exception(MP4_MISSING_START_TS, msg);
			}
		}

		// update metadata
		setMetadata();

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
		mState.sentCommandToControlModule = false;
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
			isMp4SeekFrame = true;
			setMetadata();
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
		if (!isVideoFileFound)
		{
			if (!cof->probe(boost::filesystem::path(mState.mVideoPath), mState.mVideoPath))
			{
				return false;
			}
			isVideoFileFound = true;
		}
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
		// prependSpsPps
		mState.shouldPrependSpsPps = true;
		isMp4SeekFrame = true;
		setMetadata();
		LOG_INFO << "seek successfull";
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
			mState.demux = nullptr;
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
			if(ex.getCode() == MP4_MISSING_VIDEOTRACK)
			{
				if ((controlModule != nullptr))
				{
					// Stubbing the eventual application's control module & the handleMp4MissingVideotrack method
					boost::shared_ptr<AbsControlModule>ctl = boost::dynamic_pointer_cast<AbsControlModule>(controlModule);
					ctl->handleMp4MissingVideotrack(ex.getPreviousFile(), ex.getNextFile());
				}
				return false;
			}
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

	void makeAndSendMp4Error(int errorType, int errorCode, std::string errorMsg, int openErrorCode, uint64_t _errorMp4TS)
	{
		LOG_ERROR << "makeAndSendMp4Error <" << errorType << "," << errorCode << "," << errorMsg << "," << openErrorCode << "," << _errorMp4TS << ">";
		frame_sp errorFrame = boost::shared_ptr<Mp4ErrorFrame>(new Mp4ErrorFrame(errorType, errorCode, errorMsg, openErrorCode, _errorMp4TS));
		sendMp4ErrorFrame(errorFrame);
	}

	bool isOpenVideoFinished()
	{
		if (mState.direction && (mState.mFrameCounterIdx >= mState.mFramesInVideo))
		{
			return true;
		}
		if (!mState.direction && mState.mFrameCounterIdx <= -1)
		{
			return true;
		}
		return false;
	}

	void readNextFrame(frame_sp& imgFrame, frame_sp& metadetaFrame, size_t& imgSize, size_t& metadataSize, uint64_t& frameTSInMsecs, int32_t& mp4FIndex) noexcept
	{
		try
		{
			readNextFrameInternal(imgFrame, metadetaFrame, imgSize, metadataSize, frameTSInMsecs, mp4FIndex);
		}
		catch (Mp4_Exception& ex)
		{
			if(ex.getCode() == MP4_MISSING_VIDEOTRACK)
			{
				if ((controlModule != nullptr))
				{
					// Stubbing the eventual application's control module & the handleMp4MissingVideotrack method
					boost::shared_ptr<AbsControlModule>ctl = boost::dynamic_pointer_cast<AbsControlModule>(controlModule);
					ctl->handleMp4MissingVideotrack(ex.getPreviousFile(), ex.getNextFile());
				}
				return;
			}
			imgSize = 0;
			// send the last frame timestamp 
			makeAndSendMp4Error(Mp4ErrorFrame::MP4_STEP, ex.getCode(), ex.getError(), ex.getOpenFileErrorCode(), mState.frameTSInMsecs);
			return;
		}
		catch (...)
		{
			imgSize = 0;
			std::string msg = "uknown error in readNextFrame";
			makeAndSendMp4Error(Mp4ErrorFrame::MP4_STEP, MP4_UNEXPECTED_STATE, msg, 0, mState.frameTSInMsecs);
			return;
		}
	}

	void readNextFrameInternal(frame_sp& imgFrame, frame_sp& metadetaFrame, size_t& imageFrameSize, size_t& metadataFrameSize, uint64_t& frameTSInMsecs, int32_t& mp4FIndex)
	{
		if (!isVideoFileFound)
		{
			currentTS = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
			if (currentTS >= recheckDiskTS)
			{
				if (!cof->probe(boost::filesystem::path(mState.mVideoPath), mState.mVideoPath))
				{
					imgFrame = nullptr;
					imageFrameSize = 0;
					recheckDiskTS = currentTS + mProps.parseFSTimeoutDuration;
					return;
				}
			}
			else
			{
				imgFrame = nullptr;
				imageFrameSize = 0;
				return;
			}
		}
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
			uint8_t* sampleFrame = static_cast<uint8_t*>(imgFrame->data());
			uint8_t* sampleMetadataFrame = static_cast<uint8_t*>(metadetaFrame->data());

			uint32_t imageSize = mProps.biggerFrameSize;
			uint32_t metadataSize = mProps.biggerMetadataFrameSize;
			if (mState.direction)
			{
				ret = mp4_demux_get_track_sample(mState.demux,
					mState.video.id,
					1,
					sampleFrame,
					imageSize,
					sampleMetadataFrame,
					metadataSize,
					&mState.sample);
				mp4FIndex = mState.mFrameCounterIdx++;
			}
			else
			{
				ret = mp4_demux_get_track_sample_rev(mState.demux,
					mState.video.id,
					1,
					sampleFrame,
					imageSize,
					sampleMetadataFrame,
					metadataSize,
					&mState.sample);
				mp4FIndex = mState.mFrameCounterIdx--;
			}

			/* To get only info about the frames
			ret = mp4_demux_get_track_sample(
				demux, id, 1, NULL, 0, NULL, 0, &sample);
			*/

			/* check the buffer size props */
			if (mState.sample.size > mProps.biggerFrameSize)
			{
				std::string msg = "Buffer size too small. Please check maxImgFrameSize property. maxImgFrameSize <" + std::to_string(mProps.biggerFrameSize) + "> frame size <" + std::to_string(mState.sample.size) + ">";
				throw Mp4Exception(MP4_BUFFER_TOO_SMALL, msg);
			}
			if (mState.sample.metadata_size > mProps.biggerMetadataFrameSize)
			{
				std::string msg = "Buffer size too small. Please check maxMetadataFrameSize property. maxMetadataFrameSize <" + std::to_string(mProps.biggerMetadataFrameSize) + "> frame size <" + std::to_string(mState.sample.metadata_size) + ">";
				throw Mp4Exception(MP4_BUFFER_TOO_SMALL, msg);
			}

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
			imageFrameSize += static_cast<size_t>(mState.sample.size);
			metadataFrameSize = static_cast<size_t>(mState.sample.metadata_size);
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
		bool shouldPrependSpsPps = false;
		bool foundFirstReverseIFrame = false;
		bool end = false;
		Mp4ReaderSourceProps props;
		float speed;
		bool direction;
		bool sentCommandToControlModule = false;
	} mState;
	uint64_t openVideoStartingTS = 0;
	uint64_t reloadFileAfter = 0;
	int seekedToFrame = -1;
	bool isVideoFileFound = true;
	uint64_t currentTS = 0;
	bool sentEOSSignal = false;
	bool seekReachedEOF = false;
	bool waitFlag = false;
	uint64_t recheckDiskTS = 0;
	boost::shared_ptr<OrderedCacheOfFiles> cof;
	framemetadata_sp updatedEncodedImgMetadata;
	framemetadata_sp mH264Metadata;
	/*
		mState.end = true is possible only in two cases:
		- if parseFS found no more relevant files on the disk
		- parseFS is disabled and intial video has finished playing
	*/
public:
	int mWidth = 0;
	int mHeight = 0;
	bool mDirection;
	bool isMp4SeekFrame = false;
	int ret;
	double mFPS = 0;
	float playbackSpeed = 1;
	float framesToSkip = 0;
	double mDurationInSecs = 0;
	std::function<frame_sp(size_t size, string& pinId)> makeFrame;
	std::function<void(frame_sp frame)> sendEOS;
	std::function<frame_sp(frame_sp& bigFrame, size_t& size, string& pinId)> makeFrameTrim;
	std::function<void(frame_sp& errorFrame)> sendMp4ErrorFrame;
	std::function<void(std::string& pinId, framemetadata_sp& metadata)> mSetMetadata;
	std::function<void(Mp4ReaderSourceProps& props)> setMp4ReaderProps;
	std::string h264ImagePinId;
	std::string encodedImagePinId;
	std::string metadataFramePinId;
	boost::shared_ptr<Module> controlModule = nullptr;
};

class Mp4ReaderDetailJpeg : public Mp4ReaderDetailAbs
{
public:
	Mp4ReaderDetailJpeg(Mp4ReaderSourceProps& props, std::function<frame_sp(size_t size, std::string& pinId)> _makeFrame,
		std::function<frame_sp(frame_sp& bigFrame, size_t& size, string& pinId)> _makeFrameTrim, std::function<void(frame_sp frame)> _sendEOS, std::function<void(std::string& pinId, framemetadata_sp& metadata)> _setMetadata, std::function<void(frame_sp& frame)> _sendMp4ErrorFrame, std::function<void(Mp4ReaderSourceProps& props)> _setProps) : Mp4ReaderDetailAbs(props, _makeFrame, _makeFrameTrim, _sendEOS, _setMetadata, _sendMp4ErrorFrame, _setProps)
	{}
	~Mp4ReaderDetailJpeg() {}
	void setMetadata();
	bool produceFrames(frame_container& frames);
	void sendEndOfStream() {}
	int mp4Seek(mp4_demux* demux, uint64_t time_offset_usec, mp4_seek_method syncType, int& seekedToFrame);
	int getGop();
};

class Mp4ReaderDetailH264 : public Mp4ReaderDetailAbs
{
public:
	Mp4ReaderDetailH264(Mp4ReaderSourceProps& props, std::function<frame_sp(size_t size, string& pinId)> _makeFrame,
		std::function<frame_sp(frame_sp& bigFrame, size_t& size, string& pinId)> _makeFrameTrim, std::function<void(frame_sp frame)> _sendEOS, std::function<void(std::string& pinId, framemetadata_sp& metadata)> _setMetadata, std::function<void(frame_sp& frame)> _sendMp4ErrorFrame, std::function<void(Mp4ReaderSourceProps& props)> _setProps) : Mp4ReaderDetailAbs(props, _makeFrame, _makeFrameTrim, _sendEOS, _setMetadata, _sendMp4ErrorFrame, _setProps)
	{}
	~Mp4ReaderDetailH264() {}
	void setMetadata();
	void readSPSPPS();
	bool produceFrames(frame_container& frames);
	void prependSpsPps(uint8_t* iFrameBuffer);
	void sendEndOfStream();
	int mp4Seek(mp4_demux* demux, uint64_t time_offset_usec, mp4_seek_method syncType, int& seekedToFrame);
	int getGop();
private:
	uint8_t* sps = nullptr;
	uint8_t* pps = nullptr;
	size_t spsSize = 0;
	size_t ppsSize = 0;
	bool seekedToEndTS = false;
};

void Mp4ReaderDetailJpeg::setMetadata()
{
	auto metadata = framemetadata_sp(new EncodedImageMetadata(mWidth, mHeight));
	if (!metadata->isSet())
	{
		return;
	}
	auto encodedMetadata = FrameMetadataFactory::downcast<EncodedImageMetadata>(metadata);
	encodedMetadata->setData(*encodedMetadata);
	Mp4ReaderDetailAbs::setMetadata();
	// set at Module level
	mSetMetadata(encodedImagePinId, metadata);
}

int Mp4ReaderDetailJpeg::mp4Seek(mp4_demux* demux, uint64_t time_offset_usec, mp4_seek_method syncType, int& seekedToFrame)
{
	auto ret = mp4_demux_seek_jpeg(demux, time_offset_usec, syncType, &seekedToFrame);
	return ret;
}

int Mp4ReaderDetailJpeg::getGop()
{
	return 0;
}

bool Mp4ReaderDetailJpeg::produceFrames(frame_container& frames)
{
	frame_sp imgFrame = makeFrame(mProps.biggerFrameSize, encodedImagePinId);
	frame_sp metadataFrame = makeFrame(mProps.biggerMetadataFrameSize, metadataFramePinId);
	size_t imgSize = 0;
	size_t metadataSize = 0;
	int32_t mp4FIndex = 0;
	uint64_t frameTSInMsecs;

	try
	{
		readNextFrame(imgFrame, metadataFrame, imgSize, metadataSize, frameTSInMsecs, mp4FIndex);
	}
	catch (const std::exception& e)
	{
		LOG_ERROR << e.what();
		attemptFileClose();
	}

	if (!imgSize)
	{
		return true;
	}

	auto trimmedImgFrame = makeFrameTrim(imgFrame, imgSize, encodedImagePinId);
	trimmedImgFrame->timestamp = frameTSInMsecs;
	trimmedImgFrame->fIndex = mp4FIndex;

	// give recorded timestamps 
	if (!mProps.giveLiveTS)
	{
		/* recordedTS mode */
		trimmedImgFrame->timestamp = frameTSInMsecs;
	}
	else
	{
		/* getLiveTS mode */
		// get local epoch timestamp in milliseconds
		std::chrono::time_point<std::chrono::system_clock> t = std::chrono::system_clock::now();
		auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(t.time_since_epoch());
		auto nowTS = dur.count();
		trimmedImgFrame->timestamp = nowTS;
	}

	frames.insert(make_pair(encodedImagePinId, trimmedImgFrame));
	if (metadataSize)
	{
		auto trimmedMetadataFrame = makeFrameTrim(metadataFrame, metadataSize, metadataFramePinId);
		trimmedMetadataFrame->timestamp = frameTSInMsecs;
		trimmedMetadataFrame->fIndex = mp4FIndex;
		if (!mProps.giveLiveTS)
		{
			/* recordedTS mode */
			trimmedMetadataFrame->timestamp = frameTSInMsecs;
		}
		else
		{
			trimmedMetadataFrame->timestamp = trimmedImgFrame->timestamp;
		}

		frames.insert(make_pair(metadataFramePinId, trimmedMetadataFrame));
	}
	return true;
}

void Mp4ReaderDetailH264::setMetadata()
{
	mH264Metadata = framemetadata_sp(new H264Metadata(mWidth, mHeight));

	if (!mH264Metadata->isSet())
	{
		return;
	}
	auto h264Metadata = FrameMetadataFactory::downcast<H264Metadata>(mH264Metadata);
	h264Metadata->direction = mDirection;
	h264Metadata->mp4Seek = isMp4SeekFrame;
	h264Metadata->setData(*h264Metadata);

	readSPSPPS();

	Mp4ReaderDetailAbs::setMetadata();
	mSetMetadata(h264ImagePinId, mH264Metadata);
	return;
}

void Mp4ReaderDetailH264::readSPSPPS()
{
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
}

int Mp4ReaderDetailH264::mp4Seek(mp4_demux* demux, uint64_t time_offset_usec, mp4_seek_method syncType, int& seekedToFrame)
{
	auto ret = mp4_demux_seek(demux, time_offset_usec, syncType, &seekedToFrame);
	if (ret == -2)
	{
		seekedToFrame = mState.mFramesInVideo;
		ret = 0;
	}
	return ret;
}

int Mp4ReaderDetailH264::getGop()
{
	int gop = mState.info.syncSampleEntries[2] - mState.info.syncSampleEntries[1];
	return gop;
}

void Mp4ReaderDetailH264::sendEndOfStream()
{
	auto frame = frame_sp(new EoSFrame(EoSFrame::EoSFrameType::MP4_SEEK_EOS, 0));
	sendEOS(frame);
}

void Mp4ReaderDetailH264::prependSpsPps(uint8_t* iFrameBuffer)
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
	memcpy(iFrameBuffer, nalu, 4);
	iFrameBuffer += 4;
	memcpy(iFrameBuffer, sps, spsSize);
	iFrameBuffer += spsSize;
	memcpy(iFrameBuffer, nalu, 4);
	iFrameBuffer += 4;
	memcpy(iFrameBuffer, pps, ppsSize);
	iFrameBuffer += ppsSize;
}

bool Mp4ReaderDetailH264::produceFrames(frame_container& frames)
{
	frame_sp imgFrame = makeFrame(mProps.biggerFrameSize, h264ImagePinId);
	size_t imgSize = 0;
	frame_sp metadataFrame = makeFrame(mProps.biggerMetadataFrameSize, metadataFramePinId);
	size_t metadataSize = 0;
	uint64_t frameTSInMsecs;
	int32_t mp4FIndex = 0;

	try
	{
		readNextFrame(imgFrame, metadataFrame, imgSize, metadataSize, frameTSInMsecs, mp4FIndex);
	}
	catch (const std::exception& e)
	{
		LOG_ERROR << e.what();
		attemptFileClose();
	}

	if (!imgSize)
	{
		return true;
	}

	if (mState.shouldPrependSpsPps || (!mState.direction && !mState.foundFirstReverseIFrame))
	{
		boost::asio::mutable_buffer tmpBuffer(imgFrame->data(), imgFrame->size());
		auto type = H264Utils::getNALUType((char*)tmpBuffer.data());
		if (type == H264Utils::H264_NAL_TYPE_IDR_SLICE)
		{
			auto tempFrame = makeFrame(imgSize + spsSize + ppsSize + 8, h264ImagePinId);
			uint8_t* tempFrameBuffer = reinterpret_cast<uint8_t*>(tempFrame->data());
			prependSpsPps(tempFrameBuffer);
			tempFrameBuffer += spsSize + ppsSize + 8;
			memcpy(tempFrameBuffer, imgFrame->data(), imgSize);
			imgSize += spsSize + ppsSize + 8;
			imgFrame = tempFrame;
			mState.foundFirstReverseIFrame = true;
			mState.shouldPrependSpsPps = false;
		}
		else if (type == H264Utils::H264_NAL_TYPE_SEQ_PARAM)
		{
			mState.shouldPrependSpsPps = false;
			mState.foundFirstReverseIFrame = true;
		}
	}

	auto trimmedImgFrame = makeFrameTrim(imgFrame, imgSize, h264ImagePinId);

	uint8_t* frameData = reinterpret_cast<uint8_t*>(trimmedImgFrame->data());
	short nalType = H264Utils::getNALUType((char*)trimmedImgFrame->data());
	if (nalType == H264Utils::H264_NAL_TYPE_SEQ_PARAM)
	{
		frameData[3] = 0x1;
		frameData[spsSize + 7] = 0x1;
		frameData[spsSize + ppsSize + 8] = 0x0;
		frameData[spsSize + ppsSize + 9] = 0x0;
		frameData[spsSize + ppsSize + 10] = 0x0;
		frameData[spsSize + ppsSize + 11] = 0x1;
	}
	else
	{
		frameData[0] = 0x0;
		frameData[1] = 0x0;
		frameData[2] = 0x0;
		frameData[3] = 0x1;
	}

	trimmedImgFrame->timestamp = frameTSInMsecs;
	trimmedImgFrame->fIndex = mp4FIndex;

	// give recorded timestamps 
	if (!mProps.giveLiveTS)
	{
		/* recordedTS mode */
		trimmedImgFrame->timestamp = frameTSInMsecs;
	}
	else
	{
		/* getLiveTS mode */
		// get local epoch timestamp in milliseconds
		std::chrono::time_point<std::chrono::system_clock> t = std::chrono::system_clock::now();
		auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(t.time_since_epoch());
		auto nowTS = dur.count();
		trimmedImgFrame->timestamp = nowTS;
	}

	frames.insert(make_pair(h264ImagePinId, trimmedImgFrame));
	if (metadataSize)
	{
		auto trimmedMetadataFrame = makeFrameTrim(metadataFrame, metadataSize, metadataFramePinId);
		trimmedMetadataFrame->timestamp = frameTSInMsecs;
		trimmedMetadataFrame->fIndex = mp4FIndex;
		if (!mProps.giveLiveTS)
		{
			/* recordedTS mode */
			trimmedMetadataFrame->timestamp = frameTSInMsecs;
		}
		else
		{
			trimmedMetadataFrame->timestamp = trimmedImgFrame->timestamp;
		}
		frames.insert(make_pair(metadataFramePinId, trimmedMetadataFrame));
	}
	if (isMp4SeekFrame)
	{
		isMp4SeekFrame = false;
		setMetadata();
	}
	if((playbackSpeed == 8 || playbackSpeed == 16 || playbackSpeed == 32))
	{
		if(mDirection)
		{
			uint64_t nextFrameTs;
			if(!mState.sample.next_dts && mState.mFrameCounterIdx == mState.mFramesInVideo)//To handle the case when I frame is last frame of the video
			{
				uint64_t nextDts = mState.sample.dts - mState.sample.prev_sync_dts;
				nextDts += mState.sample.dts;
				uint64_t sample_ts_usec = mp4_sample_time_to_usec(nextDts, mState.video.timescale);
				nextFrameTs = mState.resolvedStartingTS + (sample_ts_usec / 1000);
			}
			else
			{
				uint64_t sample_ts_usec = mp4_sample_time_to_usec(mState.sample.next_dts, mState.video.timescale);
				nextFrameTs = mState.resolvedStartingTS + (sample_ts_usec / 1000);
			}
			nextFrameTs++;
			randomSeek(nextFrameTs);
		}
		else
		{
			frameTSInMsecs--;
			randomSeek(frameTSInMsecs);
		}
		
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
	auto outMetadata = getFirstOutputMetadata();
	auto  mFrameType = outMetadata->getFrameType();
	if (mFrameType == FrameMetadata::FrameType::ENCODED_IMAGE)
	{
		mDetail.reset(new Mp4ReaderDetailJpeg(
			props,
			[&](size_t size, string& pinId)
			{ return makeFrame(size, pinId); },
			[&](frame_sp& frame, size_t& size, string& pinId)
			{ return makeFrame(frame, size, pinId); },
			[&](frame_sp frame)
			{return Module::sendEOS(frame); },
			[&](std::string& pinId, framemetadata_sp& metadata)
			{ return setImageMetadata(pinId, metadata); },
			[&](frame_sp& frame) {return Module::sendMp4ErrorFrame(frame); },
			[&](Mp4ReaderSourceProps& props)
			{return setProps(props); }));
	}
	else if (mFrameType == FrameMetadata::FrameType::H264_DATA)
	{
		mDetail.reset(new Mp4ReaderDetailH264(props,
			[&](size_t size, string& pinId)
			{ return makeFrame(size, pinId); },
			[&](frame_sp& frame, size_t& size, string& pinId)
			{ return makeFrame(frame, size, pinId); },
			[&](frame_sp frame)
			{return Module::sendEOS(frame); },
			[&](std::string& pinId, framemetadata_sp& metadata)
			{ return setImageMetadata(pinId, metadata); },
			[&](frame_sp& frame) 
			{return Module::sendMp4ErrorFrame(frame); },
			[&](Mp4ReaderSourceProps& props)
			{return setProps(props);  }));
	}
	mDetail->encodedImagePinId = encodedImagePinId;
	mDetail->h264ImagePinId = h264ImagePinId;
	mDetail->metadataFramePinId = metadataFramePinId;
	mDetail->controlModule = controlModule;
	return mDetail->Init();
}

void Mp4ReaderSource::setImageMetadata(std::string& pinId, framemetadata_sp& metadata)
{
	Module::setMetadata(pinId, metadata);
	mWidth = mDetail->mWidth;
	mHeight = mDetail->mHeight;
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

int32_t Mp4ReaderSource::getOpenVideoFrameCount()
{
	if (mDetail)
	{
		return mDetail->getOpenVideoFrameCount();
	}
	return -1;
}

double Mp4ReaderSource::getOpenVideoFPS()
{
	if (mDetail)
	{
		return mDetail->mFPS;
	}
	return -1;
}

double Mp4ReaderSource::getOpenVideoDurationInSecs()
{
	if (mDetail)
	{
		return mDetail->mDurationInSecs;
	}
	return -1;
}

bool Mp4ReaderSource::getVideoRangeFromCache(std::string videoFile, uint64_t& start_ts, uint64_t& end_ts)
{
	return mDetail->getVideoRangeFromCache(videoFile, start_ts, end_ts);
}

bool Mp4ReaderSource::term()
{
	auto moduleRet = Module::term();
	mDetail->attemptFileClose();
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
		metadataFramePinId = Module::addOutputPin(metadata);
		return metadataFramePinId;
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
	bool direction = getPlayDirection();
	Mp4ReaderSourceProps props(mDetail->mProps.videoPath, mDetail->mProps.parseFS, mDetail->mProps.reInitInterval, direction, mDetail->mProps.readLoop, mDetail->mProps.giveLiveTS, mDetail->mProps.parseFSTimeoutDuration, mDetail->mProps.bFramesEnabled);
	bool ret = Module::handlePropsChange(frame, props);
	mDetail->setProps(props);
	return ret;
}

void Mp4ReaderSource::setProps(Mp4ReaderSourceProps& props)
{
	Module::addPropsToQueue(props, true);
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
		//LOG_ERROR<<"seek play 1 ";
		return mDetail->randomSeek(seekCmd.seekStartTS, seekCmd.forceReopen);
		//LOG_ERROR<<"seek play 2 ";
	}
	else
	{
		return Module::handleCommand(type, frame);
	}
}

bool Mp4ReaderSource::handlePausePlay(float speed, bool direction)
{
	//LOG_ERROR<<"hanlde play 1 ";
	mDetail->setPlayback(speed, direction);
	return Module::handlePausePlay(speed, direction);
	//LOG_ERROR<<"hanlde play 2 ";
}

bool Mp4ReaderSource::randomSeek(uint64_t skipTS, bool forceReopen)
{
	Mp4SeekCommand cmd(skipTS, forceReopen);
	return queueCommand(cmd);
}

void Mp4ReaderSource::setPlaybackSpeed(float _playbackSpeed)
{
	mDetail->playbackSpeed = _playbackSpeed;
}
