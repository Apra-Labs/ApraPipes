#include <boost/filesystem.hpp>
#include "Logger.h"
#include "libmp4.h"
#include "OrderedCacheOfFiles.h"
#include "AIPExceptions.h"

/* implies that upon dir change the shared pointer needs to be reset with new params */
OrderedCacheOfFiles::OrderedCacheOfFiles(std::string& video_folder, uint32_t initial_batch_size, uint32_t _lowerWaterMark, uint32_t _upperWaterMark)
{
	rootDir = video_folder;
	batchSize = initial_batch_size;
	lowerWaterMark = _lowerWaterMark;
	upperWaterMark = _upperWaterMark;
	cleanCacheOnMainThread = true;
}

uint64_t OrderedCacheOfFiles::getFileDuration(std::string& filename)
{
	auto videoIter = videoCache.find(filename);
	if (videoIter == videoCache.end())
	{
		return 0;
	}
	if (videoIter->end_ts != 0)
	{
		uint64_t diff = videoIter->end_ts - videoIter->start_ts;
		return diff;
	}
	return 0;
}

bool OrderedCacheOfFiles::fetchFromCache(std::string& videoFile, uint64_t& start_ts, uint64_t& end_ts)
{
	LOG_TRACE << "fetchFromCache called...";
	start_ts = 0;
	end_ts = 0;
	if (videoCache.empty())
	{
		return false;
	}

	try
	{
		videoFile = boost::filesystem::canonical(videoFile).string();
	}
	catch (...)
	{
		LOG_ERROR << "File not found on disk: " + videoFile;
		return false;
	}

	auto iter = videoCache.find(videoFile);
	if (iter == videoCache.end())
	{
		return false;
	}

	start_ts = iter->start_ts;
	end_ts = iter->end_ts;

	return true;
}

bool OrderedCacheOfFiles::fetchAndUpdateFromDisk(std::string videoFile, uint64_t& start_ts, uint64_t& end_ts)
{
	LOG_TRACE << "getFileStartAndEnd called...";
	start_ts = 0;
	end_ts = 0;
	if (videoCache.empty())
	{
		return false;
	}

	try
	{
		videoFile = boost::filesystem::canonical(videoFile).string();
	}
	catch (...)
	{
		LOG_ERROR << "File not found on disk: " + videoFile;
		return false;
	}

	auto iter = videoCache.find(videoFile);
	if (iter == videoCache.end())
	{
		return false;
	}

	start_ts = iter->start_ts;

	// force update regardless of end_ts != 0
	uint64_t tstart_ts, tend_ts;
	readVideoStartEnd(videoFile, tstart_ts, tend_ts);
	updateCache(videoFile, tstart_ts, tend_ts);

	end_ts = iter->end_ts;

	return true;
}

/*
* This method takes and returns a snapshot of the cache data. It is costly and should be used only rarely.
*/
std::map<std::string, std::pair<uint64_t, uint64_t>> OrderedCacheOfFiles::getSnapShot()
{
	std::map<std::string, std::pair<uint64_t, uint64_t>> snap;
	for (auto it = videoCache.begin(); it != videoCache.end(); ++it)
	{
		std::pair<std::string, std::pair<uint64_t, uint64_t>> elem;
		elem.first = it->path;
		elem.second.first = it->start_ts;
		elem.second.second = it->end_ts;
		snap.insert(elem);
	}
	return snap;
}

bool OrderedCacheOfFiles::probe(boost::filesystem::path potentialMp4File, std::string& videoName)
{
	try
	{
		boost::filesystem::is_empty(potentialMp4File);
	}
	catch (...)
	{
		return false;
	}

	auto dateDir = parseAndSortDateDir(potentialMp4File.string());

	for (auto& dateDirPath : dateDir)
	{
		auto hourDir = parseAndSortHourDir(dateDirPath.string());

		for (auto& hourDirPath : hourDir)
		{

			auto mp4Files = parseAndSortMp4Files(hourDirPath.string());

			if (mp4Files.size())
			{
				videoName = mp4Files.begin()->string();
				return true;
			}
			else
			{
				return false;
			}
		}
	}
	return false;
}

bool OrderedCacheOfFiles::getPreviousAndNextFile(std::string videoPath, std::string& previousFile, std::string& nextFile)
{
	auto videoIter = videoCache.find(videoPath);
	videoIter++;
	if (videoIter == videoCache.end())
	{
		nextFile = "";
		videoIter--;
		videoIter--;
		if(videoIter == videoCache.end())
		{
			previousFile = "";
			return false;
		}
		previousFile = videoIter->path;
		return true;
	}
	nextFile = videoIter->path;
	videoIter--;
	videoIter--;
	if (videoIter == videoCache.end())
	{
		previousFile = "";
		return false;
	}
	previousFile = videoIter->path;
	return true;
}

/*
Important Note:
**UNRELIABLE METHOD - Use ONLY if you know what you are doing.**
It is used to get the potential file at the timestamp.
It does not gurantee that the file will actually contain the timestamp
NOR does it guratee that the file is logically correct.
It has to be used in conjungtion with other methods i.e.
Use isTimeStampInFile() to confirm.
If ts is not present in the file returned by isTimeStampInFile(),
then use the getNextFileAfter() to conclude.
*/
std::string OrderedCacheOfFiles::getFileAt(uint64_t timestamp, bool direction)
{
	if (videoCache.empty())
	{
		auto msg = "Calling getFileAt() on empty cache. Call parseFiles first.";
		LOG_ERROR << msg;
		throw AIPException(MP4_OCOF_EMPTY, msg);
	}
	// Note: start_ts will never have cache miss
	auto lowerBoundIter = videoCacheStartTSIndex.lower_bound(timestamp);

	/*
	Idea1: If we find exact match with lower_bound(), simply return it.
	If not exact match, that means the sts of file > ts queried. For these cases:
		Idea2 - For bwd play, we return the lower_bound() iterator itself. I know this is wrong. It is by design to maintain abstraction.
		Idea3 - See Usage Note above to understand the abstraction we are maintaining.
		Idea4 - For fwd play, we return the previous iterator to the lower_bound()
	Corner cases:
		1. bwd play + invalid iterator i.e. end - we need to break Idea2
		2. fwd play + first iterator - we need to break Idea4
		3. bwd play + lower_bound is end() - give empty string - so that isTimeStampInFile returns false
	Detection of End Of Files:
		1. BWD Play: bwd play + lower_bound is the first value
		2. FWD Play: we cant detect EOF boundary in fwd play, bcz the last video can be of any length. Use getNextFileAfter() to confirm.
	*/

	// lower bound itself returns end iterator, cacheIteratorState.END_ITER represents the same in string
	if (!direction && lowerBoundIter == videoCacheStartTSIndex.end())
	{
		return cacheIteratorState.END_ITER; // no correct file - isTimeStampInFile will return false for empty string
	}

	// exact match
	if (lowerBoundIter->start_ts == timestamp)
	{
		return lowerBoundIter->path;
	}

	// greater than timestamp
	if (lowerBoundIter == videoCacheStartTSIndex.begin())
	{
		if (!direction)
		{
			LOG_ERROR << "this exception will be caught!!";
			throw Mp4Exception(MP4_OCOF_END, "Reached end of cache in bwd play");	// EOF in bwd play
		}

		if (direction)
		{
			return lowerBoundIter->path; // corner case in fwd direction
		}
	}

	// Note: We are intentionally returning the next file in bwd case
	if (direction)
	{
		--lowerBoundIter;
	}

	// Note: this method will always return last file in case of ts >= last file's sts
	return lowerBoundIter->path;
}

std::string OrderedCacheOfFiles::getNextFileAfter(std::string& currentFile, bool direction)
{
	// corner case of END_ITER
	if (currentFile == cacheIteratorState.END_ITER)
	{
		auto iter = videoCache.end();
		--iter;
		return iter->path;
	}
	/* Assumption from here: currentFile will always be present in the cache */
	auto iter = videoCache.find(currentFile);
	if (iter == videoCache.end())
	{
		auto msg = "currentFile <" + currentFile + "> missing in the cache.";
		LOG_ERROR << msg;
		throw Mp4Exception(MP4_OCOF_MISSING_FILE, msg);
	}

	if (direction)
	{
		// increment then check
		++iter;
		if (iter != videoCache.end())
		{
			return iter->path;
		}
		else
		{
			LOG_ERROR << "this exception will be caught!!";
			throw Mp4Exception(MP4_OCOF_END, "Reached End of Cache in fwd play.");
		}
	}
	else
	{
		// check then decrement
		if (iter != videoCache.begin())
		{
			--iter;
			return iter->path;
		}
		else
		{
			LOG_ERROR << "this exception will be caught!!";
			throw Mp4Exception(MP4_OCOF_END, "Reached End of Cache in bwd play.");
		}
	}

	// no valid next file
	return "";
}

bool OrderedCacheOfFiles::isTimeStampInFile(std::string& filePath, uint64_t timestamp)
{
	// corner case of END_ITER
	if (filePath == cacheIteratorState.END_ITER)
	{
		return false;
	}
	auto videoIter = videoCache.find(filePath);
	if (videoIter == videoCache.end())
	{
		LOG_INFO << "Unexpected: File not present in the parsed files. Please recheck.";
		return false;
	}

	if (!videoIter->end_ts) // we havent opened the video yet.
	{
		uint64_t tstart_ts, tend_ts;
		readVideoStartEnd(filePath, tstart_ts, tend_ts);
		updateCache(filePath, tstart_ts, tend_ts);
	}

	if (timestamp >= videoIter->start_ts && timestamp <= videoIter->end_ts)
	{
		return true;
	}
	return false;
}

void OrderedCacheOfFiles::readVideoStartEnd(std::string& filePath, uint64_t& start_ts, uint64_t& end_ts)
{
	// open the file
	uint64_t duration = 0;
	start_ts = 0;
	end_ts = 0;

	struct mp4_demux* demux;
	auto ret = mp4_demux_open(filePath.c_str(), &demux);
	if (ret < 0)
	{
		auto msg = "Error opening the file <" + filePath + "> libmp4 errorcode<" + std::to_string(ret) + ">";
		LOG_ERROR << msg;
		throw Mp4Exception(MP4_OPEN_FILE_FAILED, msg);
	}
	// get the video span from the file
	try
	{
		mp4_demux_time_range(demux, &start_ts, &duration);
	}
	catch (...)
	{
		auto msg = "Unexpected error occured getting time range of the video <" + filePath + ">";
		LOG_ERROR << msg;
		throw AIPException(MP4_TIME_RANGE_FETCH_FAILED, msg);
	}
	// if timestamp not present in header (returns 0) - try reading from the filename
	if (!start_ts)
	{
		try
		{
			auto fileNameTS = boost::filesystem::path(filePath).stem().string();
			start_ts = std::stoull(fileNameTS);
		}
		catch (std::invalid_argument)
		{
			auto msg = "unexpected state - starting ts not found in video name or metadata";
			LOG_ERROR << msg;
			throw Mp4Exception(MP4_UNEXPECTED_STATE, msg);
		}
	}
	// parsed start_ts and duration of the video
	end_ts = start_ts + duration;
	if (end_ts < start_ts)
	{
		auto msg = "Invalid values: end ts < start ts in videoCache entry";
		LOG_ERROR << msg;
		throw Mp4Exception(MP4_UNEXPECTED_STATE, msg);
	}
	// close the file
	mp4_demux_close(demux);

}

void OrderedCacheOfFiles::updateCache(std::string& filePath, uint64_t& start_ts, uint64_t& end_ts)
{
	auto videoIter = videoCache.find(filePath);
	if (videoIter == videoCache.end())
	{
		auto msg = "Trying to update non existing video data";
		LOG_ERROR << msg;
		throw Mp4Exception(MP4_OCOF_MISSING_FILE, msg);
	}
	if (end_ts == videoIter->end_ts)
	{
		return;
	}
	boost::mutex::scoped_lock lock(m_mutex);
	videoCache.modify(videoIter, [start_ts, end_ts](auto& entry) {entry.start_ts = start_ts;  entry.end_ts = end_ts; });
}

/* throws MP4_UNEXPECTED_STATE, MP4_OCOF_EMPTY */
bool OrderedCacheOfFiles::getRandomSeekFile(uint64_t skipTS, bool direction, uint64_t& skipMsecs, std::string& skipVideoFile)
{
	skipMsecs = 0;
	skipVideoFile = "";
	uint64_t freshParseStartTS = skipTS;
	bool queryBeforeCacheStart = false;

	// track  playback dir
	if (lastKnownPlaybackDir != direction)
	{
		lastKnownPlaybackDir = direction;
	}

	/* Perform a fresh disk parse in way to ensure that no holes are created in the cache. */
	if (direction)
	{
		if (skipTS < videoCache.begin()->start_ts)
		{
			// parse - green
			queryBeforeCacheStart = true;
			freshParseStartTS = skipTS;
		}
		else
		{
			queryBeforeCacheStart = false;
			freshParseStartTS = videoCache.rbegin()->start_ts; // start parsing from EOC in fwd dir
		}
	}
	else if (!direction)
	{
		if (skipTS > videoCache.rbegin()->start_ts)
		{
			// check if skipTS lies in the last file, then no need to do fresh Disk parse
			std::string lastFileInCache = videoCache.rbegin()->path;
			if (!isTimeStampInFile(lastFileInCache, skipTS))
			{
				// parse - green
				queryBeforeCacheStart = true;
				freshParseStartTS = skipTS;
			}
		}
		else
		{
			queryBeforeCacheStart = false;
			freshParseStartTS = videoCache.begin()->start_ts; // in bwd dir, files till freshParseStartTS will get parsed
		}
	}

	if (!queryBeforeCacheStart)
	{
		bool isSkipFileInCache = getFileFromCache(skipTS, direction, skipVideoFile);
		if (isSkipFileInCache)
		{
			bool isTsInFile = isTimeStampInFile(skipVideoFile, skipTS);
			if (isTsInFile)
			{
				auto cachedFile = videoCache.find(skipVideoFile);
				uint64_t startTS = cachedFile->start_ts;
				skipMsecs = skipTS - startTS;
			}
			return true;
		}
	}

	/* fresh parse is required - cache will be updated according to the skipTS */
	bool foundRelevantFiles = parseFiles(freshParseStartTS, direction, true, true, skipTS); // enable includeExactMatch, disableBatchSizeCheck, drop farthest from skipTS
	if (!foundRelevantFiles)
	{
		// seek fails
		return false;
	}

	// recheck bounds with updated cache
	if (direction && skipTS < videoCache.begin()->start_ts)
	{
		// green
		skipVideoFile = videoCache.begin()->path;
		// skipMsecs = 0;
		return true;
	}
	else if (!direction && skipTS > videoCache.rbegin()->start_ts)
	{
		// green
		skipVideoFile = videoCache.rbegin()->path;
		// set correct skipMsecs if skipTS is indeed inside the skipVideoFile before returning
		bool isTSInFile = isTimeStampInFile(skipVideoFile, skipTS);
		if (isTSInFile)
		{
			auto videoEntry = videoCache.find(skipVideoFile);
			uint64_t startTS = videoEntry->start_ts;
			skipMsecs = skipTS - startTS;
		}
		return true;
	}

	// check in updated cache data
	bool isSkipFileInCache = getFileFromCache(skipTS, direction, skipVideoFile);
	if (!isSkipFileInCache)
	{
		/*case: in case of fwd parse, cache might include last file on disk as relevant file,
		but, if the skipTS is not in that file, the seek fails here. */
		return false; // seek fails
		//throw Mp4Exception(MP4_UNEXPECTED_STATE, "unexpected error happened while searching for file in cache.");
	}
	// skipMsecs = 0, if the skipTS doesn't lie inside any file. Else, calc based on start_ts of skipVideoFile.
	bool isTSInFile = isTimeStampInFile(skipVideoFile, skipTS);
	if (isTSInFile)
	{
		auto videoEntry = videoCache.find(skipVideoFile);
		uint64_t startTS = videoEntry->start_ts;
		skipMsecs = skipTS - startTS;
	}
	return true;
}

/* throws MP4_OCOF_EMPTY*/
bool OrderedCacheOfFiles::getFileFromCache(uint64_t timestamp, bool direction, std::string& fileName)
{
	try
	{
		fileName = getFileAt(timestamp, direction);
		bool tsInFileFlag = isTimeStampInFile(fileName, timestamp);

		if (tsInFileFlag)
		{
			return true;
		}
		else
		{
			/*if the file is the last file in cache -
			 and there is a possibility that file may get updated (eg. while writing) -
			 then reload the file */
			if (direction && fileName == videoCache.rbegin()->path) //boost::filesystem::equivalent(boost::filesystem::canonical(fileName), boost::filesystem::canonical(videoCache.rbegin()->path))
			{
				uint64_t tstart_ts, tend_ts;
				readVideoStartEnd(fileName, tstart_ts, tend_ts);
				updateCache(fileName, tstart_ts, tend_ts);
				if (timestamp >= tstart_ts && timestamp <= tend_ts)
				{
					return true;
				}
			}
			fileName = getNextFileAfter(fileName, direction);
			return true;
		}
	}
	catch (Mp4_Exception& exception)
	{
		if (exception.getCode() == MP4_OCOF_EMPTY)
		{
			fileName = "";
			auto msg = "Unexpected error happened in OrderedCacheOfFiles while getting file from it";
			LOG_ERROR << msg;
			throw Mp4Exception(MP4_OCOF_EMPTY, msg);
		}
		if (exception.getCode() == MP4_OCOF_END)
		{
			fileName = "";
		}
		return false;
	}
}

void OrderedCacheOfFiles::insertInVideoCache(Video vid)
{
	boost::mutex::scoped_lock lock(m_mutex);
	videoCache.insert(vid);
}


/* directory parsing:
	1. Iterate over whole root directory recursively
	2. Discard files in the root folder
	3. Dont parse the folders with other format than yyyymmdd
	4. For all the mp4 files check if they are relevant based on start_ts + direction (strictly increasing/dec)
	5. For the first batchSize number of relevant files - add them in the videoCache with the starting timestamp (name)
	6. If no relevant file found in the parsing - return false to indicate no more relevant file on disk.
	7. If the higher watermark is breached in the videoCache, trigger a cleanup in seperate thread ?
	*/

std::vector<boost::filesystem::path> OrderedCacheOfFiles::parseAndSortDateDir(const std::string& rootDir)
{
	std::vector<boost::filesystem::path> dateDir;
	fs::directory_iterator dateDirIter(rootDir), dateDirEndIter;
	for (dateDirIter; dateDirIter != dateDirEndIter; ++dateDirIter)
	{
		if (fs::is_directory(dateDirIter->path()))
		{
			auto parentPath = dateDirIter->path().parent_path();

			// potential date folder
			if (fs::equivalent(parentPath, rootDir))
			{
				if (datePatternCheck(dateDirIter->path()))
				{
					dateDir.push_back(dateDirIter->path());
				}
			}
		}

	}

	if (dateDir.size())
	{
		std::sort(dateDir.begin(), dateDir.end());
	}
	return dateDir;
}

std::vector<boost::filesystem::path> OrderedCacheOfFiles::parseAndSortHourDir(const std::string& dateDirPath)
{
	std::vector<boost::filesystem::path> hourDir;

	fs::directory_iterator hourDirIter(dateDirPath), hourDirEndIter;
	for (hourDirIter; hourDirIter != hourDirEndIter; ++hourDirIter)
	{
		if (fs::is_directory(hourDirIter->path()))
		{
			// potential hour folder
			if (hourPatternCheck(hourDirIter->path()))
			{
				hourDir.push_back(hourDirIter->path());
			}
		}
	}

	if (hourDir.size())
	{
		std::sort(hourDir.begin(), hourDir.end());
	}
	return hourDir;
}

std::vector<boost::filesystem::path> OrderedCacheOfFiles::parseAndSortMp4Files(const std::string& hourDirPath)
{
	std::vector<boost::filesystem::path> mp4Files;
	fs::directory_iterator mp4FileIter(hourDirPath), mp4FileEndIter;
	// potential video file
	for (mp4FileIter; mp4FileIter != mp4FileEndIter; ++mp4FileIter)
	{
		if (filePatternCheck(mp4FileIter->path()))
		{
			mp4Files.push_back(mp4FileIter->path());
		}
	}
	if (mp4Files.size())
	{
		std::sort(mp4Files.begin(), mp4Files.end());
	}
	return mp4Files;
}

bool OrderedCacheOfFiles::parseFiles(uint64_t start_ts, bool direction, bool includeFloorFile, bool disableBatchSizeCheck, uint64_t skipTS)
{
	// Note- direction: synced with playback direction
	int parsedFilesCount = 0;
	lastKnownPlaybackDir = direction;
	bool exactMatchFound = false;
	uint64_t startTSofRelevantFile = 0;
	uint64_t startTSofPrevFileOnDisk = 0;
	boost::filesystem::path previousFileOnDisk = "";
	boost::filesystem::path exactMatchFile = "";

	auto dateDir = parseAndSortDateDir(rootDir);

	for (auto& dateDirPath : dateDir)
	{
		auto hourDir = parseAndSortHourDir(dateDirPath.string());

		for (auto& hourDirPath : hourDir)
		{
			if (!disableBatchSizeCheck && parsedFilesCount >= batchSize)
			{
				break; // stop parsing
			}

			auto mp4Files = parseAndSortMp4Files(hourDirPath.string());

			for (auto& mp4File : mp4Files)
			{
				// time based filtering
				// force batchSize check at hour folder level

				uint64_t fileTS = 0;
				try
				{
					fileTS = std::stoull(mp4File.stem().string());
				}
				catch (...)
				{
					LOG_TRACE << "OrderedCacheOfFiles: Ignoring File <" << mp4File.string() << "> due to timestamp parsing failure.";
					continue;
				}
				if (direction && fileTS < start_ts)
				{
					// keep track of prev mp4 file on disk
					previousFileOnDisk = mp4File.string();
					startTSofPrevFileOnDisk = fileTS;
					continue;
				}
				else if (!direction && fileTS > start_ts)
				{
					continue;
				}
				else if (!includeFloorFile && fileTS == start_ts)
				{
					exactMatchFound = true;
					exactMatchFile = mp4File.string();
					continue;
				}

				// cache insertion
				// LOG_INFO << "cache insert: " << mp4File << "\n";
				Video vid(mp4File.string(), fileTS);

				/* ----- first relevant file found ----- */
				if (!startTSofRelevantFile)
				{
					startTSofRelevantFile = vid.start_ts;

					if (includeFloorFile && !exactMatchFound)
					{
						// add prev file to cache - handles start_ts lies in middle of prevFileOnDisk in fwd parse.
						if (!previousFileOnDisk.empty())
						{
							startTSofRelevantFile = startTSofPrevFileOnDisk;
							Video prevVid(previousFileOnDisk.string(), startTSofPrevFileOnDisk);
							insertInVideoCache(prevVid);
							++parsedFilesCount;
						}
					}
				}
				/* ----- first relevant file found end ----- */

				insertInVideoCache(vid);
				++parsedFilesCount;
			}
		}
	}
	/* corner case: first relevant file was never found --- */
	if (!startTSofRelevantFile) // no file with ts > start_ts was found
	{
		if (exactMatchFound) // only 1 file with exactmatch was found
		{
			startTSofRelevantFile = start_ts;
			Video exactMatchVid(exactMatchFile.string(), start_ts);
			insertInVideoCache(exactMatchVid);
			++parsedFilesCount;
		}
		else if (includeFloorFile && startTSofPrevFileOnDisk) // in case start_ts is present in prevFileOnDisk
		{
			startTSofRelevantFile = startTSofPrevFileOnDisk;
			Video prevVid(previousFileOnDisk.string(), startTSofPrevFileOnDisk);
			insertInVideoCache(prevVid);
			++parsedFilesCount;
		}
	}
	/* trigger the drop strategy
	if seek triggered parse - drop from farthest side of seekTS */
	auto startDropFromTS = skipTS ? skipTS : startTSofRelevantFile;
	retireOldFiles(startDropFromTS);

	bool foundRelevantFilesOnDisk = parsedFilesCount > 0;
	return foundRelevantFilesOnDisk;
	
}

void OrderedCacheOfFiles::retireOldFiles(uint64_t ts)
{
	/* cant blindly delete from the cache end which is opposite to direction --
	eg. seek queryBeforeCache will drop new files as well */
	if (cleanCacheOnMainThread)
	{
		dropFarthestFromTS(ts);
		return;
	}
	if (mThread)
	{
		mThread->join();
	}
	mThread = boost::shared_ptr<boost::thread>(new boost::thread(boost::bind(&OrderedCacheOfFiles::dropFarthestFromTS, this, ts)));
}

void OrderedCacheOfFiles::dropFarthestFromTS(uint64_t ts)
{
	if (videoCache.empty())
	{
		return;
	}

	/* dropping algo */
	int64_t begDistTS = ts - videoCache.begin()->start_ts;
	auto absBeginDistance = abs(begDistTS);
	int64_t endDistTS = ts - videoCache.rbegin()->start_ts;
	auto absEndDistance = abs(endDistTS);
	if (videoCache.size() >= upperWaterMark)
	{
		if (absEndDistance <= absBeginDistance)
		{
			auto itr = videoCache.begin();
			while (itr != videoCache.end())
			{
				auto path = itr->path;
				if (videoCache.size() >= lowerWaterMark)
				{
					boost::mutex::scoped_lock(m_mutex);
					// Note - erase returns the iterator of next element after deletion.
					itr = videoCache.erase(itr);
				}
				else
				{
					return;
				}
			}
		}
		else
		{
			// delete from end using the fwd iterator.
			auto itr = videoCache.end();
			--itr;
			while (itr != videoCache.begin())
			{
				auto path = itr->path;
				if (videoCache.size() >= lowerWaterMark)
				{
					boost::mutex::scoped_lock(m_mutex);
					// Note - erase returns the iterator of next element after deletion.
					itr = videoCache.erase(itr);
					--itr;
				}
				else
				{
					return;
				}
			}
		}
	}
}

void OrderedCacheOfFiles::deleteLostEntry(std::string& filePath)
{
	auto itr = videoCache.find(filePath);
	if (itr == videoCache.end())
	{
		return;
	}

	boost::mutex::scoped_lock(m_mutex);
	itr = videoCache.erase(itr); // erase gives updated itr from cache

	return;
}

void OrderedCacheOfFiles::clearCache()
{
	if (videoCache.size())
	{
		boost::mutex::scoped_lock(m_mutex);
		videoCache.clear();
	}
}

bool OrderedCacheOfFiles::refreshCache()
{
	auto direction = lastKnownPlaybackDir;
	auto startParsingFrom = direction ? videoCache.begin()->start_ts : videoCache.rbegin()->end_ts;
	return parseFiles(startParsingFrom, direction, true, true);
}

/* Utils methods */
bool OrderedCacheOfFiles::filePatternCheck(const fs::path& path)
{
	if (fs::is_regular_file(path) && fs::extension(path) == ".mp4" &&
		path.stem().string().find_first_not_of("0123456789") == std::string::npos)
	{
		return true;
	}
	return false;
}

bool OrderedCacheOfFiles::datePatternCheck(const boost::filesystem::path& path)
{
	/*auto parentPath = path.parent_path();
	if (!boost::filesystem::equivalent(parentPath, rootDir))
	{
		return false;
	}*/
	auto pathStr = path.filename().string();
	return (pathStr.find_first_not_of("0123456789") == std::string::npos && pathStr.size() == 8);
}

bool OrderedCacheOfFiles::hourPatternCheck(const boost::filesystem::path& path)
{
	auto parentPath = path.parent_path();
	if (!datePatternCheck(parentPath))
	{
		return false;
	}
	auto pathStr = path.filename().string();
	return (pathStr.find_first_not_of("0123456789") == std::string::npos && pathStr.size() == 4);
}
