#include "Mp4ReaderSourceUtils.h"
#include "Logger.h"

void FileStructureParser::boostPathToStrUtil(std::vector<boost::filesystem::path>& boost_paths, std::vector<std::string>& paths)
{
	for (auto i = 0; i < boost_paths.size(); ++i)
	{
		paths.push_back(boost_paths[i].string());
	}
}

bool FileStructureParser::setParseLimit(uint32_t _nParseFiles)
{
	/* Note - this limit is checked after min one yyyymmdd folder is parsed */
	nParseFiles = _nParseFiles;
	return true;
}

std::string FileStructureParser::format_2(int& num)
{
	if (num < 10)
	{
		return "0" + std::to_string(num);
	}
	else
	{
		return std::to_string(num);
	}
}

std::string FileStructureParser::format_hrs(int& hr)
{
	if (hr < 10)
	{
		return "000" + std::to_string(hr);
	}
	else
	{
		return "00" + std::to_string(hr);
	}
}

FileStructureParser::FileStructureParser()
{
}

void FileStructureParser::parseFilesInDirectory(boost::filesystem::path& folderPath, std::vector<boost::filesystem::path>& files)
{
	for (auto&& itr : boost::filesystem::directory_iterator(folderPath))
	{
		auto dirPath = itr.path();
		if (boost::filesystem::is_regular_file(dirPath) && boost::filesystem::extension(dirPath) == ".mp4")
		{
			files.push_back(dirPath);
		}
	}
}

void FileStructureParser::parseFilesInDirectories(std::vector<boost::filesystem::path>& dirs, std::vector<boost::filesystem::path>& files)
{
	/* updates the files with the files found in dirs - non-recursive */
	for (auto itr = dirs.begin(); itr != dirs.end(); ++itr)
	{
		parseFilesInDirectory(*itr, files);
	}

}

void FileStructureParser::parseDirectoriesInDirectory(boost::filesystem::path& folderPath, std::vector<boost::filesystem::path>& hourDirectories)
{
	for (auto&& itr : boost::filesystem::directory_iterator(folderPath))
	{
		auto dirPath = itr.path();
		if (boost::filesystem::is_directory(dirPath))
		{
			hourDirectories.push_back(dirPath);
		}
	}
}

void FileStructureParser::filterRelevantPaths(std::vector<boost::filesystem::path>& allPaths, boost::filesystem::path& startingPath, std::vector<boost::filesystem::path>& revelantPaths)
{
	/*
	assumes allPaths is sorted
	sets relevantPaths = paths after the startingPath in allPaths
	*/
	bool relevant = false;
	for (int i = 0; i < allPaths.size(); ++i)
	{
		if (allPaths[i] == startingPath)
		{
			relevant = true;
			continue;
		}
		if (relevant)
		{
			revelantPaths.push_back(allPaths[i]);
		}
	}
}

void FileStructureParser::parseDayDirectories(std::vector<boost::filesystem::path>& relevantDayDirectories, std::vector<boost::filesystem::path>& relevantVideoFiles)
{
	/*
		Parses the all day/hour dirs for given arg - 2 level deep only
		Parses files hour by hour till the nParseFiles limit is reached.
	*/
	for (auto i = 0; i < relevantDayDirectories.size(); ++i)
	{
		for (auto&& hrDiritr : boost::filesystem::directory_iterator(relevantDayDirectories[i]))
		{
			auto hrDirPath = hrDiritr.path();
			if (boost::filesystem::is_directory(hrDirPath)) // hr directory
			{
				parseFilesInDirectory(hrDirPath, relevantVideoFiles);
				if (relevantVideoFiles.size() >= 24 * 60)
				{
					// save the last parsed file
					latestParsedVideoPath = relevantVideoFiles[relevantVideoFiles.size() - 1];
					return;
				}
			}
		}
	}
}

bool FileStructureParser::parse(std::string& startingVideoFile, std::vector<std::string>& parsedVideoFiles, bool includeStarting)
{
	auto startingVideoPath = boost::filesystem::path(startingVideoFile);
	auto startingHourDirPath = startingVideoPath.parent_path();
	auto startingDayDirPath = startingHourDirPath.parent_path();
	auto rootDirPath = startingDayDirPath.parent_path();

	// return if file does not exist
	if (!boost::filesystem::exists(startingVideoPath))
	{
		LOG_ERROR << "file does not exist";
		return false;
	}

	// parse the relevant mp4 files from dir of the startingVideoPath file
	std::vector<boost::filesystem::path> videoFiles;
	std::vector<boost::filesystem::path> relevantVideoFiles;
	if (includeStarting)
	{
		relevantVideoFiles.push_back(startingVideoPath);
	}
	parseFilesInDirectory(startingHourDirPath, videoFiles);
	filterRelevantPaths(videoFiles, startingVideoPath, relevantVideoFiles);

	// parse the relevant hour directory names of this day
	std::vector<boost::filesystem::path> hourDirectories;
	std::vector<boost::filesystem::path> relevantHourDirectories;
	parseDirectoriesInDirectory(startingDayDirPath, hourDirectories);
	filterRelevantPaths(hourDirectories, startingHourDirPath, relevantHourDirectories);
	parseFilesInDirectories(relevantHourDirectories, relevantVideoFiles);

	if (relevantVideoFiles.size() >= nParseFiles)
	{
		LOG_DEBUG << "number of video files <" << relevantVideoFiles.size() << ">";
		boostPathToStrUtil(relevantVideoFiles, parsedVideoFiles);
		return true;
	}

	// parse the relevant day directory names
	std::vector<boost::filesystem::path> dayDirectories;
	std::vector<boost::filesystem::path> relevantDayDirectories;
	parseDirectoriesInDirectory(rootDirPath, dayDirectories);
	filterRelevantPaths(dayDirectories, startingDayDirPath, relevantDayDirectories);

	// parse all the relevant day directories for files
	parseDayDirectories(relevantDayDirectories, relevantVideoFiles);

	if (relevantVideoFiles.empty())
	{
		LOG_ERROR << "no video file found";
	}

	LOG_DEBUG << "number of video files <" << relevantVideoFiles.size() << ">";
	boostPathToStrUtil(relevantVideoFiles, parsedVideoFiles);
	return true;
}

bool FileStructureParser::init(std::string& startingVideoFile, std::vector<std::string>& parsedVideoFiles, bool includeStarting, bool parseDirs)
{
	/*
		parse the dir stucture format of mp4WriterSink if parseDirs is set
		otherwise read the individual mp4 file
	*/
	if (parseDirs)
	{
		if (!latestParsedVideoPath.empty())
		{
			includeStarting = false;
		}
		parse(startingVideoFile, parsedVideoFiles, includeStarting);
	}
	else
	{
		parsedVideoFiles.push_back(startingVideoFile);
	}
	return true;
}

int FileStructureParser::firstOfNextDay(boost::filesystem::path& baseFolder, std::string& yyyymmdd, std::string& videoFile)
{
	std::vector<std::string> dates;
	for (auto&& itr : boost::filesystem::directory_iterator(baseFolder))
	{
		auto filePath = itr.path();
		if (boost::filesystem::is_directory(filePath))
		{
			dates.push_back(filePath.filename().string());
		}
	}
	std::sort(dates.begin(), dates.end()); // is this required ?
	int idx = greater(dates, yyyymmdd);
	// no further dates present
	if (idx < 0)
	{
		LOG_ERROR << "end of recordings <" << ParseStatus::END_OF_RECORDINGS << ">";
		return ParseStatus::END_OF_RECORDINGS;
	}
	auto yyyymmddDir = baseFolder / dates[idx];
	std::string hr = "0"; // hint: "00" > "0"
	int ret = firstOfNextHour(yyyymmddDir, hr, videoFile);
	if (ret < 0)
	{
		LOG_ERROR << "end of recordings <" << ParseStatus::END_OF_RECORDINGS << ">";
		return ParseStatus::END_OF_RECORDINGS; //should never come here if empty directories are not present
	}
	return ParseStatus::FOUND_NEXT;
}

int FileStructureParser::firstOfNextHour(boost::filesystem::path& yyyymmddDir, std::string& hr, std::string& videoFile)
{
	std::vector<std::string> hrs;
	for (auto&& itr : boost::filesystem::directory_iterator(yyyymmddDir))
	{
		auto filePath = itr.path();
		if (boost::filesystem::is_directory(filePath))
		{
			hrs.push_back(filePath.filename().string());
		}
	}
	std::sort(hrs.begin(), hrs.end());
	int idx = greater(hrs, hr);
	// no further hrs present
	if (idx < 0)
	{
		return idx;
	}
	auto hrDir = yyyymmddDir / hrs[idx];
	std::string min = "00.mp4";
	return findFileWithMinute(hrDir, min, videoFile);
}

int FileStructureParser::findFileWithMinute(boost::filesystem::path& hrDir, std::string& min, std::string& videoFile)
{
	// returns the possible file with the given minute.
	// whether the file actually has the minute depends on the video file.
	std::vector <std::string> mins;
	for (auto&& itr : boost::filesystem::directory_iterator(hrDir))
	{
		auto filePath = itr.path();
		if (boost::filesystem::is_regular_file(filePath) && boost::filesystem::extension(filePath) == ".mp4")
		{
			mins.push_back(filePath.filename().string());
		}
	}

	std::sort(mins.begin(), mins.end()); // is this required ?
	// find exact file (lesserOrEqual), if its not available, find the next available file (greaterOrEqual)
	int idx = lesserOrEqual(mins, min);
	if (idx < 0)
	{
		idx = greaterOrEqual(mins, min);
		if (idx < 0)
		{
			return ParseStatus::NOT_FOUND;
		}
	}
	videoFile = (hrDir / mins[idx]).string();
	// set the nextToVideoFile in case the current file is not long enough to skip
	if (mins[idx] <= min && nextToVideoFile.empty())
	{	// save nextToVideoFileInfo only once - only when exact match is found
		if (idx < mins.size() - 1)
		{
			nextToVideoFile = (hrDir / mins[idx + 1]).string();
			nextToVideoFileFlag = ParseStatus::FOUND_NEXT;
		}
		else
		{	// parse for next file in next folders
			auto hrDirStr = hrDir.filename().string();
			int ret = firstOfNextHour(hrDir.parent_path(), hrDirStr, nextToVideoFile);
			nextToVideoFileFlag = ret;
			if (ret < 0)
			{
				auto yyyymmddDir = hrDir.parent_path().filename().string();
				nextToVideoFileFlag = firstOfNextDay(hrDir.parent_path().parent_path(), yyyymmddDir, nextToVideoFile);
			}
		}
		return ParseStatus::FOUND;
	}
	return ParseStatus::FOUND_NEXT;
}

int FileStructureParser::randomSeek(uint64_t& skipTS, std::string& skipDir, std::string& videoFile, uint64_t& skipMsecsInFile)
{
	skipMsecsInFile = 0;
	nextToVideoFile = "";
	std::chrono::milliseconds duration(skipTS);
	std::chrono::seconds secondsInDuration = std::chrono::duration_cast<std::chrono::seconds>(duration);
	std::chrono::milliseconds msecsInDuration = std::chrono::duration_cast<std::chrono::milliseconds>(duration - secondsInDuration);

	std::chrono::time_point<std::chrono::system_clock> timePointInSeconds(secondsInDuration);
	std::time_t t = std::chrono::system_clock::to_time_t(timePointInSeconds);
	std::tm tm = *std::localtime(&t);
	uint16_t msecs = msecsInDuration.count();

	auto baseFolder = boost::filesystem::path(skipDir);
	std::string yyyymmdd = std::to_string(1900 + tm.tm_year) + format_2(tm.tm_mon) + format_2(tm.tm_mday);
	auto yyyymmddDir = baseFolder / yyyymmdd;
	if (!boost::filesystem::is_directory(yyyymmddDir))
	{
		return firstOfNextDay(baseFolder, yyyymmdd, videoFile);
	}

	boost::filesystem::path hrDir = baseFolder / yyyymmdd / format_hrs(tm.tm_hour);
	if (!boost::filesystem::is_directory(hrDir))
	{
		int retHr = firstOfNextHour(yyyymmddDir, format_hrs(tm.tm_hour), videoFile);
		if (retHr < 0)
		{
			return firstOfNextDay(baseFolder, yyyymmdd, videoFile); // if no next hour found on this day, move to next day
		}
		return retHr;
	}

	std::tm tm2 = tm;
	tm2.tm_sec = 0;
	std::time_t tNew = std::mktime(&tm2);
	auto tNewTimePoint = std::chrono::system_clock::from_time_t(tNew);
	std::chrono::system_clock::duration d = tNewTimePoint.time_since_epoch();
	uint64_t tsTillMinInMsecs = std::chrono::duration_cast<std::chrono::milliseconds>(d).count();

	std::string fileToSearch = std::to_string(tsTillMinInMsecs) + ".mp4";
	int ret = findFileWithMinute(hrDir, fileToSearch, videoFile);
	if (ret < 0)
	{
		auto retHr = firstOfNextHour(yyyymmddDir, format_hrs(tm.tm_hour), videoFile);
		if (retHr < 0)
		{
			return firstOfNextDay(baseFolder, yyyymmdd, videoFile);
		}
		return retHr;
	}

	// we need frame at this msec in the file videoFile
	std::string videoFileName = boost::filesystem::path(videoFile).filename().string();
	uint64_t tsOfFileFound = std::stoull(videoFileName.substr(0, videoFileName.find(".")));
	if (tsOfFileFound < skipTS)
	{
		skipMsecsInFile = skipTS - tsOfFileFound;
	}
	return ret;
}