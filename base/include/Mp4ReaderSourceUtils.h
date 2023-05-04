#pragma once

#include <string>
#include <vector>
#include <boost/filesystem.hpp>

class FileStructureParser
{
public:
	enum ParseStatus
	{
		NOT_FOUND = -3,
		END_OF_FILE,
		END_OF_RECORDINGS,
		FOUND,
		FOUND_NEXT
	};

	FileStructureParser();
	bool init(std::string& _seedVideoFilePath, std::vector<std::string>& parsedVideoFiles, bool includeStarting, bool parseDir = true);
	bool parse(std::string& startingVideoFile, std::vector<std::string>& parsedVideoFiles, bool includeStarting);
	int randomSeek(uint64_t& skipTS, std::string& skipDir, std::string& videoFile, uint64_t& skipMsecsInFile);
	bool setParseLimit(uint32_t _nParseFiles);
	bool parseDir(boost::filesystem::path dirPath, std::string& videoName);
	int getNextToVideoFileFlag()
	{
		return nextToVideoFileFlag;
	}
	std::string getNextVideoFile()
	{
		return nextToVideoFile;
	}
private:
	void parseFilesInDirectory(boost::filesystem::path& folderPath,
		std::vector<boost::filesystem::path>& files);
	void parseFilesInDirectories(std::vector<boost::filesystem::path>& dirs,
		std::vector<boost::filesystem::path>& files);
	void parseDirectoriesInDirectory(boost::filesystem::path& folderPath,
		std::vector<boost::filesystem::path>& hourDirectories);
	void filterRelevantPaths(std::vector<boost::filesystem::path>& allPaths,
		boost::filesystem::path& startingPath, std::vector<boost::filesystem::path>& revelantPaths);
	void parseDayDirectories(std::vector<boost::filesystem::path>& relevantDayDirectories,
		std::vector<boost::filesystem::path>& relevantVideoFiles);
	void boostPathToStrUtil(std::vector<boost::filesystem::path>& boost_paths,
		std::vector<std::string>& paths);
	template<typename T>
	int greater(std::vector<T>& list, T& elem)
	{
		int st = 0;
		int listSize = list.size();
		int end = listSize - 1;
		while (st < end)
		{
			int mid = (st + end) / 2;
			if (list[mid] <= elem)
			{
				st = mid + 1;
			}
			else
			{
				end = mid;
			}
		}
		if (st == listSize - 1 && list[st] <= elem)
		{
			return -1;
		}
		return st;
	}

	template<typename T>
	int greaterOrEqual(std::vector<T>& list, T& elem)
	{
		int st = 0;
		int listSize = list.size();
		int end = listSize - 1;
		while (st < end)
		{
			int mid = (st + end) / 2;
			if (list[mid] < elem)
			{
				st = mid + 1;
			}
			else
			{
				end = mid;
			}
		}
		if (st == listSize - 1 && list[st] < elem)
		{
			return -1;
		}
		return st;
	}

	template<typename T>
	int lesserOrEqual(std::vector<T>& list, T& elem)
	{
		int st = 0;
		int listSize = list.size();
		int end = listSize - 1;
		if (list[end] < elem)
		{
			return end;
		}
		while (st <= end)
		{
			int mid = (st + end) / 2;
			if (list[mid] == elem)
			{
				return mid;
			}
			if (mid > 0 && list[mid - 1] <= elem && list[mid] > elem)
			{
				return mid - 1;
			}

			if (list[mid] > elem)
			{
				end = mid - 1;
			}
			else
			{
				st = mid + 1;
			}
		}
		if (!st && list[st] > elem)
		{
			return -1;
		}
		return st;
	}

	// random seek
	int firstOfNextDay(boost::filesystem::path& baseFolder, std::string& yyyymmdd, std::string& videoFile);
	int firstOfNextHour(boost::filesystem::path& yyyymmddDir, std::string& hr, std::string& videoFile);
	int findFileWithMinute(boost::filesystem::path& hrDir, std::string& min, std::string& videoFile);

	bool filePatternCheck(const boost::filesystem::path& path);
	bool datePatternCheck(const boost::filesystem::path& path);
	bool hourPatternCheck(const boost::filesystem::path& path);

	std::string format_2(int& num);
	std::string format_hrs(int& hr);

	/* this limit is checked after min one yymmdd folder parsed */
	uint32_t nParseFiles = 24 * 60;
	boost::filesystem::path latestParsedVideoPath;
	std::string nextToVideoFile = "";
	int nextToVideoFileFlag;
};