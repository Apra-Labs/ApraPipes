#pragma once

#include <string>
#include <boost/filesystem.hpp>
#include "Logger.h"
class Test_Utils
{
public:
	static void createDirIfNotExist(std::string path);
	static bool saveOrCompare(const char* fileName, const unsigned char* dataToSC, size_t sizeToSC, int tolerance);
	static bool saveOrCompare(const char* fileName, int tolerance);
	static bool readFile(std::string fileNameToUse, const uint8_t*& data, unsigned int& size);
	static std::string getArgValue(std::string argName, std::string argDefaultValue="");
	struct FileCleaner {
		FileCleaner(std::vector<std::string> paths) {
			pathsOfFiles = paths;
		};
		~FileCleaner() {
			for (int i = 0; i < pathsOfFiles.size(); i++) {
				boost::filesystem::path filePath(pathsOfFiles[i]);
				if (boost::filesystem::exists(filePath))
				{
					boost::filesystem::remove(filePath);
				}
			}
		};
		std::vector<std::string> pathsOfFiles;
	};
	static void sleep_for_seconds(unsigned short seconds)
	{
		LOG_INFO << " Sleeping for " << seconds << " seconds";
		boost::this_thread::sleep_for(boost::chrono::seconds(seconds));
		LOG_INFO << "Done sleeping for " << seconds << " seconds";
	}
};
