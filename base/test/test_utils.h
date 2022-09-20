#pragma once

#include <string>
#include <vector>
class Test_Utils
{
public:
	static void createDirIfNotExist(std::string path);
	static bool saveOrCompare(const char* fileName, const unsigned char* dataToSC, size_t sizeToSC, int tolerance);
	static bool saveOrCompare(std::string fileName, int tolerance);
	static bool saveOrCompare(const char* fileName, int tolerance);
	static bool readFile(std::string fileNameToUse, const uint8_t*& data, unsigned int& size);
	static std::string getArgValue(std::string argName, std::string argDefaultValue="");
	static void deleteFolder(std::string folderPath);
	struct FileCleaner {
		FileCleaner(std::vector<std::string> paths);
		~FileCleaner();
		std::vector<std::string> pathsOfFiles;
	};
	static void sleep_for_seconds(unsigned short seconds);
};