#include "Utils.h"
#include "test_utils.h"
#include "Logger.h"
#include <boost/test/unit_test.hpp>
#include <boost/test/framework.hpp>
#include "iostream"
#include <boost/filesystem.hpp>
#include <fstream>

#ifndef SAFE_DELETE_ARRAY
#define SAFE_DELETE_ARRAY(p) \
    delete[] p ; \
    p = nullptr;
#endif //SAFE_DELETE

bool Test_Utils::readFile(std::string fileNameToUse, const uint8_t*& data, unsigned int& size)
{
	BOOST_TEST_MESSAGE(fileNameToUse);
	bool readRes = false;
	if (!fileNameToUse.empty())
	{
		if (boost::filesystem::is_regular_file(fileNameToUse.c_str()))
		{
			std::ifstream file(fileNameToUse.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
			if (file.is_open())
			{
				//ios::ate implies that file has been seeked to the end
				//so tellg should give total size
				size = static_cast<unsigned int>(file.tellg());
				if (size > 0U)
				{
					data = new uint8_t[size];
					file.seekg(0, std::ios::beg);
					file.read((char*)data, size);


					readRes = true;
				}

				file.close();
			}
		}
	}
	return readRes;
}

bool writeFile(std::string fileNameToUse, const uint8_t* data, size_t size)
{
	bool writeRes = false;

	if (!fileNameToUse.empty())
	{
		std::ofstream file(fileNameToUse.c_str(), std::ios::out | std::ios::binary);
		if (file.is_open())
		{
			file.write((const char*)data, size);

			writeRes = true;

			file.close();
		}
	}
	return writeRes;
}

bool CompareData(const uint8_t* data01, const uint8_t* data02, unsigned int dataSize, int tolerance)
{
	int mismatch = 0;

	for (unsigned int i = 0; i < dataSize; i++)
	{
		if (data01[i] == data02[i])
		{
			continue;

		}
		else
		{
			mismatch += 1;
			LOG_ERROR<<"The mismatch occured at data element"<<i;

			if (mismatch > tolerance)
			{
				LOG_ERROR<<"Mismatch has crossed tolerance. Mismatch "<<mismatch<<" > tolerance "<<tolerance;
				break;
			}
		}
	}

	return (mismatch <= tolerance);
}

void Test_Utils::createDirIfNotExist(std::string path)
{
	// it always goes 1 back and creates that directory if not exists
	// example - /a/b/c/d - then it recursively creates /a/b/c if any doesnt exist - doesnt care about d
	// example - /a/b/c/d.abc - recursively creates /a/b/c if any doesnt exist

	boost::filesystem::path p(path);
	boost::filesystem::path dirPath = p.parent_path();

	if (!boost::filesystem::exists(dirPath))
	{
		boost::filesystem::create_directories(dirPath);
	}
}

bool Test_Utils::saveOrCompare(const char* fileName, const unsigned char* dataToSC, size_t sizeToSC, int tolerance)
{
	Test_Utils::createDirIfNotExist(fileName);

	bool compareRes = true;

	if (boost::filesystem::is_regular_file(fileName))
	{
		const std::string strFullFileName = fileName;
		std::string strFileBaseName = strFullFileName;
		std::string strExtension;
		std::string nameOfFile;
		auto test_name = std::string(boost::unit_test::framework::current_test_case().p_name);
		const size_t fileBaseIndex = strFullFileName.find_last_of(".");
		const size_t fileBaseLocation = strFullFileName.find_last_of("/");
		if (std::string::npos != fileBaseIndex)
		{
			strFileBaseName = strFullFileName.substr(0U, fileBaseIndex);
			strExtension = strFullFileName.substr(fileBaseIndex);
		}
		if (std::string::npos != fileBaseLocation)
		{
			 nameOfFile = strFullFileName.substr(fileBaseLocation + 1, fileBaseIndex - fileBaseLocation -1);
		}

		const uint8_t* dataRead = nullptr;
		unsigned int dataSize = 0U;
		BOOST_TEST(readFile(fileName, dataRead, dataSize));
		BOOST_TEST(dataSize == sizeToSC);
		compareRes = CompareData(dataToSC, dataRead, dataSize, tolerance);
		if(!compareRes)
		{
			boost::filesystem::create_directory("./data/SaveOrCompareFail");
			std::string saveFile = "./data/SaveOrCompareFail/" + test_name + nameOfFile + strExtension;
			writeFile(saveFile.c_str(), dataToSC, sizeToSC);
		}
		BOOST_TEST(compareRes);

		SAFE_DELETE_ARRAY(dataRead);
	}
	else
	{
		BOOST_TEST(writeFile(fileName, dataToSC, sizeToSC));
	}

	return compareRes;
}
bool Test_Utils::saveOrCompare(std::string fileName, int tolerance)
{
	return saveOrCompare(fileName.c_str(), tolerance);
}
bool Test_Utils::saveOrCompare(const char* fileName, int tolerance)
{
	bool res = false;

	const std::string strFullFileName = fileName;
	std::string strFileBaseName = strFullFileName;
	std::string strExtension;

	const size_t fileBaseIndex = strFullFileName.find_last_of(".");
	if (std::string::npos != fileBaseIndex)
	{
		strFileBaseName = strFullFileName.substr(0U, fileBaseIndex);
		strExtension = strFullFileName.substr(fileBaseIndex);
	}

	strFileBaseName += "_Compare";

	const std::string strCompareFileName = strFileBaseName + strExtension;

	const uint8_t* dataRead = nullptr;
	unsigned int dataSize = 0U;
	if (readFile(strCompareFileName, dataRead, dataSize))
	{
		res = saveOrCompare(fileName, dataRead, dataSize, tolerance);
	}
	else
	{
		//No comparison file has been created yet..Will create one now
		SAFE_DELETE_ARRAY(dataRead);
		BOOST_TEST(readFile(fileName, dataRead, dataSize));
		BOOST_TEST(writeFile(strCompareFileName.c_str(), dataRead, dataSize));

		res = true;
	}

	SAFE_DELETE_ARRAY(dataRead);

	return res;
}

std::string Test_Utils::getArgValue(std::string argName, std::string argDefaultValue)
{
	argName = "-" + argName;
	for (int i = 1; i < boost::unit_test::framework::master_test_suite().argc; i++)
	{
		LOG_DEBUG << "Arg[" << i << "] is ------------ " << boost::unit_test::framework::master_test_suite().argv[i];
	}

	for (int i = 1; i < boost::unit_test::framework::master_test_suite().argc; i++)
	{
		if (boost::unit_test::framework::master_test_suite().argv[i] == argName)
		{
			return boost::unit_test::framework::master_test_suite().argv[i + 1];
		}
	}

	return argDefaultValue;
}
void Test_Utils::deleteFolder(std::string folderPath)
{
	boost::filesystem::remove_all(folderPath);
}

void Test_Utils::sleep_for_seconds(unsigned short seconds)
{
	if (seconds <= 0) return;
	LOG_INFO << " Sleeping for " << seconds << " seconds";
	boost::this_thread::sleep_for(boost::chrono::seconds(seconds));
	LOG_INFO << "Done sleeping for " << seconds << " seconds";
}

void Test_Utils::sleep_for_milliseconds(unsigned short milliseconds)
{
	if (milliseconds <= 0) return;
	LOG_TRACE << " Sleeping for " << milliseconds << " milliseconds";
	boost::this_thread::sleep_for(boost::chrono::milliseconds(milliseconds));
	LOG_TRACE << "Done sleeping for " << milliseconds << " milliseconds";
}

Test_Utils::FileCleaner::FileCleaner(std::vector<std::string> paths) {
	pathsOfFiles = paths;
};
Test_Utils::FileCleaner::~FileCleaner() {
	for (int i = 0; i < pathsOfFiles.size(); i++) {
		boost::filesystem::path filePath(pathsOfFiles[i]);
		if (boost::filesystem::exists(filePath))
		{
			if (boost::filesystem::is_regular_file(filePath))
			{
				boost::filesystem::remove(filePath);
			}
			else
			{
				boost::filesystem::remove_all(filePath);
			}
		}
	}
};
