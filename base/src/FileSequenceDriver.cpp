#include "stdafx.h"
#include "FileSequenceDriver.h"
#include "FilenameStrategy.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include "Logger.h"
#include "BufferMaker.h"

FileSequenceDriver::FileSequenceDriver(const std::string& strPath,
	int startIndex,
	int maxIndex,
	bool readLoop,
	const std::vector<std::string>& files
): mAppend(false)
{
    //maxIndex should be greater than StartIndex
    assert(maxIndex < 0 || startIndex < maxIndex);
	mStrategy = FilenameStrategy::getStrategy(strPath, startIndex, maxIndex, readLoop, files, mAppend);
}

FileSequenceDriver::FileSequenceDriver(const std::string& strPath,
	bool append
): mAppend(append)
{
	// use this to append to 1 single file - FileWriterModule
	auto files = std::vector<std::string>();
	mStrategy = FilenameStrategy::getStrategy(strPath, 0, -1, false, files, mAppend);
}

FileSequenceDriver::~FileSequenceDriver()
{

}

bool FileSequenceDriver::Connect()
{
	auto ret = mStrategy->Connect();
	if (ret && mAppend)
	{
		// assuming write mode
		
		uint64_t index = 0;
		const std::string fileNameToUse = mStrategy->GetFileNameToUse(false, index);
		LOG_TRACE << "FileSequenceDriver::Writing Empty File " << fileNameToUse;

		auto mode = std::ios::out | std::ios::binary;
		if (mAppend)
		{
			mode = mode | std::ios::app;
		}
		std::ofstream file(fileNameToUse.c_str(), mode);
		if (file.is_open())
		{
			file.close();
		}
	}
	
	return ret;
}

bool FileSequenceDriver::Disconnect()
{
	return mStrategy->Disconnect();
}

bool FileSequenceDriver::IsConnected() const
{
    return mStrategy->IsConnected();
}

void FileSequenceDriver::notifyPlay(bool play)
{
	mStrategy->play(play);
}

void FileSequenceDriver::jump(uint64_t index)
{
	mStrategy->jump(index);
}

bool FileSequenceDriver::ReadP(BufferMaker& buffMaker, uint64_t& index)
{
    bool readRes = false;
	
    const std::string fileNameToUse = mStrategy->GetFileNameToUse(true, index);

    if (!fileNameToUse.empty())
    {
		LOG_TRACE << "FileSequenceDriver::Reading File " << fileNameToUse;

        std::ifstream file(fileNameToUse.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
        if (file.is_open())
        {
            //ios::ate implies that file has been seeked to the end
            //so tellg should give total size
			size_t req_size = static_cast<size_t>(file.tellg());
			if (req_size > 0U)
			{
				auto dataToRead = buffMaker.make(req_size);
				file.seekg(0, std::ios::beg);
				file.read((char*)dataToRead, req_size);

				LOG_TRACE << "FileSequenceDriver::Read " << req_size << " Bytes ";

				readRes = true;

			}
			else 
			{
				LOG_ERROR << "FileSequenceDriver::Read can not read file " << fileNameToUse;
			}
            
            file.close();
        }
    }

    return readRes;
}

// Reads the file and makes the buffer when metadata is provided
bool FileSequenceDriver::ReadP(BufferMaker& buffMaker, uint64_t& index, size_t &userMetadataSize)
{
    bool readRes = false;
	
    const std::string fileNameToUse = mStrategy->GetFileNameToUse(true, index);

    if (!fileNameToUse.empty())
    {
		LOG_TRACE << "FileSequenceDriver::Reading File " << fileNameToUse;

        std::ifstream file(fileNameToUse.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
        if (file.is_open())
        {
            //ios::ate implies that file has been seeked to the end
            //so tellg should give total size
			size_t fileSize = static_cast<size_t>(file.tellg());
			if (fileSize > 0U)
			{
				// Determine buffer size
                size_t bufferSize = fileSize;
				if (userMetadataSize > fileSize) // check if userMetadataSize is larger than file size
                {
                    bufferSize = userMetadataSize; // Resize frame to userMetadataSize
                }
                else
                {
                    LOG_WARNING << "File size is larger than user metadata size. Reading the entire file.";
                }

				auto dataToRead = buffMaker.make(bufferSize);
				file.seekg(0, std::ios::beg);
				file.read((char*)dataToRead, fileSize);

				LOG_TRACE << "FileSequenceDriver::Read " << fileSize << " Bytes ";

				readRes = true;
			}
			else 
			{
				LOG_ERROR << "FileSequenceDriver::Read can not read file " << fileNameToUse;
			}
            
            file.close();
        }
    }

    return readRes;
}

bool FileSequenceDriver::Read(uint8_t*& dataToRead, size_t& dataSize, uint64_t& index)
{
	bool readRes = false;

	const std::string fileNameToUse = mStrategy->GetFileNameToUse(true, index);

	if (!fileNameToUse.empty())
	{
		LOG_TRACE << "FileSequenceDriver::Reading File " << fileNameToUse;

		std::ifstream file(fileNameToUse.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
		if (file.is_open())
		{
			//ios::ate implies that file has been seeked to the end
			//so tellg should give total size
			dataSize = static_cast<size_t>(file.tellg());
			if (dataSize > 0U)
			{
				dataToRead = new uint8_t[dataSize];
				file.seekg(0, std::ios::beg);
				file.read((char*)dataToRead, dataSize);

				LOG_TRACE << "FileSequenceDriver::Read " << dataSize << " Bytes ";

				readRes = true;
			}

			file.close();
		}
	}

	return readRes;
}


bool FileSequenceDriver::Write(const uint8_t* dataToWrite, size_t dataSize)
{
	uint64_t index = 0;
	const std::string fileNameToUse = mStrategy->GetFileNameToUse(false, index);

	LOG_TRACE << "FileSequenceDriver::Writing File " << fileNameToUse;
  
	writeHelper(fileNameToUse, dataToWrite, dataSize, mAppend);
	return true;
}

bool FileSequenceDriver::writeHelper(const std::string &fileName, const uint8_t *dataToWrite, size_t dataSize, bool append)
{
	if (fileName.empty())
	{
		return false;
	}
	
	auto mode = std::ios::out | std::ios::binary;
	if (append)
	{
		mode = mode | std::ios::app;
	}

	auto res = true;

	std::ofstream file(fileName.c_str(), mode);
	if (file.is_open())
	{
		file.write((const char *)dataToWrite, dataSize);

		res = !file.bad() && !file.eof() && !file.fail();

		file.close();
	}
	else
	{
		res = false;
	}

	return res;
}

void FileSequenceDriver::SetReadLoop(bool readLoop)
{
	return mStrategy->SetReadLoop(readLoop);
}


bool FileSequenceDriver::canCache()
{
	return mStrategy->canCache();
}