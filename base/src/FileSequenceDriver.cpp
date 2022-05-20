#include "stdafx.h"
#include "FileSequenceDriver.h"
#include "FilenameStrategy.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include "Logger.h"


FileSequenceDriver::FileSequenceDriver(const std::string& strPath,
	int startIndex,
	int maxIndex,
	bool readLoop,
	const std::vector<std::string>& files
): mAppend(false)
{
    //maxIndex should be greater than StartIndex
    assert(maxIndex < 0 || startIndex < maxIndex);
	mStrategy = FilenameStrategy::getStrategy(strPath, startIndex, maxIndex, readLoop, files);
}

FileSequenceDriver::FileSequenceDriver(const std::string& strPath,
	bool append
): mAppend(append)
{
	// use this to append to 1 single file - FileWriterModule
	auto files = std::vector<std::string>();
	mStrategy = FilenameStrategy::getStrategy(strPath, 0, -1, false, files);
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
		// LOG_ERROR << "File Name is " << fileNameToUse << "   " << __func__;

		auto mode = std::ios::out | std::ios::binary;
		std::ofstream file(fileNameToUse.c_str(), mode);
		if (file.is_open())
		{
			file.close();
			//std::remove(fileNameToUse.c_str());
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

//reads on the supplied buffer throws error if it is too small
bool FileSequenceDriver::ReadP(uint8_t* dataToRead, size_t& dataSize, uint64_t& index)
{
    bool readRes = false;

    const std::string fileNameToUse = mStrategy->GetFileNameToUse(true, index);
	
    if (!fileNameToUse.empty())
    {
		LOG_TRACE << "FileSequenceDriver::Reading File " << fileNameToUse;

        std::ifstream file(fileNameToUse.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
		
        if (file.is_open())
        {
			// LOG_ERROR << "File Name is " << fileNameToUse << "Data Size is " << dataSize << __func__;

            //ios::ate implies that file has been seeked to the end
            //so tellg should give total size
			size_t req_size = static_cast<size_t>(file.tellg());
			if (req_size > 0U)
			{
				if(dataSize >= req_size)
				{
					dataSize = req_size; //let them know how much did I read
					file.seekg(0, std::ios::beg);
					file.read((char*)dataToRead, dataSize);

					LOG_TRACE << "FileSequenceDriver::Read " << dataSize << " Bytes ";

					readRes = true;
				}
				else {
					LOG_WARNING << "FileSequenceDriver::Read requires " << req_size << " Bytes supplied " << dataSize;
					//let them know how much do I need
					dataSize = req_size;
				}
			}
			else {
				LOG_ERROR << "FileSequenceDriver::Read can not read file " << fileNameToUse;
				dataSize = 0;
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
	// LOG_ERROR << "File Name is " << fileNameToUse << "Data Size is " << dataSize << __func__;
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

bool FileSequenceDriver::WriteFirst(frame_sp frame)
{
	uint64_t index = 0;
	// cv::Mat img;
	// framemetadata_sp frameMeta = frame->getMetadata();
	// FrameMetadata::FrameType fType=frameMeta->getFrameType();
	// auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(frameMeta);
	// auto height = rawMetadata->getHeight();
	// auto width = rawMetadata->getWidth();
	// auto st = rawMetadata->getStep();
	// img =cv::Mat(800, 800, CV_8UC4, frame->data(), 800);
	const std::string fileNameToUse = mStrategy->GetFileNameToUse(false, index);
	// LOG_ERROR << "File Name to use is " << fileNameToUse;
	// cv::imwrite(fileNameToUse ,img);
}

bool FileSequenceDriver::Write1(frame_sp frame)
{
	// return true;
	// 	uint64_t index = 0;

	// const std::string fileNameToUse = mStrategy->GetFileNameToUse(false, index);
	// // frame_number++;
	// // char filename[15];
	// // sprintf(filename, "frame-%d.raw", frame_number);
	// FILE *fp = fopen(fileNameToUse, "wb");
	// printf("Frame write %s of size %d\n", fileNameToUse, size);
	// fwrite(p, size, 1, fp);
	// fflush(fp);
	// fclose(fp);

	uint64_t index = 0;
	cv::Mat img;
	framemetadata_sp frameMeta = frame->getMetadata();
	FrameMetadata::FrameType fType=frameMeta->getFrameType();
	// LOG_ERROR << "Frame Type is " << fType;
	// LOG_ERROR << "Frame Size is " << frame->size();
	auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(frameMeta);
	auto height = rawMetadata->getHeight();
	auto width = rawMetadata->getWidth();
	auto st = rawMetadata->getStep();
	// LOG_ERROR << "Width is"<< width;
	// LOG_ERROR << "Height id "<< height;
	img =cv::Mat(height, width, CV_8UC4, frame->data(), st); //// uncomment line here
	const std::string fileNameToUse = mStrategy->GetFileNameToUse(false, index);
	LOG_ERROR << "<====================== File Name to use is ===================================>" << fileNameToUse;
	cv::imwrite(fileNameToUse ,img);  //// uncomment line here 
	return true;

	// LOG_ERROR << "File Name is----------------------------------------------------------------------------------------------------------------------------------" << fileNameToUse;


	
	// LOG_ERROR << "Setting time Stamp to " << frame->timestamp;
	// struct timeval time_now{};
	// gettimeofday(&time_now, nullptr);
	// time_t msecs_time = (time_now.tv_sec * 1000) + (time_now.tv_usec / 1000);
	// LOG_ERROR << "Printing Time Stamp inside FILE SEQUENCE WRITER MODULE " << msecs_time;
	
}

bool FileSequenceDriver::getCurrentStatus()
{
	LOG_DEBUG << "Current Status Of Module "<< isPlaying;
	return isPlaying;
}

bool FileSequenceDriver::setNoOfFrame(int x)
{
	LOG_DEBUG << "Value Of X is" << x;
	noOFFramesToCapture = x;
	return true;
}

bool FileSequenceDriver::resetInfo()
{
	isPlaying = true;
	framesCaptured = 0;
	LOG_DEBUG << " FIle Sequence info reseted Value of ISplaying is " << isPlaying;
	return true;
}

bool FileSequenceDriver::Write(const uint8_t* dataToWrite, size_t dataSize)
{
	uint64_t index = 0;
	const std::string fileNameToUse = mStrategy->GetFileNameToUse(false, index);
	LOG_TRACE << "FileSequenceDriver::Writing File " << fileNameToUse;
	LOG_ERROR << "File Name to use is " << fileNameToUse;
	// LOG_ERROR << "Data Size is "<< dataSize;
	writeHelper(fileNameToUse, dataToWrite, dataSize, mAppend);
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
	// LOG_ERROR << "File Name " << fileName.c_str() << "Data Size is " << dataSize;

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