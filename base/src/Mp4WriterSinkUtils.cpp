#include <string>
#include <boost/filesystem.hpp>
#include <chrono>
#include "Logger.h"
#include "Mp4WriterSinkUtils.h"
#include "FrameMetadata.h"
#include "H264Utils.h"

Mp4WriterSinkUtils::Mp4WriterSinkUtils()
{
	lastVideoTS = 0;
	lastSyncTS = std::time(nullptr);
}

std::string Mp4WriterSinkUtils::format_hrs(int &hr)
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

std::string Mp4WriterSinkUtils::format_2(int &num)
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

std::string Mp4WriterSinkUtils::filePath(boost::filesystem::path relPath, std::string mp4FileName, std::string baseFolder, uint64_t chunkTimeInMinutes)
{
	boost::filesystem::path finalPath;
	std::string mp4VideoPath;
	if (customNamedFileDirCheck(baseFolder, chunkTimeInMinutes, relPath, mp4VideoPath))
	{
		return mp4VideoPath;
	}
	
	auto folderPath = boost::filesystem::path(baseFolder) / relPath;
	if (boost::filesystem::is_directory(folderPath))
	{
		finalPath = folderPath / mp4FileName;
		return finalPath.string();
	}

	if (boost::filesystem::create_directories(folderPath))
	{
		finalPath = folderPath / mp4FileName;
		return finalPath.string();
	}
	else
	{
		LOG_ERROR << "Failed to create the directory <" << folderPath << ">";
		LOG_ERROR << "Check the dir permissions.";
		return "";
	}
}

bool Mp4WriterSinkUtils::customNamedFileDirCheck(std::string baseFolder, uint32_t chunkTimeInMinutes, boost::filesystem::path relPath, std::string& nextFrameFileName)
{
	if (boost::filesystem::extension(baseFolder) == ".mp4")
	{
		if (chunkTimeInMinutes == UINT32_MAX)
		{
			auto folderPath = boost::filesystem::path(baseFolder) / relPath;
			auto path = folderPath.remove_filename();
			if (boost::filesystem::is_directory(path))
			{
				nextFrameFileName = baseFolder;
				return true;
			}

			if (boost::filesystem::create_directories(path))
			{
				nextFrameFileName = baseFolder;
				return true;
			}
			else
			{
				LOG_ERROR << "Failed to create the directory <" << folderPath << ">";
				LOG_ERROR << "Check the dir permissions.";
				return true;
			}
		}
		else
		{
			LOG_ERROR << "Custom video file name only supported while writing to a single file.";
			throw AIPException(AIP_FATAL, "Custom video file name only supported while writing to a single file.");
		}
	}
	else
	{
		return false;
	}
}

void Mp4WriterSinkUtils::parseTSJpeg(uint64_t &ts, uint32_t &chunkTimeInMinutes, uint32_t &syncTimeInSeconds,
	boost::filesystem::path &relPath, std::string &mp4FileName, bool &syncFlag, std::string baseFolder, std::string& nextFrameFileName)
{
	std::chrono::milliseconds duration(ts);
	std::chrono::seconds secondsInDuration = std::chrono::duration_cast<std::chrono::seconds>(duration);
	std::chrono::milliseconds msecsInDuration = std::chrono::duration_cast<std::chrono::milliseconds>(duration - secondsInDuration);

	std::chrono::time_point<std::chrono::system_clock> timePointInSeconds(secondsInDuration);
	std::time_t t = std::chrono::system_clock::to_time_t(timePointInSeconds);
	std::tm tm = *std::localtime(&t);
	uint16_t msecs = msecsInDuration.count();

	if ((t - lastSyncTS) >= syncTimeInSeconds)
	{
		syncFlag = true;
		lastSyncTS = t;
	}
	else
	{
		syncFlag = false;
	}

	// used cached values if the difference in ts is less than chunkTime
	uint32_t chunkTimeInSecs = 60 * chunkTimeInMinutes;
	if ((t - lastVideoTS) < chunkTimeInSecs && currentFolder == baseFolder)
	{
		relPath = lastVideoFolderPath;
		mp4FileName = lastVideoName;
		nextFrameFileName = filePath(relPath, mp4FileName, baseFolder, chunkTimeInMinutes);
		return;
	}

	// cache new values
	currentFolder = baseFolder;
	lastVideoTS = t;
	lastVideoMinute = tm.tm_min;

	if (customNamedFileDirCheck(baseFolder, chunkTimeInMinutes, relPath, nextFrameFileName))
		return;

	std::string yyyymmdd = std::to_string(1900 + tm.tm_year) + format_2(tm.tm_mon) + format_2(tm.tm_mday);
	relPath = boost::filesystem::path(yyyymmdd) / format_hrs(tm.tm_hour);
	mp4FileName = std::to_string(ts) + ".mp4";
	lastVideoFolderPath = relPath;

	lastVideoName = mp4FileName;
	
	nextFrameFileName = filePath(relPath, mp4FileName, baseFolder, chunkTimeInMinutes);
}
void Mp4WriterSinkUtils::parseTSH264(uint64_t& ts, uint32_t& chunkTimeInMinutes, uint32_t& syncTimeInSeconds,boost::filesystem::path& relPath, 
	std::string& mp4FileName, bool& syncFlag, short frameType, short naluType, std::string baseFolder, std::string& nextFrameFileName)
{
	std::chrono::milliseconds duration(ts);
	std::chrono::seconds secondsInDuration = std::chrono::duration_cast<std::chrono::seconds>(duration);
	std::chrono::milliseconds msecsInDuration = std::chrono::duration_cast<std::chrono::milliseconds>(duration - secondsInDuration);

	std::chrono::time_point<std::chrono::system_clock> timePointInSeconds(secondsInDuration);
	std::time_t t = std::chrono::system_clock::to_time_t(timePointInSeconds);
	std::tm tm = *std::localtime(&t);
	uint16_t msecs = msecsInDuration.count();

	if ((t - lastSyncTS) >= syncTimeInSeconds)
	{
		syncFlag = true;
		lastSyncTS = t;
	}
	else
	{
		syncFlag = false;
	}

	if (boost::filesystem::extension(baseFolder) == ".mp4")
	{
		if(currentFolder != baseFolder)
		{
			if(naluType == H264Utils::H264_NAL_TYPE::H264_NAL_TYPE_IDR_SLICE || naluType == H264Utils::H264_NAL_TYPE_SEQ_PARAM)
			{
				currentFolder = baseFolder;
			}
			else
			{
				return;
			}
		}
		if(currentFolder == baseFolder)
		{
			customNamedFileDirCheck(baseFolder, chunkTimeInMinutes, relPath, nextFrameFileName);
			return;
		}
	}	
	// used cached values if the difference in ts is less than chunkTime
	uint32_t chunkTimeInSecs = 60 * chunkTimeInMinutes;
	if ((t - lastVideoTS) < chunkTimeInSecs && currentFolder == baseFolder)// && chunkTimeInMinutes != UINT32_MAX
	{
		relPath = lastVideoFolderPath;
		mp4FileName = lastVideoName;
		nextFrameFileName = filePath(relPath, mp4FileName, baseFolder, chunkTimeInMinutes);
		return;
	}
	// cannot be merged with if condition above.
	if (naluType != H264Utils::H264_NAL_TYPE::H264_NAL_TYPE_IDR_SLICE && naluType != H264Utils::H264_NAL_TYPE_SEQ_PARAM)
	{
		relPath = lastVideoFolderPath;
		mp4FileName = lastVideoName;
		nextFrameFileName = tempNextFrameFileName;
		return;
	}
	// get new video path

	// cache new values
	currentFolder = baseFolder;
	lastVideoTS = t;
	lastVideoMinute = tm.tm_min;

	if (customNamedFileDirCheck(baseFolder, chunkTimeInMinutes, relPath, nextFrameFileName))
		return;

	std::string yyyymmdd = std::to_string(1900 + tm.tm_year) + format_2(tm.tm_mon) + format_2(tm.tm_mday);
	relPath = boost::filesystem::path(yyyymmdd) / format_hrs(tm.tm_hour);
	mp4FileName = std::to_string(ts) + ".mp4";
	lastVideoName = mp4FileName;
	lastVideoFolderPath = relPath;

	nextFrameFileName = filePath(relPath, mp4FileName, baseFolder, chunkTimeInMinutes);
	tempNextFrameFileName = nextFrameFileName;
}

void Mp4WriterSinkUtils::getFilenameForNextFrame(std::string& nextFrameFileName ,uint64_t& timestamp, std::string& basefolder,
	uint32_t chunkTimeInMinutes, uint32_t syncTimeInSeconds, bool& syncFlag, short& frameType , short naluType)
{
	boost::filesystem::path finalPath;
	std::string mp4FileName;
	boost::filesystem::path relPath;

	if (frameType == FrameMetadata::FrameType::H264_DATA)
	{
		parseTSH264(timestamp, chunkTimeInMinutes, syncTimeInSeconds, relPath, mp4FileName, syncFlag, frameType, naluType, basefolder, nextFrameFileName);
	}
	else if (frameType == FrameMetadata::FrameType::ENCODED_IMAGE)
	{
		parseTSJpeg(timestamp, chunkTimeInMinutes, syncTimeInSeconds, relPath, mp4FileName, syncFlag, basefolder, nextFrameFileName);
	}
	
}

Mp4WriterSinkUtils::~Mp4WriterSinkUtils()
{
}
