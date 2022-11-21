#include <string>
#include <boost/filesystem.hpp>

#include "Logger.h"
#include "Mp4WriterSinkUtils.h"
#include "FrameMetadata.h"
#include "H264Utils.h"

Mp4WriterSinkUtils::Mp4WriterSinkUtils()
{
	lastVideoTS = 0;
	lastSyncTS = std::time(nullptr);
}

std::string Mp4WriterSinkUtils::format_hrs(int& hr)
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

std::string Mp4WriterSinkUtils::format_2(int& num)
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

std::string Mp4WriterSinkUtils::filePath(boost::filesystem::path relPath, std::string mp4FileName, std::string baseFolder)
{
	boost::filesystem::path finalPath;
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

void Mp4WriterSinkUtils::parseTSJpeg(uint64_t& ts, uint32_t& chunkTimeInMinutes, uint32_t& syncTimeInSeconds,
	boost::filesystem::path& relPath, std::string& mp4FileName, bool& syncFlag, std::string baseFolder, std::string& nextFrameFileName, std::string& lastWrittenTimeStamp, bool& isVideoClosed)
{
	std::chrono::milliseconds duration(ts);
	std::chrono::seconds secondsInDuration = std::chrono::duration_cast<std::chrono::seconds>(duration);
	std::chrono::milliseconds msecsInDuration = std::chrono::duration_cast<std::chrono::milliseconds>(duration - secondsInDuration);

	std::chrono::time_point<std::chrono::system_clock> timePointInSeconds(secondsInDuration);
	std::time_t t = std::chrono::system_clock::to_time_t(timePointInSeconds);
	std::tm tm = *std::localtime(&t);
	uint16_t msecs = msecsInDuration.count();

	if (isVideoClosed)
	{
		lastVideoTS = 0;
		lastSyncTS = std::time(nullptr);
	}

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
		nextFrameFileName = filePath(relPath, mp4FileName, baseFolder);
		return;
	}

	// get new video path
	std::string yyyymmdd = std::to_string(1900 + tm.tm_year) + format_2(tm.tm_mon) + format_2(tm.tm_mday);
	relPath = boost::filesystem::path(yyyymmdd) / format_hrs(tm.tm_hour);
	mp4FileName = std::to_string(ts) + ".mp4";

	// cache new values
	currentFolder = baseFolder;
	lastVideoTS = t;
	lastVideoFolderPath = relPath;
	lastVideoMinute = tm.tm_min;
	lastVideoName = mp4FileName;
	lastWrittenTimeStamp = lastVideoName;

	nextFrameFileName = filePath(relPath, mp4FileName, baseFolder);
}
void Mp4WriterSinkUtils::parseTSH264(uint64_t& ts, uint32_t& chunkTimeInMinutes, uint32_t& syncTimeInSeconds, boost::filesystem::path& relPath,
	std::string& mp4FileName, bool& syncFlag, short frameType, short naluType, std::string baseFolder, std::string& nextFrameFileName, std::string& lastWrittenTimeStamp, bool& isVideoClosed)
{
	std::chrono::milliseconds duration(ts);

	std::chrono::seconds secondsInDuration = std::chrono::duration_cast<std::chrono::seconds>(duration);
	std::chrono::milliseconds msecsInDuration = std::chrono::duration_cast<std::chrono::milliseconds>(duration - secondsInDuration);

	std::chrono::time_point<std::chrono::system_clock> timePointInSeconds(secondsInDuration);
	std::time_t t = std::chrono::system_clock::to_time_t(timePointInSeconds);
	std::tm tm = *std::localtime(&t);
	uint16_t msecs = msecsInDuration.count();

	if (isVideoClosed)
	{
		lastVideoTS = 0;
		lastSyncTS = std::time(nullptr);
		isVideoClosed = false;
	}

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
		nextFrameFileName = filePath(relPath, mp4FileName, baseFolder);
		return;
	}
	// cannot be merged with if condition above.
	if (!((naluType == H264Utils::H264_NAL_TYPE::H264_NAL_TYPE_IDR_SLICE) || (naluType == H264Utils::H264_NAL_TYPE::H264_NAL_TYPE_SEQ_PARAM)))
	{
		relPath = lastVideoFolderPath;
		mp4FileName = lastVideoName;
		nextFrameFileName = tempNextFrameFileName;
		return;
	}
	// get new video path
	std::string yyyymmdd = std::to_string(1900 + tm.tm_year) + format_2(tm.tm_mon) + format_2(tm.tm_mday);
	relPath = boost::filesystem::path(yyyymmdd) / format_hrs(tm.tm_hour);
	mp4FileName = std::to_string(ts) + ".mp4";

	// cache new values
	currentFolder = baseFolder;
	lastVideoTS = t;
	lastVideoFolderPath = relPath;
	lastVideoMinute = tm.tm_min;
	lastVideoName = mp4FileName;
	lastWrittenTimeStamp = lastVideoName;

	nextFrameFileName = filePath(relPath, mp4FileName, baseFolder);
	tempNextFrameFileName = nextFrameFileName;
}

void Mp4WriterSinkUtils::getFilenameForNextFrame(std::string& nextFrameFileName, uint64_t& timestamp, std::string& basefolder,
	uint32_t chunkTimeInMinutes, uint32_t syncTimeInSeconds, bool& syncFlag, short& frameType, short naluType, std::string& lastWrittenTimeStamp, bool& isVideoClosed)
{
	boost::filesystem::path finalPath;
	std::string mp4FileName;
	boost::filesystem::path relPath;

	if (frameType == FrameMetadata::FrameType::H264_DATA)
	{
		parseTSH264(timestamp, chunkTimeInMinutes, syncTimeInSeconds, relPath, mp4FileName, syncFlag, frameType, naluType, basefolder, nextFrameFileName, lastWrittenTimeStamp, isVideoClosed);
	}
	else if (frameType == FrameMetadata::FrameType::ENCODED_IMAGE)
	{
		parseTSJpeg(timestamp, chunkTimeInMinutes, syncTimeInSeconds, relPath, mp4FileName, syncFlag, basefolder, nextFrameFileName, lastWrittenTimeStamp, isVideoClosed);
	}

}

Mp4WriterSinkUtils::~Mp4WriterSinkUtils()
{
}
