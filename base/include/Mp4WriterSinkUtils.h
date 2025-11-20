#include <ctime>
#include <chrono>
#include <string>
#include <filesystem>

class Mp4WriterSinkUtils
{
public:
	Mp4WriterSinkUtils();
	void getFilenameForNextFrame(std::string& nextFrameFileName ,uint64_t &timestamp, std::string &basefolder,
		uint32_t chunkTimeInMinutes, uint32_t syncTimeInSeconds, bool &syncFlag,short& frameType, short naluType);
	void parseTSJpeg(uint64_t &tm, uint32_t &chunkTimeInMinutes, uint32_t & syncTimeInSeconds,
		std::filesystem::path &relPath, std::string &mp4FileName, bool &syncFlag, std::string baseFolder, std::string& nextFrameFileName);
	void parseTSH264(uint64_t& tm, uint32_t& chunkTimeInMinutes, uint32_t& syncTimeInSeconds,std::filesystem::path& relPath,
		std::string& mp4FileName, bool& syncFlag,short frameType, short naluType, std::string baseFolder, std::string& nextFrameFileName);
	bool customNamedFileDirCheck(std::string baseFolder, uint32_t chunkTimeInMinutes, std::filesystem::path relPath, std::string& nextFrameFileName);
	std::string format_hrs(int &hr);
	std::string format_2(int &min);
	std::string filePath(std::filesystem::path relPath, std::string mp4FileName, std::string baseFolder, uint64_t chunkTimeInMins);
	~Mp4WriterSinkUtils();
private:
	int lastVideoMinute=0;
	std::time_t lastVideoTS;
	std::string lastVideoName;
	std::time_t lastSyncTS;
	std::string currentFolder;
	std::filesystem::path lastVideoFolderPath;
	std::string tempNextFrameFileName;
};