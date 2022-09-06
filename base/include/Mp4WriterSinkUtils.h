#include <ctime>

class Mp4WriterSinkUtils
{
public:
	Mp4WriterSinkUtils();
	std::string getFilenameForNextFrame(uint64_t &timestamp, std::string &basefolder,
		uint32_t chunkTimeInMinutes, uint32_t syncTimeInMinutes, bool &syncFlag,short& frameType, short naluType);
	void parseTSJpeg(uint64_t &tm, uint32_t &chunkTimeInMinutes, uint32_t &syncTimeInMinutes,
		boost::filesystem::path &relPath, std::string &mp4FileName, bool &syncFlag, std::string baseFolder);
	void parseTSH264(uint64_t& tm, uint32_t& chunkTimeInMinutes, uint32_t& syncTimeInMinutes,
		boost::filesystem::path& relPath, std::string& mp4FileName, bool& syncFlag,short frameType, short naluType, std::string baseFolder);
	std::string format_hrs(int &hr);
	std::string format_2(int &min);
	~Mp4WriterSinkUtils();
private:
	int lastVideoMinute=0;
	std::time_t lastVideoTS;
	std::string lastVideoName;
	std::time_t lastSyncTS;
	std::string currentFolder;
	boost::filesystem::path lastVideoFolderPath;
};