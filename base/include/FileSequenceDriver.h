#pragma once

#include <vector>
#include <string>
#include <memory>

class FilenameStrategy;
class BufferMaker;
class FileSequenceDriver
{
public:
    FileSequenceDriver(const std::string& strPath,
                        int startIndex = 0, int maxIndex =  -1, bool readLoop = true, const std::vector<std::string>& files = std::vector<std::string>());

	FileSequenceDriver(const std::string& strPath, bool append);

    virtual ~FileSequenceDriver();

    FileSequenceDriver(const FileSequenceDriver& other) = delete;
    FileSequenceDriver& operator=(const FileSequenceDriver& other) = delete;

    bool Connect();
    bool Disconnect();
    bool IsConnected() const;
	bool canCache();

    bool Read(uint8_t*& dataToRead, size_t& dataSize, uint64_t& index);
	bool ReadP(BufferMaker& buffMaker, uint64_t& index);
    bool ReadP(BufferMaker& buffMaker, uint64_t& index, size_t& userMetadataSize);
    bool Write(const uint8_t* dataToWrite, size_t dataSize);
	    
    void SetReadLoop(bool readLoop);
	void notifyPlay(bool play);
	void jump(uint64_t index);

private:
    bool writeHelper(const std::string& fileName, const uint8_t* dataToWrite, size_t dataSize, bool append);

private:
	bool mAppend;
	std::shared_ptr<FilenameStrategy> mStrategy;
};
