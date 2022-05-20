#pragma once

#include <vector>
#include <string>
#include <boost/shared_ptr.hpp>
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"

class FilenameStrategy;

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

    bool Read(uint8_t *&dataToRead, size_t &dataSize, uint64_t &index);
    bool ReadP(uint8_t *dataToRead, size_t &dataSize, uint64_t &index);
    bool Write(const uint8_t *dataToWrite, size_t dataSize);
    bool Write1(frame_sp frame);
    bool WriteFirst(frame_sp frame);

    void SetReadLoop(bool readLoop);
    void notifyPlay(bool play);
    void jump(uint64_t index);
    bool setNoOfFrame(int x);
    bool getCurrentStatus();
    bool resetInfo();
    int noOFFramesToCapture = 10;
    bool isPlaying = false;
    int framesCaptured = 0;

private:
    bool writeHelper(const std::string& fileName, const uint8_t* dataToWrite, size_t dataSize, bool append);

private:       
	bool mAppend;
	boost::shared_ptr<FilenameStrategy> mStrategy;
};
