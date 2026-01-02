#pragma once

#include <memory>
#include <filesystem>
#include <vector>
#include <string>

class FilenameStrategy
{
public:

	static std::shared_ptr<FilenameStrategy> getStrategy(const std::string& strPath,
		int startIndex,
		int maxIndex,
		bool readLoop,
		const std::vector<std::string>& files = std::vector<std::string>(),
		bool appendFlag = false
	);
		
	virtual ~FilenameStrategy();

	bool IsConnected() const;
	virtual bool Connect();
	bool Disconnect();

	virtual bool canCache();

	void play(bool play);
	void jump(uint64_t index);

	virtual std::string GetFileNameToUse(bool checkForExistence, uint64_t& index);

	void SetReadLoop(bool readLoop);
	static bool fileExists(const char *path);

	FilenameStrategy(const std::string& strPath,
		int startIndex,
		int maxIndex,
		bool readLoop,
		bool appendFlag);

	FilenameStrategy(bool readLoop);

protected:

	void incrementIndex();

	bool mIsConnected;
	bool mReadLoop;
	bool mAppend;

	std::string mDirName;
	int mCurrentIndex;

	int mStartIndex;
	int mMaxIndex;

private:	
	std::string GetFileNameForCurrentIndex(bool checkForExistence) const;

	std::string mFileBaseName;
	std::string mFileTailName;

	bool mPlay;
	short mWildCardLen;
};

class BoostDirectoryStrategy: public FilenameStrategy
{
public:
	virtual ~BoostDirectoryStrategy();

	bool Connect();
	bool canCache() { return false; }

	std::string GetFileNameToUse(bool checkForExistence, uint64_t& index);

	friend class FilenameStrategy;

	BoostDirectoryStrategy(const std::string& strPath,
		int startIndex,
		int maxIndex,
		bool readLoop);

protected:

private:
	std::vector<std::filesystem::path> mFiles;
};

class ListStrategy : public FilenameStrategy
{
public:
	virtual ~ListStrategy();

	bool Connect();
	bool canCache() { return false; }

	std::string GetFileNameToUse(bool checkForExistence, uint64_t& index);

	friend class FilenameStrategy;

	ListStrategy(const std::vector<std::string>& files, const std::string& dirPath, bool readLoop);

protected:

private:
	std::vector<std::string> mFiles;
	std::filesystem::path mRootDir;
};