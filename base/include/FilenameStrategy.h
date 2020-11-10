#pragma once

#include <boost/shared_ptr.hpp>
#include <boost/filesystem.hpp>

class FilenameStrategy
{
public:

	static boost::shared_ptr<FilenameStrategy> getStrategy(const std::string& strPath,
		int startIndex,
		int maxIndex,
		bool readLoop,
		const std::vector<std::string>& files = std::vector<std::string>()
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

protected:
	FilenameStrategy(const std::string& strPath,
		int startIndex,
		int maxIndex,
		bool readLoop);
	
	FilenameStrategy(bool readLoop);

	void incrementIndex();

	bool mIsConnected;
	bool mReadLoop;

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

protected:
	BoostDirectoryStrategy(const std::string& strPath,
		int startIndex,
		int maxIndex,
		bool readLoop);

private:	
	std::vector<boost::filesystem::path> mFiles;	
};

class ListStrategy : public FilenameStrategy
{
public:
	virtual ~ListStrategy();

	bool Connect();
	bool canCache() { return false; }

	std::string GetFileNameToUse(bool checkForExistence, uint64_t& index);

	friend class FilenameStrategy;

protected:
	ListStrategy(const std::vector<std::string>& files, const std::string& dirPath, bool readLoop);

private:
	std::vector<std::string> mFiles;
	boost::filesystem::path mRootDir;
};
