#include "FilenameStrategy.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include "boost/format.hpp"

#define CH_WILD_CARD '?'
#ifdef _WIN32
#define SZ_FILE_SEPERATOR_STRING "\\"
#else
#define SZ_FILE_SEPERATOR_STRING "/"
#endif //_WIN32

boost::shared_ptr<FilenameStrategy> FilenameStrategy::getStrategy(const std::string& strPath,
	int startIndex,
	int maxIndex,
	bool readLoop,
	const std::vector<std::string>& files
)
{
	bool isDirectory = boost::filesystem::is_directory(strPath);	
		
	if (files.size())
	{
		return boost::shared_ptr<FilenameStrategy>(new ListStrategy(files, strPath, readLoop));
	}
	else if (isDirectory)
	{
		return boost::shared_ptr<FilenameStrategy>(new BoostDirectoryStrategy(strPath, startIndex, maxIndex, readLoop));
	}
	else
	{
		return boost::shared_ptr<FilenameStrategy>(new FilenameStrategy(strPath, startIndex, maxIndex, readLoop));		
	}
}

FilenameStrategy::FilenameStrategy(const std::string& strPath,
	int startIndex,
	int maxIndex,
	bool readLoop)
	: mIsConnected(false),
	mCurrentIndex(-1),
	mStartIndex(startIndex),
	mMaxIndex(maxIndex),
	mWildCardLen(0),
	mDirName(strPath),
	mReadLoop(readLoop),
	mPlay(false)
{

}

FilenameStrategy::FilenameStrategy(bool readLoop) :
	mIsConnected(false),
	mCurrentIndex(-1),
	mStartIndex(0),
	mMaxIndex(-1),
	mWildCardLen(0),
	mDirName(""),
	mReadLoop(readLoop),
	mPlay(false)
{

}

FilenameStrategy::~FilenameStrategy()
{

}

bool FilenameStrategy::IsConnected() const
{
	return mIsConnected;
}

bool FilenameStrategy::Connect()
{
	if (IsConnected())
	{
		return true;
	}

	bool connectionRes = false;

	std::string origPath = mDirName;

	const size_t fileStartIndex = mDirName.find_last_of("/\\");
	if (std::string::npos != fileStartIndex)
	{
		const std::string strFullFileNameWithPattern = mDirName;

		mDirName = strFullFileNameWithPattern.substr(0U, fileStartIndex);
		mFileBaseName = strFullFileNameWithPattern.substr(fileStartIndex + 1);
	}

	if (!mDirName.empty())
	{
		connectionRes = boost::filesystem::is_directory(mDirName.c_str());
	}

	if (connectionRes)
	{
		mCurrentIndex = mStartIndex;
		mWildCardLen = (short)std::count(mFileBaseName.begin(), mFileBaseName.end(), CH_WILD_CARD);

		if (mWildCardLen > 0)
		{
			const std::string strFileNameWithPattern = mFileBaseName;
			const std::string::size_type basePos = strFileNameWithPattern.find_first_of(CH_WILD_CARD);

			mFileBaseName = strFileNameWithPattern.substr(0, basePos);
			mFileTailName = strFileNameWithPattern.substr(basePos + mWildCardLen);
		}
		else if (!boost::filesystem::exists(boost::filesystem::path(origPath)))
		{
			// if not a directory or pattern .. then it is a single file .. so checking if path exist
			LOG_ERROR << "DirName " << origPath << " is Invalid";
			connectionRes = true;			
		}
	}
	else
	{
		LOG_ERROR << "DirName " << mDirName << " is Invalid";
		assert(false); //Passed Dir is invalid
	}	

	return connectionRes;
}

bool FilenameStrategy::Disconnect()
{
	mCurrentIndex = -1;
	return true;
}

bool FilenameStrategy::canCache() 
{
	return mWildCardLen == 0;
}

void FilenameStrategy::play(bool play)
{
	mPlay = play;
}

void FilenameStrategy::incrementIndex()
{
	if (mPlay)
	{
		mCurrentIndex++;
	}
}

void FilenameStrategy::jump(uint64_t index)
{
	// any validation required?
	if (index < mStartIndex)
	{
		index = mStartIndex;
	}
	mCurrentIndex = index;
}

std::string FilenameStrategy::GetFileNameToUse(bool checkForExistence, uint64_t& index)
{	
	std::string fileNameToUse;

	if (mCurrentIndex >= 0)
	{
		fileNameToUse = GetFileNameForCurrentIndex(checkForExistence);

		if (mWildCardLen > 0)
		{
			if (fileNameToUse.empty())
			{
				if (mCurrentIndex > 0)
				{
					if (mReadLoop)
					{
						mCurrentIndex = mStartIndex;						
						fileNameToUse = GetFileNameForCurrentIndex(checkForExistence);
					}
				}
				else
				{
					assert(false); //Even First File Not Present
				}
			}
						
			index = mCurrentIndex; // propagating file index
			incrementIndex();

			if (mMaxIndex >= 0
				&& mCurrentIndex > mMaxIndex)
			{
				mCurrentIndex = mStartIndex;
			}
		}

	}	

	return fileNameToUse;
}

std::string FilenameStrategy::GetFileNameForCurrentIndex(bool checkForExistence) const
{
	std::string strFileNameForIndex;

	std::string strIndexedName;

	if (mWildCardLen > 0)
	{
		// https://www.boost.org/doc/libs/1_71_0/libs/format/doc/format.html								
		auto fmt = boost::format("%0"+ std::to_string(mWildCardLen)+"d") % mCurrentIndex;
		strIndexedName = fmt.str();		
	}

	strFileNameForIndex = mDirName + SZ_FILE_SEPERATOR_STRING + mFileBaseName
		+ strIndexedName + mFileTailName;

	if (checkForExistence)
	{
		if (!fileExists(strFileNameForIndex.c_str()))
		{
			strFileNameForIndex = std::string();
		}
	}
	return strFileNameForIndex;
}

void FilenameStrategy::SetReadLoop(bool readLoop)
{
	mReadLoop = readLoop;
}

bool FilenameStrategy::fileExists(const char *path)
{
	return boost::filesystem::exists(path);
}

BoostDirectoryStrategy::BoostDirectoryStrategy(const std::string& strPath,	int startIndex,	int maxIndex, bool readLoop) : FilenameStrategy(strPath, startIndex, maxIndex, readLoop)
{
	
}

BoostDirectoryStrategy::~BoostDirectoryStrategy()
{

}

bool BoostDirectoryStrategy::Connect()
{ 
	if (IsConnected())
	{
		return true;
	}
		
	for (auto &&itr : boost::filesystem::directory_iterator(mDirName))
	{
		auto fileNameToUse = itr.path().generic_string();
		if(boost::filesystem::is_regular_file(fileNameToUse))
		{
			mFiles.push_back(itr.path());
		}		
	}

	std::sort(mFiles.begin(), mFiles.end());

	if(mMaxIndex == -1 || mMaxIndex >= mFiles.size())
	{
		mMaxIndex = ((int)mFiles.size()) - 1;
	}

	if (mMaxIndex == -1)
	{
		throw AIPException(AIP_NOTFOUND, "No Matching files found in <" + mDirName+">");
	}

	if (mStartIndex != 0 && mStartIndex > mMaxIndex)
	{
		throw AIPException(AIP_PARAM_OUTOFRANGE, std::string("start index is out of range"));
	}

	mCurrentIndex = mStartIndex;

	return true;
}

std::string BoostDirectoryStrategy::GetFileNameToUse(bool checkForExistence, uint64_t& index)
{
	if (  mCurrentIndex > mMaxIndex && mReadLoop)
	{
		// current index is greater than max index 
		// readLoop is true

		mCurrentIndex = mStartIndex;
	}	

	if (mCurrentIndex < 0 || mCurrentIndex > mMaxIndex)
	{
		return std::string();
	}

	index = mCurrentIndex; // propagating file index	
	auto _index = mCurrentIndex;
	incrementIndex();
	   
	return mFiles[_index].generic_string();
}

ListStrategy::~ListStrategy()
{

}

bool ListStrategy::Connect()
{
	return true;
}

std::string ListStrategy::GetFileNameToUse(bool checkForExistence, uint64_t& index)
{
	if (mCurrentIndex > mMaxIndex && mReadLoop)
	{
		// looping
		mCurrentIndex = 0;
	}

	if (mCurrentIndex < 0 || mCurrentIndex > mMaxIndex)
	{
		return std::string();
	}
		
	index = mCurrentIndex; // propagating file index
	auto _index = mCurrentIndex;
	incrementIndex();

	return (mRootDir /mFiles[_index]).string();
}

ListStrategy::ListStrategy(const std::vector<std::string>& files, const std::string& dirPath, bool readLoop) : FilenameStrategy(readLoop)
{
	// deep copy
	mFiles = files;
	mMaxIndex = files.size() - 1;

	if (mMaxIndex < 0)
	{
		throw AIPException(AIP_NOTFOUND, "List is Empty");
	}

	mCurrentIndex = 0;
	mRootDir = boost::filesystem::path(dirPath);
}