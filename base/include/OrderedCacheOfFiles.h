#pragma once

#include <boost/filesystem.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/identity.hpp>
#include <boost/multi_index/member.hpp>

#include <string>
#define batch_size 1440

namespace fs = boost::filesystem;
namespace bmi = boost::multi_index;


class OrderedCacheOfFiles
{
public:
	OrderedCacheOfFiles(std::string& video_folder, uint32_t initial_batch_size = 1440, uint32_t _lowerWaterMark = 1441, uint32_t _upperWaterMark = 2880);
	~OrderedCacheOfFiles()
	{
		if (!cleanCacheOnMainThread && mThread)
		{
			mThread->join();
		}
	}
	void updateBatchSize(uint32_t _batchSize)
	{
		batchSize = _batchSize;
	}
	void cleanCacheOnSeperateThread(bool flag)
	{
		cleanCacheOnMainThread = !flag;
	}
	void deleteLostEntry(std::string& filePath);
	uint64_t getFileDuration(std::string& filename);
	// Note - getFileAt() is an unreliable method. Use ONLY if you know what you are doing.
	std::string getFileAt(uint64_t timestamp, bool direction);
	bool isTimeStampInFile(std::string& filename, uint64_t timestamp);
	std::string getNextFileAfter(std::string& currentFile, bool direction);
	std::vector<boost::filesystem::path> parseAndSortDateDir(const std::string& rootDir);
	std::vector<boost::filesystem::path> parseAndSortHourDir(const std::string& rootDir);
	std::vector<boost::filesystem::path> parseAndSortMp4Files(const std::string& rootDir);
	bool parseFiles(uint64_t start_ts, bool direction, bool includeFloorFile = false, bool disableBatchSizeCheck = false, uint64_t skipTS = 0);
	bool getRandomSeekFile(uint64_t ts, bool direction, uint64_t& skipMsecs, std::string& fileName);
	bool getFileFromCache(uint64_t timestamp, bool direction, std::string& fileName);
	size_t getCacheSize()
	{
		return videoCache.size();
	}
	bool fetchAndUpdateFromDisk(std::string videoFile, uint64_t& start_ts, uint64_t& end_ts);
	bool fetchFromCache(std::string& videoFile, uint64_t& start_ts, uint64_t& end_ts);
	void readVideoStartEnd(std::string& filePath, uint64_t& start_ts, uint64_t& end_ts);
	void clearCache();
	bool refreshCache();
	std::string getLastVideoInCache() { return videoCache.rbegin()->path; }
	void updateCache(std::string& filePath, uint64_t& start_ts, uint64_t& end_ts); // allow updates from playback
	std::map<std::string, std::pair<uint64_t, uint64_t>> getSnapShot(); // too costly, use for debugging only
	bool probe(boost::filesystem::path dirPath, std::string& videoName);
	bool getPreviousAndNextFile(std::string videoPath, std::string& previousFile, std::string& nextFile);
private:
	bool lastKnownPlaybackDir = true; // sync with mp4 playback
	boost::mutex m_mutex;
	boost::shared_ptr<boost::thread> mThread = nullptr;
	int cacheSize = 1440;
	int batchSize = 1440;
	std::string rootDir = "";
	uint32_t lowerWaterMark;
	uint32_t upperWaterMark;
	bool cleanCacheOnMainThread = true;
	/* Util methods */
	bool filePatternCheck(const fs::path& path);
	bool datePatternCheck(const boost::filesystem::path& path);
	bool hourPatternCheck(const boost::filesystem::path& path);
	//bool openFileUpdateCacheByIter(iter);
	//bool openFileUpdateCacheByFilename(filePath);

	/* ---------Cache Stuff ------------*/
	struct CacheIteratorState
	{
		std::string END_ITER = "END_ITERATOR";
	} cacheIteratorState;

	// cache entry format
	struct Video
	{
		uint64_t start_ts, end_ts;
		std::string path;

		Video(std::string& _path, uint64_t _start_ts) : path(_path), start_ts(_start_ts)
		{
			end_ts = 0;
		}

		Video(const std::string _path, uint64_t _start_ts) : path(_path), start_ts(_start_ts)
		{
			end_ts = 0;
		}

		Video(std::string _path, uint64_t _start_ts, uint64_t _end_ts) : path(_path), start_ts(_start_ts), end_ts(_end_ts) {}

		void updateEndTS(uint64_t ts)
		{
			end_ts = ts;
		}
	};

	// cache index tags
	struct videoPath {};
	struct videoTS {};

	// VideoCache data type
	typedef boost::multi_index_container<
		Video,
		bmi::indexed_by<
		// sort by less<string> on path
		bmi::ordered_unique<bmi::tag<videoPath>, bmi::member<Video, std::string, &Video::path> >,

		// sort by less<int> on videoTS
		bmi::ordered_non_unique<bmi::tag<videoTS>, bmi::member<Video, uint64_t, &Video::start_ts> >
		>
	> VideoCache;

	// Cache object and index
	VideoCache videoCache;
	VideoCache::index<videoTS>::type& videoCacheStartTSIndex = videoCache.get<videoTS>();

	// Cache methods
	void insertInVideoCache(Video vid);

	// Cache cleanup strategy
	void retireOldFiles(uint64_t startTSofRelevantFile);
	void dropFarthestFromTS(uint64_t startTS);

};