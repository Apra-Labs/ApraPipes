#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/identity.hpp>
#include <boost/multi_index/member.hpp>

#include "Logger.h"
#include "OrderedCacheOfFiles.h"
#include "AIPExceptions.h"

BOOST_AUTO_TEST_SUITE(ordered_file_cache)

struct LoggerSetup
{
	LoggerSetup()
	{
		LoggerProps loggerProps;
		loggerProps.logLevel = boost::log::trivial::severity_level::info;
		Logger::setLogLevel(boost::log::trivial::severity_level::info);
		Logger::initLogger(loggerProps);
	}

	~LoggerSetup()
	{
	}
};

struct DiskFiles
{
	std::map<uint64_t, std::string> files =
	{
		{1655895162221, "data/Mp4_videos//mp4_seek_tests/20220522/0016/1655895162221.mp4"},
		{1655895288956, "data/Mp4_videos//mp4_seek_tests/20220522/0016/1655895288956.mp4"},
		{1655919060000, "data/Mp4_videos//mp4_seek_tests/20220522/0023/1655919060000.mp4"},
		{1655926320000, "data/Mp4_videos//mp4_seek_tests/20220523/0001/1655926320000.mp4"}

	};

	/* sts -> duration (msec) [imprecise] */
	std::map<uint64_t, uint64_t> fileDurations =
	{
		{1655895162221, 4000},
		{1655895288956, 10005},
		{1655919060000, 22000},
		{1655926320000, 60000}

	};
};

/* -----------Functionality Tests---------------- */

BOOST_AUTO_TEST_CASE(fsParseExactMatch)
{
	LoggerSetup setup;
	DiskFiles diskFiles;
	std::string dir = "data/Mp4_videos/mp4_seek_tests";
	dir = boost::filesystem::canonical(dir).string();
	OrderedCacheOfFiles cof(dir, 100, 100, 1000);
	cof.parseFiles(102, true);
	std::string fileName;
	// check the map
	bool direction = true;
	for (auto it = diskFiles.files.begin(); it != diskFiles.files.end(); ++it)
	{
		auto timestamp = it->first;
		auto isFilefound = cof.getFileFromCache(timestamp, direction, fileName);
		BOOST_TEST(isFilefound);
		LOG_INFO << "Checking: fsParseExactMatch" << it->second << "<>" << fileName;
		BOOST_TEST(boost::filesystem::equivalent(boost::filesystem::path(it->second), boost::filesystem::path(fileName)));
	}
	// reverse 
	direction = false;
	for (auto it = diskFiles.files.begin(); it != diskFiles.files.end(); ++it)
	{
		auto timestamp = it->first;
		auto isFilefound = cof.getFileFromCache(timestamp, direction, fileName);
		BOOST_TEST(isFilefound);
		LOG_INFO << "Checking: fsParseExactMatch" << it->second << "<>" << fileName;
		BOOST_TEST(boost::filesystem::equivalent(boost::filesystem::path(it->second), boost::filesystem::path(fileName)));
	}
}

BOOST_AUTO_TEST_CASE(fwdBasic_FileOpen_QueryHole)
{
	LoggerSetup setup;
	DiskFiles diskFiles;
	std::string dir = "data/Mp4_videos/mp4_seek_tests";
	dir = boost::filesystem::canonical(dir).string();
	OrderedCacheOfFiles cof(dir, 100, 100, 1000);
	cof.parseFiles(102, true);
	std::string fileName;

	// check the map
	bool direction = true;
	for (auto it = diskFiles.files.begin(); it != diskFiles.files.end(); ++it)
	{
		auto ts = it->first + diskFiles.fileDurations[it->first] + 5;
		auto isFileFound = cof.getFileFromCache(ts, direction, fileName);
		if (isFileFound)
		{
			BOOST_TEST(isFileFound);
			auto nextDiskFile = it;
			nextDiskFile++;
			LOG_INFO << "Checking bwdBasicOpen_NoCacheUse: ts " << it->first << "> " << fileName << "<>" << nextDiskFile->second;
			BOOST_TEST(boost::filesystem::equivalent(boost::filesystem::path(fileName), boost::filesystem::path(nextDiskFile->second)));
		}
		else
		{
			BOOST_TEST(fileName == "");
		}
	}


}

BOOST_AUTO_TEST_CASE(bwd_BasicFileOpen_QueryHole)
{
	LoggerSetup setup;
	DiskFiles diskFiles;
	std::string dir = "data/Mp4_videos/mp4_seek_tests";
	dir = boost::filesystem::canonical(dir).string();
	OrderedCacheOfFiles cof(dir, 100, 100, 1000);
	cof.parseFiles(102, true);
	std::string fileName;
	// check the map
	bool direction = false;
	for (auto it = diskFiles.files.begin(); it != diskFiles.files.end(); ++it)
	{
		auto ts = it->first + diskFiles.fileDurations[it->first];

		auto isFileFound = cof.getFileFromCache(ts, direction, fileName);
		BOOST_TEST(isFileFound);
		LOG_INFO << "Checking bwdBasicOpen_NoCacheUse: ts " << it->first << "> " << fileName << "<>" << it->second;
		BOOST_TEST(boost::filesystem::equivalent(boost::filesystem::path(fileName), boost::filesystem::path(it->second)));

	}
}

BOOST_AUTO_TEST_CASE(biDirectional_BasicFileOpen_QueryMidFile)
{
	LoggerSetup setup;
	DiskFiles diskFiles;
	std::string dir = "data/Mp4_videos/mp4_seek_tests";
	dir = boost::filesystem::canonical(dir).string();
	OrderedCacheOfFiles cof(dir, 100, 100, 1000);
	cof.parseFiles(102, true);
	std::string fileName;
	// check the map
	bool direction = true;
	for (auto it = diskFiles.files.begin(); it != diskFiles.files.end(); ++it)
	{
		// As the queryTS is present in the file , getFileFromCache() will give the actual fileName and return true.
		auto ts = it->first + diskFiles.fileDurations[it->first] - 1200;
		auto isFileFound = cof.getFileFromCache(ts, direction, fileName);
		BOOST_TEST(isFileFound);
		LOG_INFO << "Checking bwdBasicOpen_NoCacheUse: ts " << it->first << "> " << fileName << "<>" << it->second;
		BOOST_TEST(boost::filesystem::equivalent(boost::filesystem::path(fileName), boost::filesystem::path(it->second)));
	}

	direction = false;
	for (auto it = diskFiles.files.begin(); it != diskFiles.files.end(); ++it)
	{
		auto ts = it->first + diskFiles.fileDurations[it->first] - 1200;
		auto isFileFound = cof.getFileFromCache(ts, direction, fileName);
		BOOST_TEST(isFileFound);
		LOG_INFO << "Checking bwdBasicOpen_NoCacheUse: ts " << it->first << "> " << fileName << "<>" << it->second;
		BOOST_TEST(boost::filesystem::equivalent(boost::filesystem::path(fileName), boost::filesystem::path(it->second)));
	}
}


BOOST_AUTO_TEST_CASE(random_seek_fwd_queryBeforeCache)
{
	LoggerSetup setup;
	DiskFiles diskFiles;

	std::string dir = "data/Mp4_videos/mp4_seek_tests";
	dir = boost::filesystem::canonical(dir).string();
	OrderedCacheOfFiles cof(dir, 100, 100, 100);
	uint64_t skipMsecs = 0;
	// parse all files
	cof.parseFiles(102, true);
	// beyond EOF
	auto timeStamp = 1655895162221 - 10000;
	bool direction = true;
	std::string fileName;
	//EOF returns empty string for the fileName and isFilePresent becomes false
	auto isFilePresent = cof.getRandomSeekFile(timeStamp, direction, skipMsecs, fileName);

	BOOST_TEST(skipMsecs == 0);
	BOOST_TEST(isFilePresent);
	BOOST_TEST(boost::filesystem::equivalent(diskFiles.files[1655895162221], fileName));
}

/* ----------End Of Files Detection Tests----------- */
BOOST_AUTO_TEST_CASE(fwdEOFDetection)
{
	LoggerSetup setup;
	DiskFiles diskFiles;
	std::string dir = "data/Mp4_videos/mp4_seek_tests";
	dir = boost::filesystem::canonical(dir).string();
	OrderedCacheOfFiles cof(dir, 100, 100, 1000);
	cof.parseFiles(102, true);
	// check the map
	bool direction = true;

	auto diskFileIter = diskFiles.files.begin();
	auto currentFile = boost::filesystem::canonical(diskFileIter->second).string();
	try
	{
		while (true)
		{

			currentFile = cof.getNextFileAfter(currentFile, direction);
			++diskFileIter;
			BOOST_TEST(boost::filesystem::equivalent(boost::filesystem::path(currentFile), boost::filesystem::path(diskFileIter->second)));
		}
	}
	catch (AIP_Exception& exception)
	{
		BOOST_TEST(exception.getCode() == MP4_OCOF_END);
	}
	catch (...)
	{
		BOOST_TEST(false);
	}
}

BOOST_AUTO_TEST_CASE(bwdEOFDetection_getNextFileAfter)
{
	LoggerSetup setup;
	DiskFiles diskFiles;
	std::string dir = "data/Mp4_videos/mp4_seek_tests";
	dir = boost::filesystem::canonical(dir).string();
	OrderedCacheOfFiles cof(dir, 100, 100, 1000);
	cof.parseFiles(102, true);
	// check the map
	bool direction = false;

	auto diskFileIter = diskFiles.files.rbegin();
	auto currentFile = boost::filesystem::canonical(diskFileIter->second).string();
	try
	{
		while (true)
		{

			currentFile = cof.getNextFileAfter(currentFile, direction);
			++diskFileIter;
			BOOST_TEST(boost::filesystem::equivalent(boost::filesystem::path(currentFile), boost::filesystem::path(diskFileIter->second)));
		}
	}
	catch (AIP_Exception& exception)
	{
		BOOST_TEST(exception.getCode() == MP4_OCOF_END);
	}
}

// skipVideoFile = 
// skipMsecs = ts - start_ts of skipVideoFile
// return 1 - if success 
// return 0 - if fails
// return -1 -> EOF -> catch(..) => return -1

BOOST_AUTO_TEST_CASE(random_seek_fwd_query_mid_file)
{
	LoggerSetup setup;
	DiskFiles diskFiles;

	std::string dir = "data/Mp4_videos/mp4_seek_tests";
	dir = boost::filesystem::canonical(dir).string();
	OrderedCacheOfFiles cof(dir, 2, 100, 1000);
	bool direction = true;
	cof.parseFiles(102, direction);
	uint64_t skipMsecs = 0;
	auto iter = diskFiles.files.begin();
	++iter;
	auto timeStamp = iter->first + 200;
	std::string fileName;
	auto ret = cof.getRandomSeekFile(timeStamp, direction, skipMsecs, fileName);
	BOOST_TEST(boost::filesystem::equivalent(iter->second, fileName));
	BOOST_TEST(ret);
}

BOOST_AUTO_TEST_CASE(random_seek_bwd_query_mid_file)
{
	LoggerSetup setup;
	DiskFiles diskFiles;

	std::string dir = "data/Mp4_videos/mp4_seek_tests";
	dir = boost::filesystem::canonical(dir).string();
	OrderedCacheOfFiles cof(dir, 2, 100, 1000);
	bool direction = false;
	cof.parseFiles(102, true);
	uint64_t skipMsecs = 0;
	std::string fileName;

	auto iter = diskFiles.files.begin();
	++iter;
	auto timeStamp = iter->first + 200;

	auto ret = cof.getRandomSeekFile(timeStamp, direction, skipMsecs, fileName);

	BOOST_TEST(boost::filesystem::equivalent(iter->second, fileName));
	BOOST_TEST(ret);
}

BOOST_AUTO_TEST_CASE(random_seek_bwd_queryhole)
{
	LoggerSetup setup;
	DiskFiles diskFiles;

	std::string dir = "data/Mp4_videos/mp4_seek_tests";
	dir = boost::filesystem::canonical(dir).string();
	OrderedCacheOfFiles cof(dir, 100, 100, 1000);
	cof.parseFiles(102, true);
	uint64_t skipMsecs = 0;
	auto timeStamp = diskFiles.fileDurations.begin()->first + diskFiles.fileDurations.begin()->second + 300;
	bool direction = false;
	std::string fileName;
	auto ret = cof.getRandomSeekFile(timeStamp, direction, skipMsecs, fileName);
	BOOST_TEST(skipMsecs == 0);
	BOOST_TEST(ret);
	BOOST_TEST(boost::filesystem::equivalent(fileName, diskFiles.files.begin()->second));
}

BOOST_AUTO_TEST_CASE(random_seek_fwd_EOC)
{
	LoggerSetup setup;
	DiskFiles diskFiles;

	std::string dir = "data/Mp4_videos/mp4_seek_tests";
	dir = boost::filesystem::canonical(dir).string();
	OrderedCacheOfFiles cof(dir, 2, 100, 1000);
	cof.parseFiles(102, true); // only first two files are now in cache
	uint64_t skipMsecs = 0;

	// ts of third
	uint64_t timeStamp = 1655895288956 + 25000;
	bool direction = true;
	std::string fileName;

	auto isFileInCache = cof.getFileFromCache(timeStamp, direction, fileName);
	BOOST_TEST(!isFileInCache);

	auto isFilePresent = cof.getRandomSeekFile(timeStamp, direction, skipMsecs, fileName);
	BOOST_TEST(isFilePresent);
}

BOOST_AUTO_TEST_CASE(random_seek_fwd_EOC_green)
{
	// green => cache has [3,4] - seek happens in file [1,2]
	LoggerSetup setup;
	DiskFiles diskFiles;

	std::string dir = "data/Mp4_videos/mp4_seek_tests";
	dir = boost::filesystem::canonical(dir).string();

	OrderedCacheOfFiles cof(dir, 2, 100, 1000);
	// last one files in cache
	bool direction = true;
	cof.parseFiles(1655919060000, direction);

	// skip is in middle of second file
	uint64_t skipTS = 1655895288956 + 2000;
	std::string fileName;
	uint64_t skipMsecs;

	auto isFilePresent = cof.getRandomSeekFile(skipTS, direction, skipMsecs, fileName);
	BOOST_TEST(boost::filesystem::equivalent(fileName, diskFiles.files[1655895288956]));
	BOOST_TEST(skipMsecs == 2000);
}

BOOST_AUTO_TEST_CASE(random_seek_bwd_EOC_green)
{
	// green => cache has files [1,2] - seek happens in file [3,4]
	LoggerSetup setup;
	DiskFiles diskFiles;

	std::string dir = "data/Mp4_videos/mp4_seek_tests";
	dir = boost::filesystem::canonical(dir).string();

	OrderedCacheOfFiles cof(dir, 2, 100, 1000);
	// first files in cache
	bool direction = false;
	bool includeExactMatch = true;
	cof.parseFiles(1655895288956, direction, includeExactMatch);

	// skip is in middle of third file
	uint64_t skipTS = 1655919060000 + 2000;
	std::string fileName;
	uint64_t skipMsecs;

	auto isFilePresent = cof.getRandomSeekFile(skipTS, direction, skipMsecs, fileName);
	BOOST_TEST(boost::filesystem::equivalent(fileName, diskFiles.files[1655919060000]));
	BOOST_TEST(skipMsecs == 2000);
}

BOOST_AUTO_TEST_CASE(random_seek_bwd_EOC)
{
	LoggerSetup setup;
	DiskFiles diskFiles;

	std::string dir = "data/Mp4_videos/mp4_seek_tests";
	dir = boost::filesystem::canonical(dir).string();
	OrderedCacheOfFiles cof(dir, 2, 3, 3);
	cof.parseFiles(102, true);
	uint64_t skipMsecs = 0;
	auto timeStamp = 1655895162221 + 100;
	bool direction = false;
	std::string fileName;

	cof.parseFiles(1655919060000 - 1, true);

	auto isFileInCache = cof.getFileFromCache(timeStamp, direction, fileName);
	BOOST_TEST(!isFileInCache);

	auto isFilePresent = cof.getRandomSeekFile(timeStamp, direction, skipMsecs, fileName);
	BOOST_TEST(isFilePresent);
}

BOOST_AUTO_TEST_CASE(random_seek_bwd_EOF)
{
	LoggerSetup setup;
	DiskFiles diskFiles;

	std::string dir = "data/Mp4_videos/mp4_seek_tests";
	dir = boost::filesystem::canonical(dir).string();
	OrderedCacheOfFiles cof(dir, 3, 100, 1000);
	cof.parseFiles(102, true);
	uint64_t skipMsecs = 0;
	auto timeStamp = diskFiles.fileDurations.begin()->first - 1000;
	bool direction = false;
	std::string fileName;
	//EOF returns empty string for the fileName and isFilePresent becomes false
	auto isFilePresent = cof.getRandomSeekFile(timeStamp, direction, skipMsecs, fileName);

	BOOST_TEST(skipMsecs == 0);
	BOOST_TEST(!isFilePresent);
	BOOST_TEST(fileName == "");
}

BOOST_AUTO_TEST_CASE(random_seek_fwd_EOF)
{
	LoggerSetup setup;
	DiskFiles diskFiles;

	std::string dir = "data/Mp4_videos/mp4_seek_tests";
	dir = boost::filesystem::canonical(dir).string();
	OrderedCacheOfFiles cof(dir, 100, 100, 100);
	uint64_t skipMsecs = 0;
	// parse all files
	cof.parseFiles(102, true);
	// beyond EOF
	auto timeStamp = 1655926320000 + 90000;
	bool direction = true;
	std::string fileName;
	//EOF returns empty string for the fileName and isFilePresent becomes false
	auto isFilePresent = cof.getRandomSeekFile(timeStamp, direction, skipMsecs, fileName);

	BOOST_TEST(skipMsecs == 0);
	BOOST_TEST(!isFilePresent);
	BOOST_TEST(fileName == "");
}

BOOST_AUTO_TEST_CASE(dropstrategy_fwd)
{
	LoggerSetup setup;
	DiskFiles diskFiles;

	std::string dir = "data/Mp4_videos/mp4_seek_tests";
	dir = boost::filesystem::canonical(dir).string();
	uint32_t cacheSize = 2;
	uint32_t lowerWaterMark = 3;
	uint32_t upperWaterMark = 3;

	OrderedCacheOfFiles cof(dir, cacheSize, lowerWaterMark, upperWaterMark);

	cof.parseFiles(102, true);
	bool direction = true;
	std::string fileName;
	std::string secondFileNameBeforeDeleting;
	std::string secondFileNameAfterDeleting;
	cof.getFileFromCache(0, direction, secondFileNameBeforeDeleting);

	cof.parseFiles(1655919060000 - 100, true);

	cof.getFileFromCache(0, direction, secondFileNameAfterDeleting);

	BOOST_TEST(secondFileNameBeforeDeleting != secondFileNameAfterDeleting);

	auto videoCacheSize = cof.getCacheSize();

	BOOST_TEST(videoCacheSize == 2);
}

BOOST_AUTO_TEST_CASE(dropstrategy_bwd)
{
	LoggerSetup setup;
	DiskFiles diskFiles;

	std::string dir = "data/Mp4_videos/mp4_seek_tests";
	dir = boost::filesystem::canonical(dir).string();
	uint32_t cacheSize = 2;
	uint32_t lowerWaterMark = 3;
	uint32_t upperWaterMark = 3;

	OrderedCacheOfFiles cof(dir, cacheSize, lowerWaterMark, upperWaterMark);

	cof.parseFiles(102, true);
	bool direction = true;
	std::string fileName;
	std::string secondFileNameBeforeDeleting;
	std::string secondFileNameAfterDeleting;
	cof.getFileFromCache(0, direction, secondFileNameBeforeDeleting);

	auto videoCacheSize = cof.getCacheSize();

	BOOST_TEST(videoCacheSize == 2);

	cof.parseFiles(1655919060000 - 100, true);

	cof.getFileFromCache(0, direction, secondFileNameAfterDeleting);

	BOOST_TEST(secondFileNameBeforeDeleting != secondFileNameAfterDeleting);

	cof.parseFiles(UINT64_MAX, false);

	std::string firstFileNameAfterDeletingInBwd;
	cof.getFileFromCache(UINT64_MAX, false, firstFileNameAfterDeletingInBwd);

	videoCacheSize = cof.getCacheSize();

	BOOST_TEST(videoCacheSize == 2);

	BOOST_TEST(boost::filesystem::equivalent(firstFileNameAfterDeletingInBwd, secondFileNameBeforeDeleting));
}

BOOST_AUTO_TEST_CASE(skipTS_exact_match_with_fileName)
{
	LoggerSetup setup;
	DiskFiles diskFiles;

	std::string dir = "data/Mp4_videos/mp4_seek_tests";
	dir = boost::filesystem::canonical(dir).string();
	uint32_t cacheSize = 2;
	uint32_t lowerWaterMark = 3;
	uint32_t upperWaterMark = 3;
	std::string fileName;

	OrderedCacheOfFiles cof(dir, cacheSize, lowerWaterMark, upperWaterMark);
	uint64_t skipMsecs = 0;

	// 3rd and 4th file will come in cache
	cof.parseFiles(1655919060000 - 100, true);

	auto isFilePresent = cof.getRandomSeekFile(diskFiles.files.begin()->first, true, skipMsecs, fileName);
	BOOST_TEST(isFilePresent);
	BOOST_TEST(boost::filesystem::equivalent(fileName, diskFiles.files.begin()->second));
}

BOOST_AUTO_TEST_CASE(get_start_end)
{
	LoggerSetup setup;
	DiskFiles diskFiles;

	std::string dir = "data/Mp4_videos/mp4_seek_tests";
	dir = boost::filesystem::canonical(dir).string();
	uint32_t cacheSize = 8;
	uint32_t lowerWaterMark = 10;
	uint32_t upperWaterMark = 20;
	std::string fileName;

	OrderedCacheOfFiles cof(dir, cacheSize, lowerWaterMark, upperWaterMark);
	cof.parseFiles(0, true);

	std::string videoFile = dir + "\\20220522\\0016\\1655895288956.mp4";
	uint64_t start_ts, end_ts;
	bool isFileInCache = cof.fetchFromCache(videoFile, start_ts, end_ts);
	BOOST_TEST(isFileInCache == true);
	BOOST_TEST(start_ts == 1655895288956);
	BOOST_TEST(end_ts == 0);

	// force update from disk
	isFileInCache = cof.fetchAndUpdateFromDisk(videoFile, start_ts, end_ts);
	BOOST_TEST(isFileInCache == true);
	BOOST_TEST(start_ts == 1655895288956);
	BOOST_TEST(end_ts == 1655895298961);

	// not in cache + not on disk
	videoFile = dir + "\\20220630\\0012\\1659163533000.mp4";
	isFileInCache = cof.fetchFromCache(videoFile, start_ts, end_ts);
	BOOST_TEST(isFileInCache == false);
	BOOST_TEST(start_ts == 0);
	BOOST_TEST(end_ts == 0);
}

BOOST_AUTO_TEST_CASE(parse_noFirstRelevantFileFound_prevFile)
{
	// tests the case where random seek forces a fresh disk parse
	// and exactly 1 file is being added to cache that has start_ts < seekTS
	LoggerSetup setup;
	DiskFiles diskFiles;

	std::string dir = "data/Mp4_videos/mp4_seek_tests";
	dir = boost::filesystem::canonical(dir).string();
	uint32_t cacheSize = 3;
	uint32_t lowerWaterMark = 4;
	uint32_t upperWaterMark = 4;
	std::string fileName;

	OrderedCacheOfFiles cof(dir, cacheSize, lowerWaterMark, upperWaterMark);
	uint64_t skipMsecs = 0;

	// 1st file will come in cache
	cof.parseFiles(1655895162221 - 10, true);

	// seek to ts that is inside the last file
	auto isFilePresent = cof.getRandomSeekFile(1655926320000 + 10, true, skipMsecs, fileName);
	BOOST_TEST(isFilePresent);
	BOOST_TEST(boost::filesystem::equivalent(fileName, diskFiles.files[1655926320000]));
}

BOOST_AUTO_TEST_CASE(parse_noFirstRelevantFileFound_exactMatch)
{
	LoggerSetup setup;
	DiskFiles diskFiles;

	std::string dir = "data/Mp4_videos/mp4_seek_tests";
	dir = boost::filesystem::canonical(dir).string();
	uint32_t cacheSize = 1;
	uint32_t lowerWaterMark = 4;
	uint32_t upperWaterMark = 4;
	std::string fileName;

	OrderedCacheOfFiles cof(dir, cacheSize, lowerWaterMark, upperWaterMark);
	uint64_t skipMsecs = 0;

	// 1st file will come in cache
	cof.parseFiles(1655895162221 - 10, true);

	// seek to ts that is inside the last file
	auto isFilePresent = cof.getRandomSeekFile(1655926320000, true, skipMsecs, fileName);
	BOOST_TEST(isFilePresent);
	BOOST_TEST(boost::filesystem::equivalent(fileName, diskFiles.files[1655926320000]));
}

void printCache(std::map<std::string, std::pair<uint64_t, uint64_t> > &snap)
{
	LOG_INFO << "===printing cache===";
	for (auto it = snap.begin(); it != snap.end(); ++it)
	{
		LOG_INFO << it->first << ": <" << it->second.first << "> <" << it->second.second << ">";
	}
}

BOOST_AUTO_TEST_CASE(fwd_seek_trig_parse_hole_check)
{
	/* checks that no holes should be created in cache on fresh parse while seeking
	fwd dir */
	LoggerSetup setup;
	DiskFiles diskFiles;

	std::string dir = "data/Mp4_videos/mp4_seek_tests";
	dir = boost::filesystem::canonical(dir).string();
	uint32_t cacheSize = 1;
	uint32_t lowerWaterMark = 5;
	uint32_t upperWaterMark = 6;
	std::string fileName;

	OrderedCacheOfFiles cof(dir, cacheSize, lowerWaterMark, upperWaterMark);
	uint64_t skipMsecs = 0;

	// only 2 file in cache (first hours) even though size is 1
	cof.parseFiles(1655895162221 - 10, true);
	auto snap = cof.getSnapShot();
	printCache(snap);
	BOOST_TEST((snap.find(boost::filesystem::canonical(diskFiles.files[1655895162221]).string()) != snap.end()));
	BOOST_TEST((snap.find(boost::filesystem::canonical(diskFiles.files[1655895288956]).string()) != snap.end()));

	// seek to end
	auto isFilePresent = cof.getRandomSeekFile(1655926320000 + 5, true, skipMsecs, fileName);
	BOOST_TEST(isFilePresent);
	BOOST_TEST(boost::filesystem::equivalent(fileName, diskFiles.files[1655926320000]));

	// all four file will come in cache
	snap = cof.getSnapShot();
	printCache(snap);

	for (auto it = diskFiles.files.begin(); it != diskFiles.files.end(); ++it)
	{
		auto fileItr = snap.find(boost::filesystem::canonical(it->second).string());
		BOOST_TEST((fileItr != snap.end()));
	}	
}

BOOST_AUTO_TEST_CASE(bwd_seek_trig_parse_hole_check)
{
	LoggerSetup setup;
	DiskFiles diskFiles;

	std::string dir = "data/Mp4_videos/mp4_seek_tests";
	dir = boost::filesystem::canonical(dir).string();
	uint32_t cacheSize = 1;
	uint32_t lowerWaterMark = 5;
	uint32_t upperWaterMark = 6;
	std::string fileName;

	OrderedCacheOfFiles cof(dir, cacheSize, lowerWaterMark, upperWaterMark);
	uint64_t skipMsecs = 0;

	// only 1 file in cache
	cof.parseFiles(1655926320000 - 10, true);
	LOG_INFO << "first parse files";
	auto snap = cof.getSnapShot();
	printCache(snap);

	// seek to first in bwd dir
	auto isFilePresent = cof.getRandomSeekFile(1655895162221 + 5, false, skipMsecs, fileName);
	BOOST_TEST(isFilePresent);
	BOOST_TEST(boost::filesystem::equivalent(fileName, diskFiles.files[1655895162221]));

	// all four file will come in cache
	snap = cof.getSnapShot();
	printCache(snap);
	for (auto it = diskFiles.files.begin(); it != diskFiles.files.end(); ++it)
	{
		auto fileItr = snap.find(boost::filesystem::canonical(it->second).string());
		BOOST_TEST((fileItr != snap.end()));
	}
}

BOOST_AUTO_TEST_CASE(randomSeek_trig_drop)
{
	LoggerSetup setup;
	DiskFiles diskFiles;

	std::string dir = "data/Mp4_videos/mp4_seek_tests";
	dir = boost::filesystem::canonical(dir).string();
	uint32_t cacheSize = 1;
	uint32_t lowerWaterMark = 3;
	uint32_t upperWaterMark = 3;
	std::string fileName;

	OrderedCacheOfFiles cof(dir, cacheSize, lowerWaterMark, upperWaterMark);
	uint64_t skipMsecs = 0;

	// only 1 file in cache
	cof.parseFiles(1655926320000 - 10, true);
	auto snap = cof.getSnapShot();
	printCache(snap);

	// seek to first in bwd dir
	auto isFilePresent = cof.getRandomSeekFile(1655895162221 + 5, false, skipMsecs, fileName);
	BOOST_TEST(isFilePresent);
	BOOST_TEST(boost::filesystem::equivalent(fileName, diskFiles.files[1655895162221]));

	// all four file should come in cache - but last two will be dropped due to drop logic
	snap = cof.getSnapShot();
	printCache(snap);
	BOOST_TEST((snap.find(boost::filesystem::canonical(diskFiles.files[1655895162221]).string()) != snap.end()));
	BOOST_TEST((snap.find(boost::filesystem::canonical(diskFiles.files[1655895288956]).string()) != snap.end()));

	// seek in fwd dir to last file again
	isFilePresent = cof.getRandomSeekFile(1655926320000 + 5, true, skipMsecs, fileName);
	BOOST_TEST(isFilePresent);
	BOOST_TEST(boost::filesystem::equivalent(fileName, diskFiles.files[1655926320000]));

	// now only last two files will be present in cache
	snap = cof.getSnapShot();
	printCache(snap);
	BOOST_TEST((snap.find(boost::filesystem::canonical(diskFiles.files[1655919060000]).string()) != snap.end()));
	BOOST_TEST((snap.find(boost::filesystem::canonical(diskFiles.files[1655926320000]).string()) != snap.end()));
}

BOOST_AUTO_TEST_CASE(cache_refresh)
{
	LoggerSetup setup;
	DiskFiles diskFiles;

	std::string dir = "data/Mp4_videos/mp4_seek_tests";
	dir = boost::filesystem::canonical(dir).string();
	uint32_t cacheSize = 2;
	uint32_t lowerWaterMark = 3;
	uint32_t upperWaterMark = 6;
	std::string fileName;

	OrderedCacheOfFiles cof(dir, cacheSize, lowerWaterMark, upperWaterMark);

	// only 3 file in cache
	cof.parseFiles(1655895162221 - 10, true);

	auto snap = cof.getSnapShot();
	printCache(snap);

	BOOST_TEST((snap.find(diskFiles.files[1655926320000]) == snap.end()));

	cof.refreshCache();

	// all four file will come in cache
	snap = cof.getSnapShot();
	printCache(snap);

	for (auto it = diskFiles.files.begin(); it != diskFiles.files.end(); ++it)
	{
		auto fileItr = snap.find(boost::filesystem::canonical(it->second).string());
		BOOST_TEST((fileItr != snap.end()));
	}
}

BOOST_AUTO_TEST_SUITE_END()