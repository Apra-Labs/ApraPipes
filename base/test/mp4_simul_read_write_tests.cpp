#include <vector>
#include <string>
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include "Logger.h"

#include "Mp4ReaderSource.h"
#include "Mp4WriterSink.h"
#include "StatSink.h"
#include "EncodedImageMetadata.h"
#include "Mp4VideoMetadata.h"
#include "FileReaderModule.h"
#include "FrameContainerQueue.h"
#include "PipeLine.h"
#include "Mp4ErrorFrame.h"
BOOST_AUTO_TEST_SUITE(mp4_simul_read_write_tests)

class ExternalSinkProps : public ModuleProps
{
public:
	ExternalSinkProps() : ModuleProps()
	{
	}
};
class ExternalSink : public Module
{
public:
	ExternalSink(ExternalSinkProps props) : Module(SINK, "ExternalSink", props)
	{
	}

	frame_container pop()
	{
		return Module::pop();
	}

	boost::shared_ptr<FrameContainerQueue> getQue()
	{
		return Module::getQue();
	}

	~ExternalSink()
	{
	}

protected:
	bool process(frame_container &frames)
	{
		//LOG_ERROR << "ExternalSinkProcess <>";
		for (const auto &pair : frames)
		{
			LOG_TRACE << pair.first << "," << pair.second;
		}

		auto frame = Module::getFrameByType(frames, FrameMetadata::FrameType::ENCODED_IMAGE);
		if (frame)
			LOG_INFO << "Timestamp <" << frame->timestamp << ">";

		return true;
	}

	bool validateInputPins()
	{
		return true;
	}

	bool validateInputOutputPins()
	{
		return true;
	}

}; // ExternalSink

struct WritePipeline {
	WritePipeline(std::string readFolderPath, int readfps, int width, int height,
		std::string _writeOutPath, int writeChunkTime, int writeSyncTimeInSecs, int writefps)
	{
		LoggerProps loggerProps;
		loggerProps.logLevel = boost::log::trivial::severity_level::info;
		Logger::setLogLevel(boost::log::trivial::severity_level::info);
		Logger::initLogger(loggerProps);

		writeOutPath = _writeOutPath;
		auto fileReaderProps = FileReaderModuleProps(readFolderPath, 0, -1);
		fileReaderProps.fps = readfps;
		fileReaderProps.readLoop = true;
		fileReader = boost::shared_ptr<Module>(new FileReaderModule(fileReaderProps));
		auto encodedImageMetadata = framemetadata_sp(new EncodedImageMetadata(width, height));
		fileReader->addOutputPin(encodedImageMetadata);

		auto mp4WriterSinkProps = Mp4WriterSinkProps(writeChunkTime, writeSyncTimeInSecs, writefps, _writeOutPath);
		mp4WriterSink = boost::shared_ptr<Mp4WriterSink>(new Mp4WriterSink(mp4WriterSinkProps));
		fileReader->setNext(mp4WriterSink);

		BOOST_TEST(fileReader->init());
		BOOST_TEST(mp4WriterSink->init());
	}

	~WritePipeline()
	{
		//termPipeline();
		// delete any existing stuff in directory
		if (!boost::filesystem::is_empty(writeOutPath))
		{
			for (auto &&itr : boost::filesystem::recursive_directory_iterator(writeOutPath))
			{
				auto dirPath = itr.path();
				boost::filesystem::remove_all(dirPath);
			}
		}
	}

	void termPipeline()
	{
		fileReader->term();
		mp4WriterSink->term();
	}

	std::string writeOutPath;
	boost::shared_ptr<Mp4WriterSink> mp4WriterSink = nullptr;
	boost::shared_ptr<Module> fileReader = nullptr;
};

struct WritePipelineIndependent {
	WritePipelineIndependent(std::string readFolderPath, int readfps, int width, int height,
		std::string _writeOutPath, int writeChunkTime, int writeSyncTimeInSecs, int writefps)
	{
		LoggerProps loggerProps;
		loggerProps.logLevel = boost::log::trivial::severity_level::info;
		Logger::setLogLevel(boost::log::trivial::severity_level::info);
		Logger::initLogger(loggerProps);

		writeOutPath = _writeOutPath;
		auto fileReaderProps = FileReaderModuleProps(readFolderPath, 0, -1);
		fileReaderProps.fps = readfps;
		fileReaderProps.readLoop = true;
		fileReader = boost::shared_ptr<Module>(new FileReaderModule(fileReaderProps));
		auto encodedImageMetadata = framemetadata_sp(new EncodedImageMetadata(width, height));
		fileReader->addOutputPin(encodedImageMetadata);

		auto mp4WriterSinkProps = Mp4WriterSinkProps(writeChunkTime, writeSyncTimeInSecs, writefps, _writeOutPath);
		mp4WriterSink = boost::shared_ptr<Mp4WriterSink>(new Mp4WriterSink(mp4WriterSinkProps));
		fileReader->setNext(mp4WriterSink);

		p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
		p->appendModule(fileReader);

		p->init();
		p->run_all_threaded();
	}

	~WritePipelineIndependent()
	{
		p->stop();
		p->term();
		p->wait_for_all();
		p.reset();

		// delete any existing stuff in directory
		if (!boost::filesystem::is_empty(writeOutPath))
		{
			for (auto &&itr : boost::filesystem::recursive_directory_iterator(writeOutPath))
			{
				auto dirPath = itr.path();
				boost::filesystem::remove_all(dirPath);
				break;
			}
		}
	}

	void termPipeline()
	{
		fileReader->term();
		mp4WriterSink->term();
	}

	std::string writeOutPath;
	boost::shared_ptr<PipeLine> p;
	boost::shared_ptr<Mp4WriterSink> mp4WriterSink = nullptr;
	boost::shared_ptr<Module> fileReader = nullptr;
};

struct ReadPipeline {
	ReadPipeline(std::string videoPath, uint16_t reInitInterval, bool direction, bool parseFS, int fps = 30)
	{
		LoggerProps loggerProps;
		loggerProps.logLevel = boost::log::trivial::severity_level::info;
		Logger::setLogLevel(boost::log::trivial::severity_level::info);
		Logger::initLogger(loggerProps);

		bool readLoop = false;
		auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, parseFS, reInitInterval, direction, readLoop, false);
		mp4ReaderProps.fps = fps;
		mp4ReaderProps.logHealth = true;
		mp4ReaderProps.logHealthFrequency = 100;
		mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
		auto encodedImageMetadata = framemetadata_sp(new EncodedImageMetadata(0,0));
		mp4Reader->addOutPutPin(encodedImageMetadata);
		auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1_0"));
		mp4Reader->addOutPutPin(mp4Metadata);

		std::vector<std::string> mImagePin;
		mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::ENCODED_IMAGE);

		auto sinkProps = ExternalSinkProps();;
		sink = boost::shared_ptr<ExternalSink>(new ExternalSink(sinkProps));
		mp4Reader->setNext(sink, mImagePin);

		BOOST_TEST(mp4Reader->init());
		BOOST_TEST(sink->init());
	}

	~ReadPipeline()
	{
		termPipeline();
	}
	 
	void termPipeline()
	{
		mp4Reader->term();
		sink->term();
	}

	boost::shared_ptr<Mp4ReaderSource> mp4Reader = nullptr;
	boost::shared_ptr<ExternalSink> sink = nullptr;
};

BOOST_AUTO_TEST_CASE(basic)
{
	/* write pipeline params */
	std::string readFolderPath = "data/resized_mono_jpg/";
	int fileReaderFPS = 10;
	int height = 160;
	int width = 80;
	std::string writeFolderPath = "data/Mp4_videos/mp4_read_write_tests/";
	int chunkTimeMins = UINT32_MAX;
	int syncTimeInSecs = 1;
	int writeFPS = 10;
	WritePipeline w(readFolderPath, fileReaderFPS, width, height, writeFolderPath, chunkTimeMins, syncTimeInSecs, writeFPS);

	uint64_t lastFrameTS = 0;
	// write 4 frames
	for (auto i = 0; i < 3; ++i)
	{
		w.fileReader->step();
		w.mp4WriterSink->step();
	}
	// sync the mp4 with next step
	boost::this_thread::sleep_for(boost::chrono::seconds(syncTimeInSecs));
	LOG_INFO << "WRITING 4th FRAME";
	w.fileReader->step();
	w.mp4WriterSink->step();
	LOG_INFO << "FIFTH FRAME WRITTEN";

	/* read Pipeline params */
	std::string readPath, rootPath;
	int reInitIntervalSecs = 5;
	bool direction = true;
	bool parseFS = true;

	// read the first and only file in the directory
	if (!boost::filesystem::is_directory(writeFolderPath))
	{
		boost::filesystem::create_directories(writeFolderPath);
	}
	for (auto &&itr : boost::filesystem::recursive_directory_iterator(writeFolderPath))
	{
		auto dirPath = itr.path();
		if (boost::filesystem::is_regular_file(dirPath) && dirPath.extension() == ".mp4")
		{
			readPath = dirPath.string();
			rootPath = boost::filesystem::path(readPath).parent_path().parent_path().string();
			break;
		}
	}
	auto boostVideoTS = std::stoull(boost::filesystem::path(readPath).stem().string());
	ReadPipeline r(readPath, reInitIntervalSecs, direction, parseFS);

	// read 3 frames
	for (auto i = 0; i < 3; ++i)
	{
		r.mp4Reader->step();
		auto frame = r.sink->pop().begin()->second;
		LOG_INFO << "reading frame < " << i + 1 << ">";
		LOG_INFO << "frame->timestamp <" << frame->timestamp << ">";
		BOOST_TEST(!frame->isEOS());
		if (lastFrameTS)
			BOOST_TEST((frame->timestamp - lastFrameTS) < 20000);
		lastFrameTS = frame->timestamp;
		if (!i)
		{
			BOOST_TEST(boostVideoTS == lastFrameTS);
		}
	}

	// EOS
	r.mp4Reader->step();
	auto frame = r.sink->pop().begin()->second;
	LOG_INFO << "frame is EOS <" << frame->isEOS() << ">";
	BOOST_TEST(frame->isEOS());
	//lastFrameTS = frame->timestamp;

	// force sync with new frame
	boost::this_thread::sleep_for(boost::chrono::seconds(syncTimeInSecs));
	LOG_INFO << "WRITING 5th FRAME";
	w.fileReader->step();
	w.mp4WriterSink->step();

	// reader should be able to get a frame now
	LOG_INFO << "attempt reading frame after reInitInterval";
	auto sinkQ = r.sink->getQue();
	while (1)
	{
		r.mp4Reader->step();
		if (sinkQ->size())
		{
			frame = r.sink->pop().begin()->second;
			BOOST_TEST(!frame->isEOS());
			BOOST_TEST(frame->timestamp > lastFrameTS);
			break;
		}
		boost::this_thread::sleep_for(boost::chrono::seconds(syncTimeInSecs));
	}
	LOG_INFO << "frame after reInitInterval < " << frame->timestamp << ">";

	// test cleanup
	w.termPipeline();
	r.termPipeline();
	boost::filesystem::remove_all(rootPath);
	boost::this_thread::sleep_for(boost::chrono::seconds(1));
}

BOOST_AUTO_TEST_CASE(basic_parseFS_disabled, *boost::unit_test::disabled())
{
	/* write pipeline params */
	std::string readFolderPath = "data/resized_mono_jpg/";
	int fileReaderFPS = 10;
	int height = 160;
	int width = 80;
	std::string writeFolderPath = "data/Mp4_videos/mp4_read_write_tests";
	int chunkTimeMins = UINT32_MAX;
	int syncTimeInSecs = 1;
	int writeFPS = 10;
	WritePipeline w(readFolderPath, fileReaderFPS, width, height, writeFolderPath, chunkTimeMins, syncTimeInSecs, writeFPS);

	uint64_t lastFrameTS = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	// write 4 frames
	for (auto i = 0; i < 3; ++i)
	{
		w.fileReader->step();
		w.mp4WriterSink->step();
	}
	// sync the mp4 with next step
	boost::this_thread::sleep_for(boost::chrono::seconds(syncTimeInSecs));
	LOG_INFO << "WRITING 4th FRAME";
	w.fileReader->step();
	w.mp4WriterSink->step();
	LOG_INFO << "FIFTH FRAME WRITTEN";

	/* read Pipeline params */
	std::string readPath, rootPath;
	int reInitIntervalSecs = 5;
	bool direction = true;
	bool parseFS = false;

	// read the first and only file in the directory
	if (!boost::filesystem::is_directory(writeFolderPath))
	{
		boost::filesystem::create_directories(writeFolderPath);
	}
	for (auto &&itr : boost::filesystem::recursive_directory_iterator(writeFolderPath))
	{
		auto dirPath = itr.path();
		if (boost::filesystem::is_regular_file(dirPath) && dirPath.extension() == ".mp4")
		{
			readPath = dirPath.string();
			rootPath = boost::filesystem::path(readPath).parent_path().parent_path().string();
			break;
		}
	}
	auto boostVideoTS = std::stoull(boost::filesystem::path(readPath).stem().string());
	ReadPipeline r(readPath, reInitIntervalSecs, direction, parseFS);

	// read 4 frames
	for (auto i = 0; i < 3; ++i)
	{
		r.mp4Reader->step();
		auto frame = r.sink->pop().begin()->second;
		LOG_INFO << "reading frame < " << i + 1 << ">";
		LOG_INFO << "frame->timestamp <" << frame->timestamp << ">";
		BOOST_TEST(!frame->isEOS());
		BOOST_TEST((frame->timestamp - lastFrameTS) < 20000);
		lastFrameTS = frame->timestamp;
		if (!i)
		{
			BOOST_TEST(boostVideoTS == lastFrameTS);
		}
	}

	// EOS
	r.mp4Reader->step();
	auto frame = r.sink->pop().begin()->second;
	LOG_INFO << "frame is EOS <" << frame->isEOS() << ">";
	BOOST_TEST(frame->isEOS());
	//lastFrameTS = frame->timestamp;

	// force sync with new frame
	boost::this_thread::sleep_for(boost::chrono::seconds(syncTimeInSecs));
	LOG_INFO << "WRITING 5th FRAME";
	w.fileReader->step();
	w.mp4WriterSink->step();

	// reader should be able to get a frame now
	LOG_INFO << "attempt reading frame after reInitInterval";
	auto sinkQ = r.sink->getQue();
	while (1)
	{
		r.mp4Reader->step();
		if (sinkQ->size())
		{
			frame = r.sink->pop().begin()->second;
			BOOST_TEST(!frame->isEOS());
			BOOST_TEST(frame->timestamp > lastFrameTS);
			break;
		}
		boost::this_thread::sleep_for(boost::chrono::seconds(syncTimeInSecs));
	}
	LOG_INFO << "frame after reInitInterval < " << frame->timestamp << ">";

	// test cleanup
	w.termPipeline();
	r.termPipeline();
	boost::filesystem::remove_all(rootPath);
	boost::this_thread::sleep_for(boost::chrono::seconds(2));
}

BOOST_AUTO_TEST_CASE(loop_no_chunking, *boost::unit_test::disabled())
{
	/* write pipeline params */
	std::string readFolderPath = "data/resized_mono_jpg/";
	int fileReaderFPS = 30;
	int height = 180;
	int width = 80;
	std::string writeFolderPath = "data/mp4_videos/mp4_read_write_tests/";
	int chunkTimeMins = UINT32_MAX;
	int syncTimeInSecs = 1;
	int writeFPS = 30;
	WritePipeline w(readFolderPath, fileReaderFPS, width, height, writeFolderPath, chunkTimeMins, syncTimeInSecs, writeFPS);

	uint64_t lastFrameTS = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	// write 4 frames
	for (auto i = 0; i < 3; ++i)
	{
		w.fileReader->step();
		w.mp4WriterSink->step();
	}
	// sync the mp4 with next step
	boost::this_thread::sleep_for(boost::chrono::seconds(syncTimeInSecs));
	LOG_INFO << "WRITING 4th FRAME";
	w.fileReader->step();
	w.mp4WriterSink->step();
	LOG_INFO << "FIFTH FRAME WRITTEN";

	/* read Pipeline params */
	std::string readPath, rootPath;
	int reInitIntervalSecs = 5;
	bool direction = true;
	bool parseFS = true;

	// read the first and only file in the directory
	if (!boost::filesystem::is_directory(writeFolderPath))
	{
		boost::filesystem::create_directories(writeFolderPath);
	}
	for (auto &&itr : boost::filesystem::recursive_directory_iterator(writeFolderPath))
	{
		auto dirPath = itr.path();
		if (boost::filesystem::is_regular_file(dirPath) && dirPath.extension() == ".mp4")
		{
			readPath = dirPath.string();
			rootPath = boost::filesystem::path(readPath).parent_path().parent_path().string();
			break;
		}
	}
	auto boostVideoTS = std::stoull(boost::filesystem::path(readPath).stem().string());
	ReadPipeline r(readPath, reInitIntervalSecs, direction, parseFS);

	// read 4 frames
	for (auto i = 0; i < 3; ++i)
	{
		r.mp4Reader->step();
		auto frame = r.sink->pop().begin()->second;
		LOG_INFO << "reading frame < " << i + 1 << ">";
		LOG_INFO << "frame->timestamp <" << frame->timestamp << ">";
		BOOST_TEST(!frame->isEOS());
		BOOST_TEST((frame->timestamp - lastFrameTS) < 20000);
		lastFrameTS = frame->timestamp;
		if (!i)
		{
			BOOST_TEST(boostVideoTS == lastFrameTS);
		}
	}

	// EOS
	r.mp4Reader->step();
	auto frame = r.sink->pop().begin()->second;
	LOG_INFO << "frame is EOS <" << frame->isEOS() << ">";
	BOOST_TEST(frame->isEOS());
	//lastFrameTS = frame->timestamp;

	LOG_INFO << "=============================Starting loop testing==========================";
	int count = 0;
	auto sinkQ = r.sink->getQue();
	while (count < 10)
	{
		LOG_INFO << "=======Round " << count + 1 << "==========";
		for (auto i = 0; i < 10; i++)
		{
			if (i == 9) // force sync all 10 frames in next step
			{
				boost::this_thread::sleep_for(boost::chrono::seconds(syncTimeInSecs));
			}
			LOG_INFO << "===>Writing new frames after EOS <" << i+1 << ">";
			w.fileReader->step();
			w.mp4WriterSink->step();
		}

		LOG_INFO << "====10 frames written on the file===";
		auto frameCount = 0;
		while (1)
		{
			r.mp4Reader->step();
			if (!sinkQ->size())
			{
				boost::this_thread::sleep_for(boost::chrono::seconds(syncTimeInSecs));
				continue;
			}
			frame = r.sink->pop().begin()->second;
			BOOST_TEST(frame->timestamp > lastFrameTS);
			LOG_INFO << "===>reading frame after EOS <" << frame->timestamp << "> isEOS <" << frame->isEOS() << ">";
			++frameCount;
			if (frame->isEOS())
			{
				--frameCount;
				break;
			}
			else
			{
				lastFrameTS = frame->timestamp; // we care about ts of image frame
			}
		}
		BOOST_TEST(frameCount == 10);
		++count;
	}
	// test cleanup
	w.termPipeline();
	r.termPipeline();
	boost::filesystem::remove_all(rootPath);
	boost::this_thread::sleep_for(boost::chrono::seconds(2));
}

BOOST_AUTO_TEST_CASE(basic_chunking, *boost::unit_test::disabled())
{
	/* write pipeline params */
	std::string readFolderPath = "data/resized_mono_jpg/";
	int fileReaderFPS = 10;
	int height = 160;
	int width = 80;
	std::string writeFolderPath = "data/mp4_videos/mp4_read_write_tests/";
	int chunkTimeMins = 1;
	int syncTimeInSecs = 1;
	int writeFPS = 10;
	WritePipeline w(readFolderPath, fileReaderFPS, width, height, writeFolderPath, chunkTimeMins, syncTimeInSecs, writeFPS);

	/* timing */
	auto nowTime = std::chrono::system_clock::now().time_since_epoch();
	uint64_t lastFrameTS = std::chrono::duration_cast<std::chrono::milliseconds>(nowTime).count();
	std::chrono::time_point<std::chrono::system_clock> timePointInSeconds(std::chrono::duration_cast<std::chrono::milliseconds>(nowTime));
	std::time_t t = std::chrono::system_clock::to_time_t(timePointInSeconds);
	std::tm tm = *std::localtime(&t);

	// write 4 frames
	for (auto i = 0; i < 3; ++i)
	{
		w.fileReader->step();
		w.mp4WriterSink->step();
	}
	// sync the mp4 with next step
	boost::this_thread::sleep_for(boost::chrono::seconds(syncTimeInSecs));
	w.fileReader->step();
	w.mp4WriterSink->step();

	/* read Pipeline params */
	std::string readPath, rootPath;
	int reInitIntervalSecs = 5;
	bool direction = true;
	bool parseFS = true;

	// read the first and only file in the directory
	if (!boost::filesystem::is_directory(writeFolderPath))
	{
		boost::filesystem::create_directories(writeFolderPath);
	}
	for (auto &&itr : boost::filesystem::recursive_directory_iterator(writeFolderPath))
	{
		auto dirPath = itr.path();
		if (boost::filesystem::is_regular_file(dirPath) && dirPath.extension() == ".mp4")
		{
			readPath = dirPath.string();
			rootPath = boost::filesystem::path(readPath).parent_path().parent_path().string();
			break;
		}
	}
	auto boostVideoTS = std::stoull(boost::filesystem::path(readPath).stem().string());
	ReadPipeline r(readPath, reInitIntervalSecs, direction, parseFS);

	// read 4 frames
	for (auto i = 0; i < 3; ++i)
	{
		r.mp4Reader->step();
		auto frame = r.sink->pop().begin()->second;
		LOG_INFO << "reading frame < " << i + 1 << ">";
		LOG_INFO << "frame->timestamp <" << frame->timestamp << ">";
		BOOST_TEST(!frame->isEOS());
		BOOST_TEST((frame->timestamp - lastFrameTS) < 20000);
		lastFrameTS = frame->timestamp;
		if (!i)
		{
			BOOST_TEST(boostVideoTS == lastFrameTS);
		}
	}

	// EOS
	r.mp4Reader->step();
	auto frame = r.sink->pop().begin()->second;
	LOG_INFO << "frame is EOS <" << frame->isEOS() << ">";
	BOOST_TEST(frame->isEOS());
	//lastFrameTS = frame->timestamp;

	/* write a new video now - dirty minute clock change logic start */
	auto nowTime2 = std::chrono::system_clock::now().time_since_epoch();
	std::chrono::time_point <std::chrono::system_clock> timePointInSeconds2(std::chrono::duration_cast<std::chrono::milliseconds>(nowTime2));
	std::time_t t2 = std::chrono::system_clock::to_time_t(timePointInSeconds2);
	std::tm tm2 = *std::localtime(&t2);

	while (tm2.tm_min != tm.tm_min + 1)
	{
		auto nowTime2 = std::chrono::system_clock::now().time_since_epoch();
		std::chrono::time_point <std::chrono::system_clock> timePointInSeconds2(std::chrono::duration_cast<std::chrono::milliseconds>(nowTime2));
		std::time_t t2 = std::chrono::system_clock::to_time_t(timePointInSeconds2);
		tm2 = *std::localtime(&t2);
	}
	/* dirty minute clock change logic end */
	LOG_INFO << "OldVideoMin < " << tm.tm_min << "> newVideoMin <" << tm2.tm_min << ">";
	auto lowerLimitNewVideo = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	LOG_INFO << "Writing 2 frames in new video!! @ <" << lowerLimitNewVideo << ">";
	for (auto i = 0; i < 2; ++i)
	{
		if (i == 1)
		{
			boost::this_thread::sleep_for(boost::chrono::seconds(syncTimeInSecs));
		}
		w.fileReader->step();
		w.mp4WriterSink->step();
	}

	// reader should get the frame from the new video now
	LOG_INFO << "=================================read from new video===================================";
	auto sinkQ = r.sink->getQue();
	auto count = 0;
	while (1)
	{
		r.mp4Reader->step();
		LOG_INFO << "sinkQ size <" << sinkQ->size() << ">";
		if (sinkQ->size())
		{
			frame = r.sink->pop().begin()->second;
			LOG_INFO << "===>reading frame after EOS <" << frame->timestamp << "> lowerLimitNewVideo <" << lowerLimitNewVideo << "> isEOS<" << frame->isEOS() << "> ";
			if (frame->isEOS())
			{
				LOG_INFO << "====EndOfStream====";
				break;
			}
			BOOST_TEST(!frame->isEOS());
			BOOST_TEST(frame->timestamp >= lowerLimitNewVideo);
		}
		boost::this_thread::sleep_for(boost::chrono::seconds(syncTimeInSecs));
	}

	// now write and read for 10 frames
	for (auto i = 0; i < 10; i++)
	{
		if (i == 9) // force sync all 10 frames in next step
		{
			boost::this_thread::sleep_for(boost::chrono::seconds(syncTimeInSecs));
		}
		LOG_INFO << "===>Writing new frames after EOS in new file <" << i + 1 << ">";
		w.fileReader->step();
		w.mp4WriterSink->step();
	}

	LOG_INFO << "====10 frames written on the new file===";
	auto frameCount = 0;
	while (1)
	{
		r.mp4Reader->step();
		if (!sinkQ->size())
		{
			boost::this_thread::sleep_for(boost::chrono::seconds(syncTimeInSecs));
			continue;
		}
		frame = r.sink->pop().begin()->second;
		BOOST_TEST(frame->timestamp > lastFrameTS);
		LOG_INFO << "===>reading frame after EOS in the new file<" << frame->timestamp << "> isEOS <" << frame->isEOS() << ">";
		++frameCount;
		if (frame->isEOS())
		{
			--frameCount;
			break;
		}
		else
		{
			lastFrameTS = frame->timestamp;
		}
	}
	LOG_INFO << "total Frames read in the new file after EOS<" << frameCount << ">";
	BOOST_TEST(frameCount == 10);
	boost::this_thread::sleep_for(boost::chrono::seconds(syncTimeInSecs+2));

	// test cleanup
	w.termPipeline();
	r.termPipeline();
	boost::filesystem::remove_all(rootPath);
	boost::this_thread::sleep_for(boost::chrono::seconds(2));
}

BOOST_AUTO_TEST_CASE(seek_in_wait_state)
{
	/* write pipeline params */
	std::string readFolderPath = "data/resized_mono_jpg/";
	int fileReaderFPS = 10;
	int height = 30;
	int width = 22;
	std::string writeFolderPath = "data/mp4_videos/mp4_read_write_tests/";
	int chunkTimeMins = UINT32_MAX;
	int syncTimeInSecs = 1;
	int writeFPS = 10;
	WritePipeline w(readFolderPath, fileReaderFPS, width, height, writeFolderPath, chunkTimeMins, syncTimeInSecs, writeFPS);

	uint64_t lastFrameTS = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	// write 4 frames
	for (auto i = 0; i < 3; ++i)
	{
		w.fileReader->step();
		w.mp4WriterSink->step();
	}
	// sync the mp4 with next step
	boost::this_thread::sleep_for(boost::chrono::seconds(syncTimeInSecs));
	LOG_INFO << "WRITING 4th FRAME";
	w.fileReader->step();
	w.mp4WriterSink->step();
	LOG_INFO << "FIFTH FRAME WRITTEN";

	/* read Pipeline params */
	std::string readPath, rootPath;
	int reInitIntervalSecs = 5;
	bool direction = true;
	bool parseFS = true;

	// read the first and only file in the directory
	if (!boost::filesystem::is_directory(writeFolderPath))
	{
		boost::filesystem::create_directories(writeFolderPath);
	}
	for (auto &&itr : boost::filesystem::recursive_directory_iterator(writeFolderPath))
	{
		auto dirPath = itr.path();
		if (boost::filesystem::is_regular_file(dirPath) && dirPath.extension() == ".mp4")
		{
			readPath = dirPath.string();
			rootPath = boost::filesystem::path(readPath).parent_path().parent_path().string();
			break;
		}
	}
	auto boostVideoTS = std::stoull(boost::filesystem::path(readPath).stem().string());
	ReadPipeline r(readPath, reInitIntervalSecs, direction, parseFS);

	// read 4 frames
	for (auto i = 0; i < 3; ++i)
	{
		r.mp4Reader->step();
		auto frame = r.sink->pop().begin()->second;
		LOG_INFO << "reading frame < " << i + 1 << ">";
		LOG_INFO << "frame->timestamp <" << frame->timestamp << "> lastFrameTS <" << lastFrameTS << ">";
		BOOST_TEST(!frame->isEOS());
		BOOST_TEST((frame->timestamp - lastFrameTS) < 20000);
		lastFrameTS = frame->timestamp;
		if (!i)
		{
			BOOST_TEST(boostVideoTS == lastFrameTS);
		}
	}

	// EOS
	r.mp4Reader->step();
	auto frame = r.sink->pop().begin()->second;
	LOG_INFO << "frame is EOS <" << frame->isEOS() << ">";
	BOOST_TEST(frame->isEOS());
	lastFrameTS = frame->timestamp;

	// wait state
	LOG_INFO << "reader should be in waiting state";
	auto sinkQ = r.sink->getQue();
	for (auto i = 0; i < 2; ++i)
	{
		r.mp4Reader->step();
		if (sinkQ->size())
		{
			BOOST_TEST(false);
			LOG_ERROR << "mp4Reader should be in waiting state.";
			break;
		}
		boost::this_thread::sleep_for(boost::chrono::seconds(syncTimeInSecs));
	}

	// seek in wait state
	LOG_INFO << "seeking in wait state...";
	r.mp4Reader->randomSeek(boostVideoTS);
	r.mp4Reader->step();
	BOOST_TEST(sinkQ->size());
	frame = r.sink->pop().begin()->second;
	BOOST_TEST(frame->timestamp == boostVideoTS);
	LOG_INFO << "seeked frameTS <" << frame->timestamp << "> videoTS <" << boostVideoTS << ">";

	// test cleanup
	w.termPipeline();
	r.termPipeline();
	boost::filesystem::remove_all(rootPath);
	boost::this_thread::sleep_for(boost::chrono::seconds(2));
}

BOOST_AUTO_TEST_CASE(seek_in_wait_parseFS_disabled)
{
	/* write pipeline params */
	std::string readFolderPath = "data/resized_mono_jpg/";
	int fileReaderFPS = 10;
	int height = 30;
	int width = 22;
	std::string writeFolderPath = "data/mp4_videos/mp4_read_write_tests/";
	int chunkTimeMins = UINT32_MAX;
	int syncTimeInSecs = 1;
	int writeFPS = 10;
	WritePipeline w(readFolderPath, fileReaderFPS, width, height, writeFolderPath, chunkTimeMins, syncTimeInSecs, writeFPS);

	uint64_t lastFrameTS = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	// write 4 frames
	for (auto i = 0; i < 3; ++i)
	{
		w.fileReader->step();
		w.mp4WriterSink->step();
	}
	// sync the mp4 with next step
	boost::this_thread::sleep_for(boost::chrono::seconds(syncTimeInSecs));
	LOG_INFO << "WRITING 4th FRAME";
	w.fileReader->step();
	w.mp4WriterSink->step();
	LOG_INFO << "FOURTH FRAME WRITTEN";

	/* read Pipeline params */
	std::string readPath, rootPath;
	int reInitIntervalSecs = 5;
	bool direction = true;
	bool parseFS = false;

	// read the first and only file in the directory
	if (!boost::filesystem::is_directory(writeFolderPath))
	{
		boost::filesystem::create_directories(writeFolderPath);
	}
	for (auto &&itr : boost::filesystem::recursive_directory_iterator(writeFolderPath))
	{
		auto dirPath = itr.path();
		if (boost::filesystem::is_regular_file(dirPath) && dirPath.extension() == ".mp4")
		{
			readPath = dirPath.string();
			rootPath = boost::filesystem::path(readPath).parent_path().parent_path().string();
			break;
		}
	}
	auto boostVideoTS = std::stoull(boost::filesystem::path(readPath).stem().string());
	ReadPipeline r(readPath, reInitIntervalSecs, direction, parseFS);
	uint64_t lastVideoTS = 0;
	// read 4 frames
	for (auto i = 0; i < 3; ++i)
	{
		r.mp4Reader->step();
		auto frame = r.sink->pop().begin()->second;
		LOG_INFO << "reading frame < " << i + 1 << ">";
		LOG_INFO << "frame->timestamp <" << frame->timestamp << "> lastFrameTS <" << lastFrameTS << ">";
		BOOST_TEST(!frame->isEOS());
		BOOST_TEST((frame->timestamp - lastFrameTS) < 20000);
		lastFrameTS = frame->timestamp;
		lastVideoTS = lastFrameTS;
		if (!i)
		{
			BOOST_TEST(boostVideoTS == lastFrameTS);
		}
	}

	// EOS
	r.mp4Reader->step();
	auto frame = r.sink->pop().begin()->second;
	LOG_INFO << "frame is EOS <" << frame->isEOS() << ">";
	BOOST_TEST(frame->isEOS());
	lastFrameTS = frame->timestamp;

	// wait state
	LOG_ERROR << "reader should be in waiting state";
	auto sinkQ = r.sink->getQue();
	for (auto i = 0; i < 2; ++i)
	{
		r.mp4Reader->step();
		if (sinkQ->size())
		{
			BOOST_TEST(false);
			LOG_ERROR << "mp4Reader should be in waiting state.";
			break;
		}
		boost::this_thread::sleep_for(boost::chrono::seconds(syncTimeInSecs));
	}

	// seek in wait state and read whole video again
	LOG_INFO << "seeking in wait state...";
	r.mp4Reader->randomSeek(boostVideoTS);
	r.mp4Reader->step();
	BOOST_TEST(sinkQ->size());
	frame = r.sink->pop().begin()->second;
	BOOST_TEST(frame->timestamp == boostVideoTS);
	LOG_INFO << "seeked frameTS <" << frame->timestamp << "> videoTS <" << boostVideoTS << ">";

	for (auto i = 0; i < 2; ++i)
	{
		r.mp4Reader->step();
		auto frame = r.sink->pop().begin()->second;
		LOG_INFO << "reading frame < " << i + 1 << ">";
		LOG_INFO << "frame->timestamp <" << frame->timestamp << "> lastFrameTS <" << lastFrameTS << ">";
		lastFrameTS = frame->timestamp;
	}

	// EOS
	r.mp4Reader->step();
	frame = r.sink->pop().begin()->second;
	LOG_INFO << "frame is EOS <" << frame->isEOS() << ">";
	BOOST_TEST(frame->isEOS());
	// no updates to the video
	BOOST_TEST(lastFrameTS == lastVideoTS);

	// test cleanup
	w.termPipeline();
	r.termPipeline();
	boost::filesystem::remove_all(rootPath);
}

BOOST_AUTO_TEST_CASE(writer_only, *boost::unit_test::disabled())
{
	/* write pipeline params */
	std::string readFolderPath = "data/bigjpeg/";
	int fileReaderFPS = 180;
	int height = 3619;
	int width = 3619;
	std::string writeFolderPath = "data/mp4_videos/mp4_read_write_tests/";
	int chunkTimeMins = 10;
	int syncTimeInSecs = 1;
	int writeFPS = 180;
	WritePipelineIndependent write(readFolderPath, fileReaderFPS, width, height, writeFolderPath, chunkTimeMins, syncTimeInSecs, writeFPS);
}

BOOST_AUTO_TEST_CASE(reader_only, *boost::unit_test::disabled())
{
	std::string writeFolderPath = "";
	std::string readPath = "", rootPath;
	int reInitIntervalSecs = 1;
	bool direction = true;
	bool parseFS = true;
	int readFps = 60;

	while (readPath.empty())
	{
		// read the first and only file in the directory
		if (!boost::filesystem::is_directory(writeFolderPath))
		{
			boost::filesystem::create_directories(writeFolderPath);
		}
		for (auto &&itr : boost::filesystem::recursive_directory_iterator(writeFolderPath))
		{
			auto dirPath = itr.path();
			if (boost::filesystem::is_regular_file(dirPath) && dirPath.extension() == ".mp4")
			{
				readPath = dirPath.string();
				rootPath = boost::filesystem::path(readPath).parent_path().parent_path().string();
				break;
			}
		}
	}
	LOG_INFO << "Waiting for first video";
	boost::this_thread::sleep_for(boost::chrono::seconds(1));
	LOG_INFO << "Resuming.....";
	auto boostVideoTS = std::stoull(boost::filesystem::path(readPath).stem().string());
	ReadPipeline r(readPath, reInitIntervalSecs, direction, parseFS, readFps);

	// read 4 frames
	auto i = 0;
	uint64_t lastFrameTS = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	bool skipStep = false;
	while (1)
	{
		r.mp4Reader->step();
		auto frame = r.sink->pop().begin()->second;
		if (frame->isMp4ErrorFrame())
		{
			auto errorFrame = dynamic_cast<Mp4ErrorFrame*>(frame.get());
			auto type = errorFrame->errorCode;
			LOG_ERROR << "**********************************************************************************";
			LOG_ERROR << "Error occured in mp4Reader <" << type << "> msg <" << errorFrame->errorMsg << ">";
			LOG_ERROR << "**********************************************************************************";
			if (type == MP4_MISSING_VIDEOTRACK)
			{
				boost::this_thread::sleep_for(boost::chrono::seconds(1));
				r.mp4Reader->randomSeek(lastFrameTS + 1);
				continue;
			}
			else if (type == MP4_OPEN_FILE_FAILED)
			{
				boost::this_thread::sleep_for(boost::chrono::seconds(1));
				r.mp4Reader->randomSeek(lastFrameTS + 1);
				continue;
			}
		}
		if (frame->isEOS())
		{
			auto q = r.sink->getQue();
			while (!q->size())
			{
				LOG_INFO << "Waiting for more data on disk .....................";
				boost::this_thread::sleep_for(boost::chrono::seconds(1));
				r.mp4Reader->step();
				skipStep = true;
			}
			continue;
		}
		LOG_INFO << "read image frame < " << i + 1 << ">";
		LOG_INFO << "frame->timestamp <" << frame->timestamp << ">";
		//BOOST_TEST((frame->timestamp - lastFrameTS) < 20000);
		lastFrameTS = frame->timestamp;
		++i;

		if (i == 10000)
		{
			break;
		}
	}
	LOG_INFO << "Total Frames Read <" << i << ">";
	boost::this_thread::sleep_for(boost::chrono::milliseconds(300));
}

// most important
BOOST_AUTO_TEST_CASE(ultimate)
{
	/* write pipeline params */
	std::string readFolderPath = "data/resized_mono_jpg/";
	int fileReaderFPS = 30;
	int height = 80;
	int width = 160;
	std::string writeFolderPath = "data/Mp4_videos/mp4_read_write_tests/";
	int chunkTimeMins = 10;
	int syncTimeInSecs = 1;
	int writeFPS = 30;
	WritePipelineIndependent write(readFolderPath, fileReaderFPS, width, height, writeFolderPath, chunkTimeMins, syncTimeInSecs, writeFPS);

	std::string readPath = "", rootPath;
	int reInitIntervalSecs = 1;
	bool direction = true;
	bool parseFS = true;
	int readFps = 60;

	while (readPath.empty())
	{
		// read the first and only file in the directory
		if (!boost::filesystem::is_directory(writeFolderPath))
		{
			boost::filesystem::create_directories(writeFolderPath);
		}
		auto cannonicalWriteFolderPath = boost::filesystem::canonical(writeFolderPath);
		for (auto &&itr : boost::filesystem::recursive_directory_iterator(cannonicalWriteFolderPath))
		{
			auto dirPath = itr.path();
			if (boost::filesystem::is_regular_file(dirPath) && dirPath.extension() == ".mp4")
			{
				readPath = dirPath.string();
				rootPath = boost::filesystem::path(readPath).parent_path().parent_path().string();
				break;
			}
		}
	}
	LOG_INFO << "Waiting for first video";
	boost::this_thread::sleep_for(boost::chrono::seconds(1));
	LOG_INFO << "Resuming.....";
	auto boostVideoTS = std::stoull(boost::filesystem::path(readPath).stem().string());
	ReadPipeline r(readPath, reInitIntervalSecs, direction, parseFS, readFps);

	// read 4 frames
	auto i = 0;
	uint64_t lastFrameTS = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	lastFrameTS -= 2000;
	bool skipStep = false;
	
	while(1)
	{
		r.mp4Reader->step();
		auto frame = r.sink->pop().begin()->second;
		if (frame->isMp4ErrorFrame())
		{ 
			auto errorFrame = dynamic_cast<Mp4ErrorFrame*>(frame.get());
			auto type = errorFrame->errorCode;
			LOG_ERROR << "**********************************************************************************";
			LOG_ERROR << "Error occured in mp4Reader <" << type << "> msg <" << errorFrame->errorMsg << ">";
			LOG_ERROR << "**********************************************************************************";
			if (type == MP4_MISSING_VIDEOTRACK)
			{
				boost::this_thread::sleep_for(boost::chrono::seconds(1));
				r.mp4Reader->randomSeek(lastFrameTS + 1);
				continue;
			}
			else if (type == MP4_OPEN_FILE_FAILED)
			{
				boost::this_thread::sleep_for(boost::chrono::seconds(1));
				r.mp4Reader->randomSeek(lastFrameTS + 1);
				continue;
			}
		}
		if (frame->isEOS())
		{
			auto q = r.sink->getQue();
			
			while (!q->size())
			{
				auto queSize = q->size();
				LOG_INFO << "Waiting for more data on disk .....................";
				boost::this_thread::sleep_for(boost::chrono::seconds(1));
 				r.mp4Reader->step();
				skipStep = true;
			}
			auto queSize = q->size();
			continue;
		}
		LOG_INFO << "read image frame < " << i + 1 << ">";
		LOG_INFO << "frame->timestamp <" << frame->timestamp << ">";
		BOOST_TEST((frame->timestamp - lastFrameTS) < 20000);

		if (!((frame->timestamp - lastFrameTS) < 20000))
		{
			LOG_INFO << "whats wrong";
		}
		lastFrameTS = frame->timestamp;
		++i;

		if (i == 2000)
		{
			break;
		}
	}
	LOG_INFO << "Total Frames Read <" << i << ">";
	boost::this_thread::sleep_for(boost::chrono::milliseconds(90));
}

BOOST_AUTO_TEST_SUITE_END()