#include <boost/test/unit_test.hpp>

#include "Mp4ReaderSource.h"
#include "FileReaderModule.h"
#include "FileWriterModule.h"
#include "Logger.h"
#include "PipeLine.h"
#include "H264Metadata.h"
#include "Mp4VideoMetadata.h"
#include "Mp4WriterSink.h"
#include "H264EncoderV4L2.h"
#include "StatSink.h"
#include "H264Decoder.h"
#include "EglRenderer.h"
#include "NvTransform.h"
#include "AffineTransform.h"
#include "CudaStreamSynchronize.h"
#include "AffineTransformRev.h"
#include "EglRendererReview.h"

BOOST_AUTO_TEST_SUITE(FileRecovery_tests)

class MetadataSinkProps : public ModuleProps
{
public:
	MetadataSinkProps(int _uniqMetadata) : ModuleProps()
	{
		uniqMetadata = _uniqMetadata;
	}
	int uniqMetadata;
};

class MetadataSink : public Module
{
public:
	MetadataSink(MetadataSinkProps props = MetadataSinkProps(0)) : Module(SINK, "MetadataSink", props), mProps(props)
	{
	}
	virtual ~MetadataSink()
	{
	}
protected:
	bool process(frame_container& frames) 
	{
		auto frame = getFrameByType(frames, FrameMetadata::FrameType::MP4_VIDEO_METADATA);
		int num = (frame->fIndex + 1) % mProps.uniqMetadata;
		if (!isFrameEmpty(frame))
		{
			LOG_ERROR << "Reading Frame number " << num;
			if (frame->fIndex < 100)
			{
				metadata.assign(reinterpret_cast<char*>(frame->data()), frame->size());

				LOG_INFO << "Metadata\n frame_numer <" << frame->fIndex + 1 << "><" << metadata << ">";
				if (!mProps.uniqMetadata)
				{
					auto frameIndex = frame->fIndex;
					auto index = frameIndex % 5;
					auto myString = "ApraPipes_" + std::to_string(index);
					BOOST_TEST(myString == metadata);
				}
				else
				{
					int num = (frame->fIndex + 1) % mProps.uniqMetadata;
					num = num ? num : mProps.uniqMetadata;
					std::string shouldBeMeta = "frame_" + std::to_string(num);

					BOOST_TEST(shouldBeMeta.size() == metadata.size());
					BOOST_TEST(shouldBeMeta.data() == metadata.data());
				}
			}
		}

		return true;
	}
	bool validateInputPins() { return true; }
	bool validateInputOutputPins() { return true; }
	std::string metadata;
	MetadataSinkProps mProps;
};

void readVideoFrames(std::string fileToPlay)
{
	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);

	auto mp4ReaderProps = Mp4ReaderSourceProps(fileToPlay, false, 10, true, false, false);
	mp4ReaderProps.fps = 1000;
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	mp4Reader->addOutPutPin(h264ImageMetadata);

	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(mp4Reader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	p->run_all_threaded();
	
	boost::this_thread::sleep_for(boost::chrono::seconds(200));

	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
	
}

void seekVideoFrames(std::string fileToPlay)
{
	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);
	auto stream = cudastream_sp(new ApraCudaStream);


	auto mp4ReaderProps = Mp4ReaderSourceProps(fileToPlay, false, 10, true, false, false);
	mp4ReaderProps.fps = 60;
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	mp4Reader->addOutPutPin(h264ImageMetadata);

	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	H264DecoderProps decprops();
	// decprops.fps = 33;
	auto Decoder = boost::shared_ptr<H264Decoder>(new H264Decoder(H264DecoderProps()));
	mp4Reader->setNext(Decoder);

	auto nvTransformProps = NvTransformProps(ImageMetadata::RGBA);
	// nvTransformProps.qlen = 2;
	nvTransformProps.fps =33; 
	auto m_nv12_to_yuv444Transform = boost::shared_ptr<NvTransform>(new NvTransform(nvTransformProps));
	Decoder->setNext(m_nv12_to_yuv444Transform);

	AffineTransformProps affineProps(AffineTransformProps::LINEAR, stream, 0, 4096, 0, 0, 1);
	affineProps.fps = 33;
	// affineProps.qlen = 20;
	auto m_reviewAffineTransform = boost::shared_ptr<AffineTransform>(new AffineTransform(affineProps));
	m_nv12_to_yuv444Transform->setNext(m_reviewAffineTransform);

	auto sync = boost::shared_ptr<CudaStreamSynchronize>(new CudaStreamSynchronize(CudaStreamSynchronizeProps(stream)));
	m_reviewAffineTransform->setNext(sync);

	EglRendererReviewProps eglProps(0,0);
	eglProps.fps = 60;
	auto sink = boost::shared_ptr<EglRendererReview>(new EglRendererReview(eglProps));
	sync->setNext(sink);

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(mp4Reader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	p->run_all_threaded();
	
	boost::this_thread::sleep_for(boost::chrono::seconds(10));

	LOG_ERROR << mp4Reader->getOpenVideoDurationInSecs();

	mp4Reader->randomSeek((1739785312264 + (mp4Reader->getOpenVideoDurationInSecs()*1000) - 1000));

	boost::this_thread::sleep_for(boost::chrono::seconds(15));

	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
	
}

void readVideo(std::string fileToPlay)
{
	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);
	int uniqMetadata = 0;

	boost::filesystem::path dir ("data/Mp4_videos/corruptFrame/frames");
	boost::filesystem::path file("frame_??????.h264");

	auto mp4ReaderProps = Mp4ReaderSourceProps(fileToPlay, false, 0, true, true, false);
	mp4ReaderProps.fps = 10000;
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	mp4Reader->addOutPutPin(h264ImageMetadata);

	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	boost::filesystem::path full_path = dir/file;
	LOG_INFO << full_path;
	auto fileWriterProps = FileWriterModuleProps(full_path.string());
	auto fileWriter = boost::shared_ptr<FileWriterModule>(new FileWriterModule(fileWriterProps));
	std::vector<std::string> mImagePin;
	mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::FrameType::H264_DATA);
	mp4Reader->setNext(fileWriter, mImagePin);

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(mp4Reader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	p->run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(2));


	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
}

void writeVideo(std::string outFolderPath)
{

	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);

	auto fileReaderProps = FileReaderModuleProps("data/Mp4_videos/corruptFrame/frames/", 0, -1);
	fileReaderProps.fps = 10000;
	fileReaderProps.readLoop = false;

	auto fileReader = boost::shared_ptr<Module>(new FileReaderModule(fileReaderProps));
	auto h264ImageMetadata2 = framemetadata_sp(new H264Metadata(1000, 1000));
	fileReader->addOutputPin(h264ImageMetadata2);

	auto mp4WriterSinkProps = Mp4WriterSinkProps(UINT32_MAX, 1, 30, outFolderPath);
	mp4WriterSinkProps.logHealth = true;
	mp4WriterSinkProps.logHealthFrequency = 100;
	auto mp4WriterSink = boost::shared_ptr<Mp4WriterSink>(new Mp4WriterSink(mp4WriterSinkProps));
	fileReader->setNext(mp4WriterSink);

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(fileReader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	p->run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(60));

	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
}

void cropVideo (std::string videoPath, std::string outFolderPath)
{
	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);
	int uniqMetadata = 0;

	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, false, 10, true, false, false, 0);
	mp4ReaderProps.fps = 1000;
	// mp4ReaderProps.logHealth = true;
	// mp4ReaderProps.logHealthFrequency = 100;
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	// mp4Reader->registerCallback();
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	mp4Reader->addOutPutPin(h264ImageMetadata);

	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	std::vector<std::string> mImagePin;
	mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::FrameType::H264_DATA);

	auto mp4WriterSinkProps = Mp4WriterSinkProps(UINT32_MAX, 1, 30, outFolderPath);
	mp4WriterSinkProps.logHealth = true;
	mp4WriterSinkProps.logHealthFrequency = 100;
	auto mp4WriterSink = boost::shared_ptr<Mp4WriterSink>(new Mp4WriterSink(mp4WriterSinkProps));
	mp4Reader->setNext(mp4WriterSink, mImagePin);

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(mp4Reader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	p->run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(10));

	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
}

void readFiles(std::string fileToPlay, std::string changedVideoPath)
{
	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::error;
	Logger::setLogLevel(boost::log::trivial::severity_level::error);
	Logger::initLogger(loggerProps);
	auto stream = cudastream_sp(new ApraCudaStream);


	LOG_ERROR << "Reading file..."<< fileToPlay <<"\n";
	auto mp4ReaderProps = Mp4ReaderSourceProps(fileToPlay, false, 0, true, false, false);
	mp4ReaderProps.fps = 1000;
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	mp4Reader->addOutPutPin(h264ImageMetadata);

	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	H264DecoderProps decprops();
	// decprops.fps = 33;
	auto Decoder = boost::shared_ptr<H264Decoder>(new H264Decoder(H264DecoderProps()));
	mp4Reader->setNext(Decoder);

	auto nvTransformProps = NvTransformProps(ImageMetadata::RGBA);
	// nvTransformProps.qlen = 2;
	nvTransformProps.fps =33; 
	auto m_nv12_to_yuv444Transform = boost::shared_ptr<NvTransform>(new NvTransform(nvTransformProps));
	Decoder->setNext(m_nv12_to_yuv444Transform);

	AffineTransformProps affineProps(AffineTransformProps::LINEAR, stream, 0, 4096, 0, 0, 1);
	affineProps.fps = 33;
	// affineProps.qlen = 20;
	auto m_reviewAffineTransform = boost::shared_ptr<AffineTransform>(new AffineTransform(affineProps));
	m_nv12_to_yuv444Transform->setNext(m_reviewAffineTransform);

	auto sync = boost::shared_ptr<CudaStreamSynchronize>(new CudaStreamSynchronize(CudaStreamSynchronizeProps(stream)));
	m_reviewAffineTransform->setNext(sync);

	EglRendererProps eglProps(0,0);
	eglProps.fps = 60;
	auto sink = boost::shared_ptr<EglRenderer>(new EglRenderer(eglProps));
	sync->setNext(sink);

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(mp4Reader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	p->run_all_threaded();
	
	boost::this_thread::sleep_for(boost::chrono::seconds(30));
	
	auto currReaderProps = mp4Reader->getProps();
	currReaderProps.videoPath = changedVideoPath;
	mp4Reader->setProps(currReaderProps);

	// Mp4ReaderSourceProps propsChange(changedVideoPath, false, 0, true, false, false);
	// propsChange.fps = 1000;
	// mp4Reader->setProps(propsChange);

	boost::this_thread::sleep_for(boost::chrono::seconds(30));

	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
	
}

void readAndCloseFiles(std::string fileToPlay, std::string changedVideoPath)
{
	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::trace;
	Logger::setLogLevel(boost::log::trivial::severity_level::trace);
	Logger::initLogger(loggerProps);
	auto stream = cudastream_sp(new ApraCudaStream);


	LOG_ERROR << "Reading file..."<< fileToPlay <<"\n";
	auto mp4ReaderProps = Mp4ReaderSourceProps(fileToPlay, false, 10, true, false, false); // 10 is iMp
	mp4ReaderProps.fps = 1000;
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	mp4Reader->addOutPutPin(h264ImageMetadata);

	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	H264DecoderProps decprops();
	// decprops.fps = 33;
	auto Decoder = boost::shared_ptr<H264Decoder>(new H264Decoder(H264DecoderProps()));
	mp4Reader->setNext(Decoder);

	auto nvTransformProps = NvTransformProps(ImageMetadata::RGBA);
	// nvTransformProps.qlen = 2;
	nvTransformProps.fps =33; 
	auto m_nv12_to_yuv444Transform = boost::shared_ptr<NvTransform>(new NvTransform(nvTransformProps));
	Decoder->setNext(m_nv12_to_yuv444Transform);

	AffineTransformProps affineProps(AffineTransformProps::LINEAR, stream, 0, 4096, 0, 0, 1);
	affineProps.fps = 33;
	// affineProps.qlen = 20;
	auto m_reviewAffineTransform = boost::shared_ptr<AffineTransform>(new AffineTransform(affineProps));
	m_nv12_to_yuv444Transform->setNext(m_reviewAffineTransform);

	auto sync = boost::shared_ptr<CudaStreamSynchronize>(new CudaStreamSynchronize(CudaStreamSynchronizeProps(stream)));
	m_reviewAffineTransform->setNext(sync);

	EglRendererReviewProps eglProps(0,0);
	eglProps.fps = 60;
	auto sink = boost::shared_ptr<EglRendererReview>(new EglRendererReview(eglProps));
	sync->setNext(sink);

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(mp4Reader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	p->run_all_threaded();
	
	boost::this_thread::sleep_for(boost::chrono::seconds(10));
	
	auto currReaderProps = mp4Reader->getProps();
	currReaderProps.videoPath = changedVideoPath;
	mp4Reader->setProps(currReaderProps);

	boost::this_thread::sleep_for(boost::chrono::seconds(5));
	mp4Reader->closeOpenFile();  // Replace "" with closeOPenFileFUnction
	boost::this_thread::sleep_for(boost::chrono::seconds(5));

	currReaderProps = mp4Reader->getProps();
	currReaderProps.videoPath = fileToPlay;
	mp4Reader->setProps(currReaderProps);

	LOG_ERROR << "<======================Close File ==============+++>>> ";
	boost::this_thread::sleep_for(boost::chrono::seconds(30));
	LOG_ERROR << "<======================Stopping Pipeline ==============+++>>> ";
	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
	
}

void checkRename(std::string fileToPlay, std::string changedVideoPath)
{
	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::trace;
	Logger::setLogLevel(boost::log::trivial::severity_level::trace);
	Logger::initLogger(loggerProps);
	auto stream = cudastream_sp(new ApraCudaStream);


	LOG_ERROR << "Reading file..."<< fileToPlay <<"\n";
	auto mp4ReaderProps = Mp4ReaderSourceProps(fileToPlay, false, 10, true, false, false); // 10 is iMp
	mp4ReaderProps.fps = 1000;
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	mp4Reader->addOutPutPin(h264ImageMetadata);

	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	H264DecoderProps decprops();
	// decprops.fps = 33;
	auto Decoder = boost::shared_ptr<H264Decoder>(new H264Decoder(H264DecoderProps()));
	mp4Reader->setNext(Decoder);

	auto nvTransformProps = NvTransformProps(ImageMetadata::RGBA);
	// nvTransformProps.qlen = 2;
	nvTransformProps.fps =33; 
	auto m_nv12_to_yuv444Transform = boost::shared_ptr<NvTransform>(new NvTransform(nvTransformProps));
	Decoder->setNext(m_nv12_to_yuv444Transform);

	AffineTransformProps affineProps(AffineTransformProps::LINEAR, stream, 0, 4096, 0, 0, 1);
	affineProps.fps = 33;
	// affineProps.qlen = 20;
	auto m_reviewAffineTransform = boost::shared_ptr<AffineTransform>(new AffineTransform(affineProps));
	m_nv12_to_yuv444Transform->setNext(m_reviewAffineTransform);

	auto sync = boost::shared_ptr<CudaStreamSynchronize>(new CudaStreamSynchronize(CudaStreamSynchronizeProps(stream)));
	m_reviewAffineTransform->setNext(sync);

	EglRendererReviewProps eglProps(0,0);
	eglProps.fps = 60;
	auto sink = boost::shared_ptr<EglRendererReview>(new EglRendererReview(eglProps));
	sync->setNext(sink);
	
	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(mp4Reader);
	
	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}
	
	p->run_all_threaded();
	
	boost::this_thread::sleep_for(boost::chrono::seconds(10));
	
	if (rename("./data/Mp4_videos/corruptFrame/repairedVideo2.mp4", changedVideoPath.c_str()) == 0) 
	{
		printf("File renamed successfully\n");
	} 
	else 
	{
		printf("Error renaming file\n");
	}

	boost::this_thread::sleep_for(boost::chrono::seconds(1));

	auto currReaderProps = mp4Reader->getProps();
	currReaderProps.videoPath = changedVideoPath;
	mp4Reader->setProps(currReaderProps);

	boost::this_thread::sleep_for(boost::chrono::seconds(5));
	mp4Reader->closeOpenFile();  // Replace "" with closeOPenFileFUnction
	boost::this_thread::sleep_for(boost::chrono::seconds(5));

	if (rename(changedVideoPath.c_str(), "./data/Mp4_videos/corruptFrame/repairedVideo2.mp4") == 0) 
	{
        printf("File renamed successfully\n");
    } 
	else 
	{
        printf("Error renaming file\n");
    }

	boost::this_thread::sleep_for(boost::chrono::seconds(1));

	currReaderProps = mp4Reader->getProps();
	currReaderProps.videoPath = fileToPlay;
	// currReaderProps.videoPath = "";
	mp4Reader->setProps(currReaderProps);

	LOG_ERROR << "<======================Close File ==============+++>>> ";
	boost::this_thread::sleep_for(boost::chrono::seconds(30));
	LOG_ERROR << "<======================Stopping Pipeline ==============+++>>> ";


	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
	
}

BOOST_AUTO_TEST_CASE(testFrameNumbers)
{
	std::string videoPath = "data/Mp4_videos/corruptFrame/repairedVideo2.mp4";
	readVideoFrames(videoPath);
}

BOOST_AUTO_TEST_CASE(seekToLast)
{
	std::string videoPath = "data/Mp4_videos/corruptFrame/repairedVideo2.mp4";
	seekVideoFrames(videoPath);
}

BOOST_AUTO_TEST_CASE(newFileWrite)
{
	std::string videoPath = "data/Mp4_videos/corruptFrame/full_rec.mp4";
	std::string outFolderPath = "data/Mp4_videos/corruptFrame/repairedVideo.mp4";
	LOG_ERROR << "Reading file...\n";
	// readVideo(videoPath);
	LOG_ERROR << "Will make new file now!\n";
	writeVideo(outFolderPath);
}

BOOST_AUTO_TEST_CASE(cropCorruptVideo)
{
	// corrupt_empty.mp4
	// std::string videoPath = "data/Mp4_videos/corruptFrame/corrupt_file.mp4";
	// std::string videoPath = "data/Mp4_videos/corruptFrame/test.mp4";
	std::string videoPath = "data/Mp4_videos/corruptFrame/27feb.mp4";
	std::string outFolderPath = "data/Mp4_videos/corruptFrame/repairedVideo.mp4";
	cropVideo(videoPath, outFolderPath);
}

BOOST_AUTO_TEST_CASE(readMultipleFiles)
{
	std::string videoPath = "./data/Mp4_videos/corruptFrame/27feb.mp4";
	std::string outFolderPath = "./data/Mp4_videos/corruptFrame/temp.mp4";
	readFiles(videoPath, outFolderPath);
}


BOOST_AUTO_TEST_CASE(closeFile)
{
	std::string videoPath = "./data/Mp4_videos/corruptFrame/27feb.mp4";
	std::string outFolderPath = "./data/Mp4_videos/corruptFrame/temp.mp4";
	readAndCloseFiles(videoPath, outFolderPath);
}

BOOST_AUTO_TEST_CASE(checkRenameFiles)
{
	std::string videoPath = "./data/Mp4_videos/corruptFrame/27feb.mp4";
	std::string outFolderPath = "./data/Mp4_videos/corruptFrame/repairedVideo.mp4";
	checkRename(videoPath, outFolderPath);
}

BOOST_AUTO_TEST_CASE(checkMultipleFiles)
{
	std::string videoPath = "./data/Mp4_videos/corruptFrame/27feb.mp4";
	std::string outFolderPath = "./data/Mp4_videos/corruptFrame/repairedVideo.mp4";
	checkRename(videoPath, outFolderPath);
}

BOOST_AUTO_TEST_SUITE_END()