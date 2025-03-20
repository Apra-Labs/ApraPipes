#include <boost/test/unit_test.hpp>

#include "Logger.h"
#include "PipeLine.h"
#include "FileReaderModule.h"
#include "Mp4ReaderSource.h"
#include "FileWriterModule.h"
#include "StatSink.h"
#include "FrameMetadata.h"
#include "EncodedImageMetadata.h"
#include "H264Metadata.h"
#include "Mp4VideoMetadata.h"
#include "H264Decoder.h"
#include "EglRenderer.h"
#include "NvTransform.h"
#include "AffineTransform.h"
#include "CudaStreamSynchronize.h"
#include "AffineTransformRev.h"

BOOST_AUTO_TEST_SUITE(Mp4ReaderSource_tests)

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
	bool process(frame_container& frames) {
		auto frame = getFrameByType(frames, FrameMetadata::FrameType::MP4_VIDEO_METADATA);
		if (!isFrameEmpty(frame))
		{
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

void read_video_extract_frames(std::string videoPath, std::string outPath, boost::filesystem::path file, framemetadata_sp inputMetadata, FrameMetadata::FrameType frameType, bool parseFS, int uniqMetadata = 0)
{
	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);

	boost::filesystem::path dir(outPath);

	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, parseFS, 0, true, false, false);
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));

	mp4Reader->addOutPutPin(inputMetadata);
	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	boost::filesystem::path full_path = dir / file;
	LOG_INFO << full_path;
	auto fileWriterProps = FileWriterModuleProps(full_path.string());
	auto fileWriter = boost::shared_ptr<FileWriterModule>(new FileWriterModule(fileWriterProps));
	std::vector<std::string> mImagePin;
	mImagePin = mp4Reader->getAllOutputPinsByType(frameType);
	mp4Reader->setNext(fileWriter, mImagePin);

	StatSinkProps statSinkProps;
	statSinkProps.logHealth = true;
	statSinkProps.logHealthFrequency = 10;
	auto statSink = boost::shared_ptr<Module>(new StatSink(statSinkProps));
	mp4Reader->setNext(statSink);

	auto metaSinkProps = MetadataSinkProps(uniqMetadata);
	metaSinkProps.logHealth = true;
	metaSinkProps.logHealthFrequency = 10;
	auto metaSink = boost::shared_ptr<Module>(new MetadataSink(metaSinkProps));
	mp4Reader->setNext(metaSink);

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(mp4Reader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	p->run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(100));

	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
}

void random_seek_video(std::string skipDir, uint64_t seekStartTS, uint64_t seekEndTS, std::string startingVideoPath, std::string outPath, framemetadata_sp inputMetadata, FrameMetadata::FrameType frameType, boost::filesystem::path file)
{
	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);

	boost::filesystem::path dir(outPath);

	auto mp4ReaderProps = Mp4ReaderSourceProps(startingVideoPath, false, 0, true, false, false);
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	mp4Reader->addOutPutPin(inputMetadata);
	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	mp4ReaderProps.skipDir = skipDir;

	boost::filesystem::path full_path = dir / file;
	LOG_INFO << full_path;
	auto fileWriterProps = FileWriterModuleProps(full_path.string());
	auto fileWriter = boost::shared_ptr<FileWriterModule>(new FileWriterModule(fileWriterProps));
	std::vector<std::string> mImagePin;
	mImagePin = mp4Reader->getAllOutputPinsByType(frameType);
	mp4Reader->setNext(fileWriter, mImagePin);

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(mp4Reader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	mp4Reader->setProps(mp4ReaderProps);
	mp4Reader->randomSeek(seekStartTS,seekEndTS);

	p->run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(10));

	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
}

BOOST_AUTO_TEST_CASE(mp4v_to_rgb_24_jpg)
{
	std::string videoPath = "./data/Mp4_videos/jpg_video/20220928/0013/10.mp4";
	std::string outPath = "data/testOutput/outFrames";
	boost::filesystem::path file("frame_??????.jpg");
	auto frameType = FrameMetadata::FrameType::ENCODED_IMAGE;
	auto encodedImageMetadata = framemetadata_sp(new EncodedImageMetadata(0, 0));
	bool parseFS = false;
	read_video_extract_frames(videoPath, outPath, file, encodedImageMetadata, frameType, parseFS);
}

BOOST_AUTO_TEST_CASE(mp4v_to_mono_8_jpg)
{
	std::string videoPath = "./data/Mp4_videos/jpg_video/20220928/0013/1666943213667.mp4";
	std::string outPath = "data/testOutput/outFrames";
	boost::filesystem::path file("frame_??????.jpg");
	auto frameType = FrameMetadata::FrameType::ENCODED_IMAGE;
	auto encodedImageMetadata = framemetadata_sp(new EncodedImageMetadata(0, 0));
	bool parseFS = false;
	read_video_extract_frames(videoPath, outPath, file, encodedImageMetadata, frameType, parseFS);
}

BOOST_AUTO_TEST_CASE(mp4v_read_metadata_jpg)
{
	std::string videoPath = "data/Mp4_videos/jpg_video_metada/20220928/0014/1666949168743.mp4";
	std::string outPath = "./data/testOutput/outFrames";
	boost::filesystem::path file("frame_??????.jpg");
	auto frameType = FrameMetadata::FrameType::ENCODED_IMAGE;
	auto encodedImageMetadata = framemetadata_sp(new EncodedImageMetadata(0, 0));
	bool parseFS = false;
	read_video_extract_frames(videoPath, outPath, file, encodedImageMetadata, frameType, parseFS);
}

BOOST_AUTO_TEST_CASE(fs_parsing_jpg)
{
	/* file structure parsing test */
	std::string videoPath = "data/Mp4_videos/jpg_video/20220928/0013/1666943213667.mp4";
	std::string outPath = "data/testOutput/outFrames";
	boost::filesystem::path file("frame_??????.jpg");
	auto frameType = FrameMetadata::FrameType::ENCODED_IMAGE;
	auto encodedImageMetadata = framemetadata_sp(new EncodedImageMetadata(0, 0));
	bool parseFS = true;
	read_video_extract_frames(videoPath, outPath, file, encodedImageMetadata, frameType, parseFS, 5);
}

BOOST_AUTO_TEST_CASE(random_seek_jpg)
{
	std::string skipDir = "data/Mp4_videos/jpg_video_metada/";
	std::string startingVideoPath = "data/Mp4_videos/jpg_video_metada/20220928/0014/1666949168743.mp4";
	std::string outPath = "data/testOutput/outFrames";
	uint64_t seekStartTS = 1666949171743;
	uint64_t seekEndTS = 1666949175743;
	boost::filesystem::path file("frame_??????.jpg");
	auto frameType = FrameMetadata::FrameType::ENCODED_IMAGE;
	auto encodedImageMetadata = framemetadata_sp(new EncodedImageMetadata(0, 0));
	random_seek_video(skipDir, seekStartTS, seekEndTS, startingVideoPath, outPath, encodedImageMetadata, frameType, file);
}

BOOST_AUTO_TEST_CASE(mp4v_to_h264frames_metadata)
{
	std::string videoPath = "./data/Mp4_videos/corruptFrame/countFrame.mp4";
	std::string outPath = "testP/outFrames";
	bool parseFS = false;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	boost::filesystem::path file("frame_??????.h264");
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	read_video_extract_frames(videoPath, outPath, file, h264ImageMetadata, frameType, parseFS);
}

BOOST_AUTO_TEST_CASE(mp4v_to_h264frames)
{
	std::string videoPath = "./data/Mp4_videos/h264_video/20221010/0012/1668064027062.mp4";
	std::string outPath = "data/testOutput/outFrames";
	bool parseFS = false;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	boost::filesystem::path file("frame_??????.h264");
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	read_video_extract_frames(videoPath, outPath, file, h264ImageMetadata, frameType, parseFS);
}

BOOST_AUTO_TEST_CASE(random_seek_h264)
{
	std::string skipDir = "data/Mp4_videos/h264_video/";
	std::string startingVideoPath = "./data/Mp4_videos/h264_video/20221010/0012/1668064027062.mp4";
	std::string outPath = "data/testOutput/outFrames";
	uint64_t seekStartTS = 1668064030062;
	uint64_t seekEndTS = 1668064032062;
	boost::filesystem::path file("frame_??????.h264");
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));

	random_seek_video(skipDir, seekStartTS, seekEndTS, startingVideoPath, outPath, h264ImageMetadata, frameType, file);
}

BOOST_AUTO_TEST_CASE(fs_parsing_h264, *boost::unit_test::disabled())
{
	/* file structure parsing test */
	std::string videoPath = "./data/Mp4_videos/h264_video/20221010/0012/1668064027062.mp4";
	std::string outPath = "data/testOutput/outFrames";
	boost::filesystem::path file("frame_??????.h264");
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	bool parseFS = true;
	read_video_extract_frames(videoPath, outPath, file, h264ImageMetadata, frameType, 5, parseFS);
}

BOOST_AUTO_TEST_CASE(read_timeStamp_from_custom_fileName)
{
	/* file structure parsing test */
	std::string videoPath = "./data/Mp4_videos/h264_video/apraH264.mp4";
	std::string outPath = "data/testOutput/outFrames";
	boost::filesystem::path file("frame_??????.h264");
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	bool parseFS = true;
	read_video_extract_frames(videoPath, outPath, file, h264ImageMetadata, frameType, 5, parseFS);
}


BOOST_AUTO_TEST_CASE(getSetProps)
{
	std::string videoPath = "/home/developer/workspace/ApraPipes/1684245623.mp4";
	std::string outPath = "./data/testOutput/outFrames/";
	std::string changedVideoPath = "/home/developer/workspace/ApraPipes/1684245623.mp4";
	bool parseFS = true;
	int uniqMetadata = 0;

	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);

	boost::filesystem::path dir(outPath);

	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, parseFS, 0, true, false, false);
	mp4ReaderProps.fps = 1000;
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	auto encodedImageMetadata = framemetadata_sp(new EncodedImageMetadata(0, 0));
	mp4Reader->addOutPutPin(encodedImageMetadata);
	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	boost::filesystem::path file("frame_??????.jpg");
	boost::filesystem::path full_path = dir / file;
	LOG_INFO << full_path;
	auto fileWriterProps = FileWriterModuleProps(full_path.string());
	auto fileWriter = boost::shared_ptr<FileWriterModule>(new FileWriterModule(fileWriterProps));
	std::vector<std::string> encodedImagePin;
	encodedImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::ENCODED_IMAGE);
	mp4Reader->setNext(fileWriter, encodedImagePin);

	StatSinkProps statSinkProps;
	statSinkProps.logHealth = true;
	statSinkProps.logHealthFrequency = 10;
	auto statSink = boost::shared_ptr<Module>(new StatSink(statSinkProps));
	mp4Reader->setNext(statSink);

	auto metaSinkProps = MetadataSinkProps(uniqMetadata);
	metaSinkProps.logHealth = true;
	metaSinkProps.logHealthFrequency = 10;
	auto metaSink = boost::shared_ptr<Module>(new MetadataSink(metaSinkProps));
	mp4Reader->setNext(metaSink);

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(mp4Reader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}
	p->run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(30));

	Mp4ReaderSourceProps propsChange(changedVideoPath, true, 0, true, false, false);
	mp4Reader->setProps(propsChange);

	boost::this_thread::sleep_for(boost::chrono::seconds(100));

	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
}

BOOST_AUTO_TEST_CASE(getSetProps2)
{
	std::string videoPath = "/home/developer/2024-02-25_20-13-02-264.mp4";
	std::string outPath = "./data/testOutput/outFrames/";
	std::string changedVideoPath = "/mnt/disks/ssd/ws_yashraj/ApraPipes/data/2024-02-01_19-18-26-747.mp4";
	bool parseFS = true;
	int uniqMetadata = 0;
	auto stream = cudastream_sp(new ApraCudaStream);
	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);

	boost::filesystem::path dir(outPath);

	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, false, 0, true, false, false);
	mp4ReaderProps.fps = 33;
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	mp4Reader->addOutPutPin(h264ImageMetadata);
	
	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	H264DecoderProps decprops();
	// decprops.fps = 33;
	auto Decoder = boost::shared_ptr<Module>(new H264Decoder(H264DecoderProps()));
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

	boost::this_thread::sleep_for(boost::chrono::seconds(3));
	LOG_ERROR << "PROPS WILL CHANGE";
	
	// p->pause();
	sink->play(false, true);
	sync->play(false, true);
	m_reviewAffineTransform->play(false, true);
	m_nv12_to_yuv444Transform->play(false, true);
	Decoder->play(false, true);
	mp4Reader->play(false, true);

	boost::this_thread::sleep_for(boost::chrono::seconds(30));
	// mp4Reader->play(true, true);
	// p->play();
	sink->play(true, true);
	sync->play(true, true);
	m_reviewAffineTransform->play(true, true);
	m_nv12_to_yuv444Transform->play(true, true);
	Decoder->play(true, true);
	mp4Reader->play(true, true);

	boost::this_thread::sleep_for(boost::chrono::seconds(300));
	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
}

BOOST_AUTO_TEST_CASE(inCompleteRead)
{

	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);

	std::string videoPath = "./testPlayback.mp4";
	bool parseFS = true;
	int uniqMetadata = 0;
	auto stream = cudastream_sp(new ApraCudaStream);

	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, false, 0, true, false, false);
	mp4ReaderProps.fps = 35;
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	mp4Reader->addOutPutPin(h264ImageMetadata);
	
	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	std::vector<std::string> mImagePin;
	mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::FrameType::H264_DATA);

	auto decProps = H264DecoderProps(); 
	decProps.fps = 35;
	auto Decoder = boost::shared_ptr<H264Decoder>(new H264Decoder(H264DecoderProps(decProps)));
	mp4Reader->setNext(Decoder, mImagePin);

	auto nvTransformProps = NvTransformProps(ImageMetadata::RGBA);
	// nvTransformProps.qlen = 2;
	nvTransformProps.fps = 35; 
	auto m_nv12_to_yuv444Transform = boost::shared_ptr<NvTransform>(new NvTransform(nvTransformProps));
	Decoder->setNext(m_nv12_to_yuv444Transform);

	AffineTransformRevProps affineProps(AffineTransformRevProps::LINEAR, stream, 0, 4096, 0, 0, 1);
	affineProps.fps = 35;
	// affineProps.qlen = 20;
	auto m_reviewAffineTransform = boost::shared_ptr<AffineTransformRev>(new AffineTransformRev(affineProps));
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
	boost::this_thread::sleep_for(boost::chrono::seconds(3000));
	//p->stop();
	//p->term();
	//p->wait_for_all();
	//p.reset();
}

BOOST_AUTO_TEST_CASE(defineSpeed)
{

	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);

	std::string videoPath = "data/Mp4_videos/corruptFrame/test.mp4";
	bool parseFS = true;
	int uniqMetadata = 0;
	auto stream = cudastream_sp(new ApraCudaStream);

	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, false, 0, true, false, false, 0);
	mp4ReaderProps.fps = 35;
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	mp4Reader->addOutPutPin(h264ImageMetadata);
	
	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	std::vector<std::string> mImagePin;
	mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::FrameType::H264_DATA);

	auto decProps = H264DecoderProps(); 
	decProps.fps = 35;
	auto Decoder = boost::shared_ptr<H264Decoder>(new H264Decoder(H264DecoderProps(decProps)));
	mp4Reader->setNext(Decoder, mImagePin);

	auto nvTransformProps = NvTransformProps(ImageMetadata::RGBA);
	// nvTransformProps.qlen = 2;
	nvTransformProps.fps = 35; 
	auto m_nv12_to_yuv444Transform = boost::shared_ptr<NvTransform>(new NvTransform(nvTransformProps));
	Decoder->setNext(m_nv12_to_yuv444Transform);

	AffineTransformRevProps affineProps(AffineTransformRevProps::LINEAR, stream, 0, 4096, 0, 0, 1);
	affineProps.fps = 35;
	// affineProps.qlen = 20;
	auto m_reviewAffineTransform = boost::shared_ptr<AffineTransformRev>(new AffineTransformRev(affineProps));
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
	// boost::this_thread::sleep_for(boost::chrono::seconds(2));
	// mp4Reader->setPlaybackSpeed(4);
	// int64_t gop = mp4Reader->getGOP();
	Decoder->changeDecoderSpeed(30,4,20);
	boost::this_thread::sleep_for(boost::chrono::seconds(200));
	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
}

BOOST_AUTO_TEST_SUITE_END()
