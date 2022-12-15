#include <boost/test/unit_test.hpp>
#include "NVRControlModule.h"
#include "FileReaderModule.h"
#include "EglRenderer.h"
#include "PipeLine.h"
#include <boost/filesystem.hpp>
#include "NvV4L2Camera.h" //Jetson
#include "NvTransform.h"
#include "H264EncoderV4L2.h"
#include "DMAFDToHostCopy.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "H264Utils.h"
#include "Mp4ReaderSource.h"
#include "Mp4VideoMetadata.h"
#include "Mp4WriterSink.h"
#include "Mp4WriterSinkUtils.h"
#include "EncodedImageMetadata.h"
#include "Module.h"
#include "PropsChangeMetadata.h"
#include "MultimediaQueueXform.h"
#include "H264Metadata.h"
#include "Command.h"
#include <boost/lexical_cast.hpp>

BOOST_AUTO_TEST_SUITE(eglrenderer_tests)

void key_func(boost::shared_ptr<NVRControlModule>& mControl)
{

	while (true) {
		int k;
		k = getchar();
		if (k == 97)
		{
			BOOST_LOG_TRIVIAL(info) << "Starting Render!!";
			mControl->nvrView(true);
			mControl->step();
		}
		else if (k == 100)
		{
			BOOST_LOG_TRIVIAL(info) << "Stopping Render!!";
			mControl->nvrView(false);
			mControl->step();
		}
		else if (k == 101)
		{
			boost::posix_time::ptime const time_epoch(boost::gregorian::date(1970, 1, 1));
			auto now = (boost::posix_time::microsec_clock::universal_time() - time_epoch).total_milliseconds();
			uint64_t seekStartTS = now + 4000;
			uint64_t seekEndTS = now + 5000;
			LOG_ERROR << "Starting export!!";
			mControl->nvrExport(seekStartTS, seekEndTS);
			mControl->step();
		}
		else
		{
			BOOST_LOG_TRIVIAL(info) << "The value pressed is .."<< k;
		}
	}
}

void key_Read_func(boost::shared_ptr<NVRControlModule>& mControl, boost::shared_ptr<Mp4ReaderSource>& mp4Reader)
{

	while (true) {
		int k;
		k = getchar();
		if (k == 97)
		{
			BOOST_LOG_TRIVIAL(info) << "Starting Render!!";
			mControl->nvrView(true);
			mControl->step();
		}
		if (k == 100)
		{
			BOOST_LOG_TRIVIAL(info) << "Stopping Render!!";
			mControl->nvrView(false);
			mControl->step();
		}
		if (k == 101)
		{
			/*uint64_t x, y;
			cout << "Enter start time of Export : ";
			cin >> x;
			cout << "Enter end time of Export : ";
			cin >> y;
			cout << "Start time is " << x << " End time is " << y;*/
			BOOST_LOG_TRIVIAL(info) << "Starting Reading from disk!!";
			boost::posix_time::ptime const time_epoch(boost::gregorian::date(1970, 1, 1));
			auto now = (boost::posix_time::microsec_clock::universal_time() - time_epoch).total_milliseconds();
			uint64_t seekStartTS = now - 180000;
			uint64_t seekEndTS = now + 120000;
			mControl->nvrExport(seekStartTS, seekEndTS);
			mControl->step();
		}
		if (k == 112)
		{
			BOOST_LOG_TRIVIAL(info) << "Stopping Pipeline Input";
		}

		else
		{
			BOOST_LOG_TRIVIAL(info) << "The value pressed is .." << k;
		}
	}
}

BOOST_AUTO_TEST_CASE(basic, *boost::unit_test::disabled())
{
	int width = 640;
	int height = 480;

    FileReaderModuleProps fileReaderProps("./data/ArgusCamera");
	fileReaderProps.fps = 30;
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::UYVY, CV_8UC1, 0, CV_8U, FrameMetadata::MemType::DMABUF, true));
	
	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0)));
	fileReader->setNext(sink);

	PipeLine p("test");
	p.appendModule(fileReader);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(10));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(NXPipeline)
{
    Logger::setLogLevel(boost::log::trivial::severity_level::error);

	auto v4L2Source = boost::shared_ptr<Module>(new NvV4L2Camera(NvV4L2CameraProps(1280, 720, 10)));
    
	//NV_Transform
	auto nv_transform = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::RGBA)));
	v4L2Source->setNext(nv_transform);
	
	//EGL_Renderer
    auto renderer = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0)));
    nv_transform->setNext(renderer);

	//NV_Transform for encoder
	auto nv_transform_encode = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::YUV420)));
	v4L2Source->setNext(nv_transform_encode);

	//v4l2Encoder
	H264EncoderV4L2Props encoderProps;
	encoderProps.targetKbps = 2048;
	auto encoder = boost::shared_ptr<Module>(new H264EncoderV4L2(encoderProps));
	nv_transform_encode->setNext(encoder);

	//auto dmaToHost = boost::shared_ptr<Module>(new DMAFDToHostCopy(DMAFDToHostCopyProps()));
	//encoder->setNext(dmaToHost);

	std::string outFolderPath_1 = "./data/testOutput/mp4_videos/24bpp/";
	auto mp4writer_1 = boost::shared_ptr<Module>(new Mp4WriterSink(Mp4WriterSinkProps(1, 10, 24, outFolderPath_1)));
	encoder->setNext(mp4writer_1);

	auto multiProps = MultimediaQueueXformProps(120000, 30000, true);
	auto multiQue = boost::shared_ptr<MultimediaQueueXform>(new MultimediaQueueXform(multiProps));
	encoder->setNext(multiQue);

	std::string outFolderPath_2 = "./data/testOutput/mp4_videos/new/ExportVids/";
	auto mp4WriterSinkProps_2 = Mp4WriterSinkProps(1, 10, 24, outFolderPath_2);
	mp4WriterSinkProps_2.logHealth = false;
	mp4WriterSinkProps_2.logHealthFrequency = 100;
	auto mp4Writer_2 = boost::shared_ptr<Mp4WriterSink>(new Mp4WriterSink(mp4WriterSinkProps_2));
	multiQue->setNext(mp4Writer_2);

	//ControlModule 
	auto mControl = boost::shared_ptr<NVRControlModule>(new NVRControlModule(NVRControlModuleProps()));

    PipeLine p("test");
	std::thread inp(key_func, std::ref(mControl));
	p.appendModule(v4L2Source);
	p.addControlModule(mControl);
	mControl->enrollModule("Renderer", renderer);
	mControl->enrollModule("Writer-1", mp4writer_1);
	mControl->enrollModule("Writer-2", mp4Writer_2);
	mControl->enrollModule("MultimediaQueue", multiQue);

	BOOST_TEST(p.init());
	mControl->init();

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(6000));
	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(NXPipeline_2)
{
    Logger::setLogLevel(boost::log::trivial::severity_level::error);

	auto v4L2Source = boost::shared_ptr<Module>(new NvV4L2Camera(NvV4L2CameraProps(1280, 720, 10)));
    
	//NV_Transform
	auto nv_transform = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::RGBA)));
	v4L2Source->setNext(nv_transform);
	
	//EGL_Renderer
    auto renderer = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0)));
    nv_transform->setNext(renderer);

	//NV_Transform for encoder
	auto nv_transform_encode = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::YUV420)));
	v4L2Source->setNext(nv_transform_encode);

	//v4l2Encoder
	H264EncoderV4L2Props encoderProps;
	encoderProps.targetKbps = 2048;
	auto encoder = boost::shared_ptr<Module>(new H264EncoderV4L2(encoderProps));
	nv_transform_encode->setNext(encoder);

	//mp4Writer-1
	std::string outFolderPath_1 = "./data/testOutput/mp4_videos/24bpp/";
	auto mp4writer_1 = boost::shared_ptr<Module>(new Mp4WriterSink(Mp4WriterSinkProps(1, 10, 24, outFolderPath_1)));
	encoder->setNext(mp4writer_1);

	//MultimediaQueue
	auto multiProps = MultimediaQueueXformProps(120000, 30000, true);
	auto multiQue = boost::shared_ptr<MultimediaQueueXform>(new MultimediaQueueXform(multiProps));
	encoder->setNext(multiQue);
	
	//mp4Writer-2
	std::string outFolderPath_2 = "./data/testOutput/mp4_videos/Export_Videos/";
	auto mp4writer_2 = boost::shared_ptr<Module>(new Mp4WriterSink(Mp4WriterSinkProps(1, 10, 24, outFolderPath_2)));
	multiQue->setNext(mp4writer_2);
	
	//mp4Reader
	std::string startingVideoPath = "./data/Mp4_videos/h264_video/20221010/0012/1668064027062.mp4";
	std::string outPath = "./data/testOutput/mp4_videos/24bpp";
	std::string changedVideoPath = "./data/testOutput/mp4_videos/24bpp/20221023/0011/";                              
	boost::filesystem::path file("frame_??????.h264");
	auto frameType = FrameMetadata::FrameType::H264_DATA;
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	boost::filesystem::path dir(outPath);
	auto mp4ReaderProps = Mp4ReaderSourceProps(startingVideoPath, false, true);
	mp4ReaderProps.logHealth = true;
	mp4ReaderProps.logHealthFrequency = 100;
	mp4ReaderProps.fps = 30;
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	mp4Reader->addOutPutPin(h264ImageMetadata);
	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);
	mp4Reader->setNext(mp4writer_2);


	//ControlModule 
	auto mControl = boost::shared_ptr<NVRControlModule>(new NVRControlModule(NVRControlModuleProps()));

    PipeLine p("test");
	std::thread inp(key_Read_func, std::ref(mControl), std::ref(mp4Reader));
	p.appendModule(v4L2Source);
	p.appendModule(mp4Reader);
	p.addControlModule(mControl);

	mControl->enrollModule("Reader", mp4Reader);
	mControl->enrollModule("Renderer", renderer);
	mControl->enrollModule("Writer-1", mp4writer_1);
	mControl->enrollModule("MultimediaQueue", multiQue);
	mControl->enrollModule("Writer-1", mp4writer_2);

	BOOST_TEST(p.init());
	mControl->init();
	mp4Reader->play(false);

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(240));
	for (const auto& folder : boost::filesystem::recursive_directory_iterator(boost::filesystem::path("./data/testOutput/mp4_videos/24bpp/20221115/0012/")))
	{
		if (boost::filesystem::is_regular_file(folder))
		{
			boost::filesystem::path p = folder.path();
			changedVideoPath = p.string();
			break;
		}
	}
	Mp4ReaderSourceProps propsChange(changedVideoPath, true);
	mp4Reader->setProps(propsChange);
	boost::this_thread::sleep_for(boost::chrono::seconds(3600));
	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(NXPipeline_3)
{
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	auto v4L2Source = boost::shared_ptr<Module>(new NvV4L2Camera(NvV4L2CameraProps(1280, 720, 10)));
    
	//NV_Transform
	auto nv_transform = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::RGBA)));
	v4L2Source->setNext(nv_transform);
	
	//EGL_Renderer
    auto renderer = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0)));
    nv_transform->setNext(renderer);

	//NV_Transform for encoder
	auto nv_transform_encode = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::YUV420)));
	v4L2Source->setNext(nv_transform_encode);

	//v4l2Encoder
	H264EncoderV4L2Props encoderProps;
	encoderProps.targetKbps = 2048;
	auto encoder = boost::shared_ptr<Module>(new H264EncoderV4L2(encoderProps));
	nv_transform_encode->setNext(encoder);

	//mp4Writer
	std::string outFolderPath_1 = "./data/testOutput/mp4_videos/newVids/24bpp/";
	auto mp4writer_1 = boost::shared_ptr<Module>(new Mp4WriterSink(Mp4WriterSinkProps(1, 10, 24, outFolderPath_1)));
	encoder->setNext(mp4writer_1);


    PipeLine p("test");
	p.appendModule(v4L2Source);

	BOOST_TEST(p.init());
	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(360));
	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_SUITE_END()