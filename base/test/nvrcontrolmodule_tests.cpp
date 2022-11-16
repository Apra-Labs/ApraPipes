#include <boost/test/unit_test.hpp>
#include <stdafx.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include "NVRPipeline.h"
#include "PipeLine.h"
#include "EncodedImageMetadata.h"
#include "FrameContainerQueue.h"
#include "Module.h"
#include "Utils.h"
#include "NVRControlModule.h"
#include "WebCamSource.h"
#include "H264EncoderNVCodec.h"
#include "Mp4WriterSink.h"
#include "CudaMemCopy.h"
#include "CudaStreamSynchronize.h"
#include "H264EncoderNVCodec.h"
#include "ColorConversionXForm.h"
#include "FileReaderModule.h"
#include "ImageViewerModule.h"
#include "KeyboardListener.h"
#include "MultimediaQueueXform.h"
#include "H264Metadata.h"
#include "ValveModule.h"
#include "FileWriterModule.h"

BOOST_AUTO_TEST_SUITE(nvrcontrolmodule_tests)


struct CheckThread {
	class SourceModuleProps : public ModuleProps
	{
	public:
		SourceModuleProps() : ModuleProps()
		{};
	};
	class TransformModuleProps : public ModuleProps
	{
	public:
		TransformModuleProps() : ModuleProps()
		{};
	};
	class SinkModuleProps : public ModuleProps
	{
	public:
		SinkModuleProps() : ModuleProps()
		{};
	};

	class SourceModule : public Module
	{
	public:
		SourceModule(SourceModuleProps props) : Module(SOURCE, "sourceModule", props)
		{
		};

	protected:
		bool process() { return false; }
		bool validateOutputPins()
		{
			return true;
		}
		bool validateInputPins()
		{
			return true;
		}
	};
	class TransformModule : public Module
	{
	public:
		TransformModule(TransformModuleProps props) :Module(TRANSFORM, "transformModule", props) {};
	protected:
		bool process() { return false; }
		bool validateOutputPins()
		{
			return true;
		}
		bool validateInputPins()
		{
			return true;
		}
	};
	class SinkModule : public Module
	{
	public:
		SinkModule(SinkModuleProps props) :Module(SINK, "mp4WritersinkModule", props) {};
	protected:
		bool process() { return false; }
		bool validateOutputPins()
		{
			return true;
		}
		bool validateInputPins()
		{
			return true;
		}
	};
};

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
		if (k == 100)
		{
			BOOST_LOG_TRIVIAL(info) << "Stopping Render!!";
			mControl->nvrView(false);
			mControl->step();
		}
		if (k == 101)
		{
			BOOST_LOG_TRIVIAL(info) << "Starting Export!!";
			mControl->nvrExport(0, 5);
			mControl->step();
		}
	}
}

BOOST_AUTO_TEST_CASE(basic)
{
	CheckThread f;

	auto m1 = boost::shared_ptr<CheckThread::SourceModule>(new CheckThread::SourceModule(CheckThread::SourceModuleProps()));
	auto metadata1 = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	m1->addOutputPin(metadata1);
	auto m2 = boost::shared_ptr<CheckThread::TransformModule>(new CheckThread::TransformModule(CheckThread::TransformModuleProps()));
	m1->setNext(m2);
	auto metadata2 = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	m2->addOutputPin(metadata2);
	auto m3 = boost::shared_ptr<CheckThread::TransformModule>(new CheckThread::TransformModule(CheckThread::TransformModuleProps()));
	m2->setNext(m3);
	auto metadata3 = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	m3->addOutputPin(metadata3);
	auto m4 = boost::shared_ptr<CheckThread::SinkModule>(new CheckThread::SinkModule(CheckThread::SinkModuleProps()));
	m3->setNext(m4);
	auto mControl = boost::shared_ptr<NVRControlModule>(new NVRControlModule(NVRControlModuleProps()));

	PipeLine p("test");
	// add all source  modules
	p.appendModule(m1);
	// add control module if any
	p.addControlModule(mControl);
	mControl->enrollModule("source", m1);
	mControl->enrollModule("transform_1", m2);
	mControl->enrollModule("writer", m3);
	mControl->enrollModule("sink", m4);
	// init
	p.init();
	// control init - do inside pipeline init
	mControl->init();
	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(10));
	mControl->nvrView(false);
	// dont need step in run_all_threaded
	mControl->step();

	boost::this_thread::sleep_for(boost::chrono::seconds(10));
	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(checkNVR)
{
	auto nvrPipe = boost::shared_ptr<NVRPipeline>(new NVRPipeline());
	nvrPipe->open();
	//nvrPipe->startRecording();
	nvrPipe->close();
}

BOOST_AUTO_TEST_CASE(NVRTest)
{
	auto cuContext = apracucontext_sp(new ApraCUcontext());
	uint32_t gopLength = 25;
	uint32_t bitRateKbps = 1000;
	uint32_t frameRate = 30;
	H264EncoderNVCodecProps::H264CodecProfile profile = H264EncoderNVCodecProps::MAIN;
	bool enableBFrames = true;
	auto width = 1920;
	auto height = 1020;

	// test with 0 - with multiple cameras

	WebCamSourceProps webCamSourceprops(0, 1920, 1080);
	auto webCam = boost::shared_ptr<WebCamSource>(new WebCamSource(webCamSourceprops));
	auto colorConvt = boost::shared_ptr<ColorConversion>(new ColorConversion(ColorConversionProps(ColorConversionProps::ConversionType::RGB_TO_YUV420PLANAR)));
	webCam->setNext(colorConvt);
	cudastream_sp cudaStream_ = boost::shared_ptr<ApraCudaStream>(new ApraCudaStream());
	auto copyProps = CudaMemCopyProps(cudaMemcpyHostToDevice, cudaStream_);
	auto copy = boost::shared_ptr<Module>(new CudaMemCopy(copyProps));
	colorConvt->setNext(copy);
	auto encoder = boost::shared_ptr<H264EncoderNVCodec>(new H264EncoderNVCodec(H264EncoderNVCodecProps(bitRateKbps, cuContext, gopLength, frameRate, profile, enableBFrames)));
	copy->setNext(encoder);
	std::string outFolderPath = "./data/testOutput/mp4_videos/24bpp/";
	auto mp4WriterSinkProps = Mp4WriterSinkProps(1, 1, 24, outFolderPath);
	mp4WriterSinkProps.logHealth = true;
	mp4WriterSinkProps.logHealthFrequency = 10;
	auto mp4Writer = boost::shared_ptr<Mp4WriterSink>(new Mp4WriterSink(mp4WriterSinkProps));
	encoder->setNext(mp4Writer);
	auto mControl = boost::shared_ptr<NVRControlModule>(new NVRControlModule(NVRControlModuleProps()));

	PipeLine p("test");
	p.appendModule(webCam);
	// add control module if any
	p.addControlModule(mControl);
	mControl->enrollModule("source", webCam);
	mControl->enrollModule("colorConversion", colorConvt);
	mControl->enrollModule("cudaCopy", copy);
	mControl->enrollModule("encoder", encoder);
	mControl->enrollModule("writer", mp4Writer);
	// init
	p.init();
	// control init - do inside pipeline init
	mControl->init();
	p.run_all_threaded();
	mControl->nvrRecord(true);
	// dont need step in run_all_threaded
	mControl->step();

	boost::this_thread::sleep_for(boost::chrono::seconds(10));
	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(NVRView)
{
	WebCamSourceProps webCamSourceprops(0, 1920, 1080);
	auto webCam = boost::shared_ptr<WebCamSource>(new WebCamSource(webCamSourceprops));
	auto multique = boost::shared_ptr<MultimediaQueueXform>(new MultimediaQueueXform(MultimediaQueueXformProps(10000, 5000, true)));
	auto view = boost::shared_ptr<ImageViewerModule>(new ImageViewerModule(ImageViewerModuleProps("NVR-View")));
	webCam->setNext(multique);
	multique->setNext(view);
	auto mControl = boost::shared_ptr<NVRControlModule>(new NVRControlModule(NVRControlModuleProps()));

	PipeLine p("test");
	p.appendModule(webCam);
	boost::thread inp(key_func, mControl);
	p.addControlModule(mControl);
	mControl->enrollModule("filereader", webCam);
	mControl->enrollModule("viewer", view);

	p.init();
	mControl->init();
	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(30));
	p.stop();
	p.term();
	p.wait_for_all();
	inp.join();
}

BOOST_AUTO_TEST_CASE(NVRViewKey)
{
	WebCamSourceProps webCamSourceprops(0, 1920, 1080);
	auto webCam = boost::shared_ptr<WebCamSource>(new WebCamSource(webCamSourceprops));
	auto view = boost::shared_ptr<ImageViewerModule>(new ImageViewerModule(ImageViewerModuleProps("NVR-View")));
	webCam->setNext(view);
	auto mControl = boost::shared_ptr<NVRControlModule>(new NVRControlModule(NVRControlModuleProps()));

	PipeLine p("test");
	std::thread inp(key_func, mControl);
	p.appendModule(webCam);
	p.addControlModule(mControl);
	mControl->enrollModule("filereader", webCam);
	mControl->enrollModule("viewer", view);

	p.init();
	mControl->init();
	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(100));
	p.stop();
	p.term();
	p.wait_for_all();
	inp.join();
}


BOOST_AUTO_TEST_CASE(NVRFile)
{
	std::string inFolderPath = "./data/Raw_YUV420_640x360";
	auto fileReaderProps = FileReaderModuleProps(inFolderPath, 0, -1);
	fileReaderProps.fps = 20;
	fileReaderProps.readLoop = true;
	auto fileReader = boost::shared_ptr<Module>(new FileReaderModule(fileReaderProps)); //
	auto metadata = framemetadata_sp(new RawImageMetadata(640, 360, ImageMetadata::ImageType::MONO, CV_8UC1, 0, CV_8U, FrameMetadata::HOST, true));
	auto pinId = fileReader->addOutputPin(metadata);
	auto view = boost::shared_ptr<ImageViewerModule>(new ImageViewerModule(ImageViewerModuleProps("NVR-View")));
	fileReader->setNext(view);
	auto mControl = boost::shared_ptr<NVRControlModule>(new NVRControlModule(NVRControlModuleProps()));

	PipeLine p("test");
	p.appendModule(fileReader);
	p.addControlModule(mControl);
	mControl->enrollModule("filereader", fileReader);
	mControl->enrollModule("viewer", view);

	p.init();
	mControl->init();
	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(15));
	mControl->nvrView(false);
	mControl->step();
	boost::this_thread::sleep_for(boost::chrono::seconds(10));
	mControl->nvrView(true);
	mControl->step();
	boost::this_thread::sleep_for(boost::chrono::seconds(15));

	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(NVRkey)
{
	std::string inFolderPath = "./data/h264_data";
	auto fileReaderProps = FileReaderModuleProps(inFolderPath, 0, -1);
	fileReaderProps.fps = 20;
	fileReaderProps.readLoop = true;
	auto fileReader = boost::shared_ptr<Module>(new FileReaderModule(fileReaderProps)); //
	auto encodedImageMetadata = framemetadata_sp(new H264Metadata(704, 576)); 
	auto pinId = fileReader->addOutputPin(encodedImageMetadata);
	//auto view = boost::shared_ptr<ImageViewerModule>(new ImageViewerModule(ImageViewerModuleProps("NVR-View")));
	std::string outFolderPath_1 = "./data/testOutput/mp4_videos/24bpp/";
	auto mp4WriterSinkProps_1 = Mp4WriterSinkProps(1, 1, 24, outFolderPath_1);
	mp4WriterSinkProps_1.logHealth = true;
	mp4WriterSinkProps_1.logHealthFrequency = 10;
	auto mp4Writer_1 = boost::shared_ptr<Mp4WriterSink>(new Mp4WriterSink(mp4WriterSinkProps_1));
	fileReader->setNext(mp4Writer_1);
	std::string outFolderPath_2 = "./data/testOutput/mp4_videos/ExportVids/";
	auto mp4WriterSinkProps_2 = Mp4WriterSinkProps(1, 1, 24, outFolderPath_2);
	mp4WriterSinkProps_2.logHealth = true;
	mp4WriterSinkProps_2.logHealthFrequency = 10;
	auto mp4Writer_2 = boost::shared_ptr<Mp4WriterSink>(new Mp4WriterSink(mp4WriterSinkProps_2));
	//fileReader->setNext(mp4Writer_2);
	auto mControl = boost::shared_ptr<NVRControlModule>(new NVRControlModule(NVRControlModuleProps()));

	PipeLine p("test");
	//std::thread inp(key_func, mControl);
	p.appendModule(fileReader);
	p.addControlModule(mControl);
	mControl->enrollModule("filereader", fileReader);
	mControl->enrollModule("writer", mp4Writer_1);

	p.init();
	mControl->init();
	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(10));
	p.stop();
	p.term();
	p.wait_for_all();
	//inp.join();
}

BOOST_AUTO_TEST_CASE(NVR_mmq)
{
	std::string inFolderPath = "./data/h264_data";
	auto fileReaderProps = FileReaderModuleProps(inFolderPath, 0, -1);
	fileReaderProps.fps = 20;
	fileReaderProps.readLoop = true;
	auto fileReader = boost::shared_ptr<Module>(new FileReaderModule(fileReaderProps)); //
	auto encodedImageMetadata = framemetadata_sp(new H264Metadata(704, 576));
	auto pinId = fileReader->addOutputPin(encodedImageMetadata);

	auto multiQueue = boost::shared_ptr<MultimediaQueueXform>(new MultimediaQueueXform(MultimediaQueueXformProps(30000, 5000, true)));
	fileReader->setNext(multiQueue);

	std::string outFolderPath_1 = "./data/testOutput/mp4_videos/24bpp/";
	auto mp4WriterSinkProps_1 = Mp4WriterSinkProps(1, 1, 24, outFolderPath_1);
	mp4WriterSinkProps_1.logHealth = true;
	mp4WriterSinkProps_1.logHealthFrequency = 10;
	auto mp4Writer_1 = boost::shared_ptr<Mp4WriterSink>(new Mp4WriterSink(mp4WriterSinkProps_1));
	multiQueue->setNext(mp4Writer_1);

	auto mControl = boost::shared_ptr<NVRControlModule>(new NVRControlModule(NVRControlModuleProps()));

	PipeLine p("test");
	std::thread inp(key_func, mControl);
	p.appendModule(fileReader);
	p.addControlModule(mControl);
	mControl->enrollModule("filereader", fileReader);
	mControl->enrollModule("multimediaQueue", multiQueue);
	mControl->enrollModule("writer", mp4Writer_1);

	p.init();
	mControl->init();
	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(50));
	p.stop();
	p.term();
	p.wait_for_all();
	inp.join();
}

BOOST_AUTO_TEST_CASE(NVR_mmq_view)
{
	auto cuContext = apracucontext_sp(new ApraCUcontext());
	uint32_t gopLength = 25;
	uint32_t bitRateKbps = 1000;
	uint32_t frameRate = 30;
	H264EncoderNVCodecProps::H264CodecProfile profile = H264EncoderNVCodecProps::MAIN;
	bool enableBFrames = true;
	auto width = 1920;
	auto height = 1020;


	WebCamSourceProps webCamSourceprops(0, 1920, 1080);
	auto webCam = boost::shared_ptr<WebCamSource>(new WebCamSource(webCamSourceprops));
	auto colorConvt = boost::shared_ptr<ColorConversion>(new ColorConversion(ColorConversionProps(ColorConversionProps::ConversionType::RGB_TO_YUV420PLANAR)));
	webCam->setNext(colorConvt);

	auto colorConvtView = boost::shared_ptr<ColorConversion>(new ColorConversion(ColorConversionProps(ColorConversionProps::ConversionType::RGB_TO_BGR)));
	webCam->setNext(colorConvtView);

	auto view = boost::shared_ptr<ImageViewerModule>(new ImageViewerModule(ImageViewerModuleProps("NVR-View")));
	colorConvtView->setNext(view);

	cudastream_sp cudaStream_ = boost::shared_ptr<ApraCudaStream>(new ApraCudaStream());
	auto copyProps = CudaMemCopyProps(cudaMemcpyHostToDevice, cudaStream_);
	auto copy = boost::shared_ptr<Module>(new CudaMemCopy(copyProps));
	colorConvt->setNext(copy);

	auto encoder = boost::shared_ptr<H264EncoderNVCodec>(new H264EncoderNVCodec(H264EncoderNVCodecProps(bitRateKbps, cuContext, gopLength, frameRate, profile, enableBFrames)));
	copy->setNext(encoder);


	auto multiQueue = boost::shared_ptr<MultimediaQueueXform>(new MultimediaQueueXform(MultimediaQueueXformProps(30000, 5000, true)));
	encoder->setNext(multiQueue);

	std::string outFolderPath_1 = "./data/testOutput/mp4_videos/24bpp/";
	auto mp4WriterSinkProps_1 = Mp4WriterSinkProps(1, 1, 24, outFolderPath_1);
	mp4WriterSinkProps_1.logHealth = true;
	mp4WriterSinkProps_1.logHealthFrequency = 10;
	auto mp4Writer_1 = boost::shared_ptr<Mp4WriterSink>(new Mp4WriterSink(mp4WriterSinkProps_1));
	multiQueue->setNext(mp4Writer_1);

	auto mControl = boost::shared_ptr<NVRControlModule>(new NVRControlModule(NVRControlModuleProps()));

	PipeLine p("test");
	std::thread inp(key_func, mControl);
	p.appendModule(webCam);
	p.addControlModule(mControl);
	mControl->enrollModule("webcamera", webCam);
	mControl->enrollModule("multimediaQueue", multiQueue);
	mControl->enrollModule("writer", mp4Writer_1);

	p.init();
	mControl->init();
	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(60));
	p.stop();
	p.term();
	p.wait_for_all();
	inp.join();
}

BOOST_AUTO_TEST_CASE(checkNVR2) //Use this for testing pipeline note - Only one mp4Writer is present in this pipeline 
{
	auto cuContext = apracucontext_sp(new ApraCUcontext());
	uint32_t gopLength = 25;
	uint32_t bitRateKbps = 1000;
	uint32_t frameRate = 30;
	H264EncoderNVCodecProps::H264CodecProfile profile = H264EncoderNVCodecProps::MAIN;
	bool enableBFrames = true;
	auto width = 640;
	auto height = 360;


	WebCamSourceProps webCamSourceprops(0, 1920, 1080);
	auto webCam = boost::shared_ptr<WebCamSource>(new WebCamSource(webCamSourceprops));
	auto colorConvt = boost::shared_ptr<ColorConversion>(new ColorConversion(ColorConversionProps(ColorConversionProps::ConversionType::RGB_TO_YUV420PLANAR)));
	webCam->setNext(colorConvt);

	cudastream_sp cudaStream_ = boost::shared_ptr<ApraCudaStream>(new ApraCudaStream());
	auto copyProps = CudaMemCopyProps(cudaMemcpyHostToDevice, cudaStream_);
	auto copy = boost::shared_ptr<Module>(new CudaMemCopy(copyProps));
	colorConvt->setNext(copy);
	auto colorConvtView = boost::shared_ptr<ColorConversion>(new ColorConversion(ColorConversionProps(ColorConversionProps::ConversionType::RGB_TO_BGR)));
	webCam->setNext(colorConvtView);
	auto view = boost::shared_ptr<ImageViewerModule>(new ImageViewerModule(ImageViewerModuleProps("NVR-View")));
	colorConvtView->setNext(view);
	H264EncoderNVCodecProps encProps(bitRateKbps, cuContext, gopLength, frameRate, profile, enableBFrames);
	auto encoder = boost::shared_ptr<H264EncoderNVCodec>(new H264EncoderNVCodec(encProps));
	copy->setNext(encoder);

	auto multiQueue = boost::shared_ptr<MultimediaQueueXform>(new MultimediaQueueXform(MultimediaQueueXformProps(10000, 5000, true)));
	encoder->setNext(multiQueue);

	std::string outFolderPath_1 = "./data/testOutput/mp4_videos/24bpp/";
	auto mp4WriterSinkProps_1 = Mp4WriterSinkProps(1, 10, 24, outFolderPath_1);
	mp4WriterSinkProps_1.logHealth = true;
	mp4WriterSinkProps_1.logHealthFrequency = 10;
	auto mp4Writer_1 = boost::shared_ptr<Mp4WriterSink>(new Mp4WriterSink(mp4WriterSinkProps_1));
	multiQueue->setNext(mp4Writer_1);

	auto mControl = boost::shared_ptr<NVRControlModule>(new NVRControlModule(NVRControlModuleProps()));

	PipeLine p("test");
	std::thread inp(key_func, mControl);
	p.appendModule(webCam);
	p.addControlModule(mControl);
	mControl->enrollModule("WebCamera", webCam);
	mControl->enrollModule("Renderer", view);
	mControl->enrollModule("Writer-1", mp4Writer_1);
	mControl->enrollModule("MultimediaQueue", multiQueue);

	p.init();
	mControl->init();
	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(360));
	p.stop();
	p.term();
	p.wait_for_all();
	BOOST_LOG_TRIVIAL(info) << "The first thread has stopped";
	inp.join();
}

BOOST_AUTO_TEST_CASE(checkNVR3) //Use this for testing pipeline note - Mimics the actual pipeline
{
	auto cuContext = apracucontext_sp(new ApraCUcontext());
	uint32_t gopLength = 25;
	uint32_t bitRateKbps = 1000;
	uint32_t frameRate = 30;
	H264EncoderNVCodecProps::H264CodecProfile profile = H264EncoderNVCodecProps::MAIN;
	bool enableBFrames = true;
	auto width = 640; //1920
	auto height = 360; //1020


	WebCamSourceProps webCamSourceprops(0, 640, 360);
	auto webCam = boost::shared_ptr<WebCamSource>(new WebCamSource(webCamSourceprops));
	auto colorConvt = boost::shared_ptr<ColorConversion>(new ColorConversion(ColorConversionProps(ColorConversionProps::ConversionType::RGB_TO_YUV420PLANAR)));
	webCam->setNext(colorConvt);

	auto colorConvtView = boost::shared_ptr<ColorConversion>(new ColorConversion(ColorConversionProps(ColorConversionProps::ConversionType::RGB_TO_BGR)));
	webCam->setNext(colorConvtView);

	auto view = boost::shared_ptr<ImageViewerModule>(new ImageViewerModule(ImageViewerModuleProps("NVR-View")));
	colorConvtView->setNext(view);

	cudastream_sp cudaStream_ = boost::shared_ptr<ApraCudaStream>(new ApraCudaStream());
	auto copyProps = CudaMemCopyProps(cudaMemcpyHostToDevice, cudaStream_);
	auto copy = boost::shared_ptr<Module>(new CudaMemCopy(copyProps));
	colorConvt->setNext(copy);

	auto encoder = boost::shared_ptr<H264EncoderNVCodec>(new H264EncoderNVCodec(H264EncoderNVCodecProps(bitRateKbps, cuContext, gopLength, frameRate, profile, enableBFrames)));
	copy->setNext(encoder);

	std::string outFolderPath_1 = "./data/testOutput/mp4_videos/24bpp/";
	auto mp4WriterSinkProps_1 = Mp4WriterSinkProps(1, 10, 24, outFolderPath_1);
	mp4WriterSinkProps_1.logHealth = true;
	mp4WriterSinkProps_1.logHealthFrequency = 10;
	auto mp4Writer_1 = boost::shared_ptr<Mp4WriterSink>(new Mp4WriterSink(mp4WriterSinkProps_1));
	encoder->setNext(mp4Writer_1);

	auto multiQue = boost::shared_ptr<MultimediaQueueXform>(new MultimediaQueueXform(MultimediaQueueXformProps(1000, 1020, false)));
	encoder->setNext(multiQue);
	std::string outFolderPath_2 = "./data/testOutput/mp4_videos/ExportVids/";
	auto mp4WriterSinkProps_2 = Mp4WriterSinkProps(1, 1, 24, outFolderPath_2);
	mp4WriterSinkProps_2.logHealth = true;
	mp4WriterSinkProps_2.logHealthFrequency = 10;
	auto mp4Writer_2 = boost::shared_ptr<Mp4WriterSink>(new Mp4WriterSink(mp4WriterSinkProps_2));
	multiQue->setNext(mp4Writer_2);

	auto mControl = boost::shared_ptr<NVRControlModule>(new NVRControlModule(NVRControlModuleProps()));
	PipeLine p("test");
	std::thread inp(key_func, mControl);
	p.appendModule(webCam);
	p.addControlModule(mControl);
	mControl->enrollModule("WebCamera", webCam);
	mControl->enrollModule("Renderer", view);
	mControl->enrollModule("Writer-1", mp4Writer_1);
	mControl->enrollModule("MultimediaQueue", multiQue);
	mControl->enrollModule("Writer-2", mp4Writer_2);

	p.init();
	mControl->init();
	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(360));
	p.stop();
	p.term();
	p.wait_for_all();
	BOOST_LOG_TRIVIAL(info) << "The first thread has stopped";
	inp.join();
}


BOOST_AUTO_TEST_CASE(NVR_mmq_view_mp4Write)
{
	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	Logger::initLogger(loggerProps);
	auto cuContext = apracucontext_sp(new ApraCUcontext());
	uint32_t gopLength = 25;
	uint32_t bitRateKbps = 1000;
	uint32_t frameRate = 30;
	H264EncoderNVCodecProps::H264CodecProfile profile = H264EncoderNVCodecProps::MAIN;
	bool enableBFrames = true;
	auto width = 640;
	auto height = 360;


	FileReaderModuleProps fileReaderProps("./data/Raw_YUV420_640x360/Image???_YUV420.raw");
	fileReaderProps.fps = 20;
	fileReaderProps.readLoop = true;
	
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U, FrameMetadata::MemType::HOST));
	fileReader->addOutputPin(metadata);
	//std::string inFolderPath = "./data/h264_data";
	//auto fileReaderProps = FileReaderModuleProps(inFolderPath, 0, -1);
	//fileReaderProps.fps = 20;
	//fileReaderProps.readLoop = true;
	//auto fileReader = boost::shared_ptr<Module>(new FileReaderModule(fileReaderProps)); //
	//auto encodedImageMetadata = framemetadata_sp(new H264Metadata(704, 576));
	//auto pinId = fileReader->addOutputPin(encodedImageMetadata);

	//auto colorConvt = boost::shared_ptr<ColorConversion>(new ColorConversion(ColorConversionProps(ColorConversionProps::ConversionType::YUV420PLANAR_TO_RGB)));

	//auto colorConvtView = boost::shared_ptr<ColorConversion>(new ColorConversion(ColorConversionProps(ColorConversionProps::ConversionType::YUV420PLANAR_TO_RGB)));
	//fileReader->setNext(colorConvtView);

	//auto view = boost::shared_ptr<ImageViewerModule>(new ImageViewerModule(ImageViewerModuleProps("NVR-View")));
	//colorConvtView->setNext(view);

	cudastream_sp cudaStream_ = boost::shared_ptr<ApraCudaStream>(new ApraCudaStream());
	auto copyProps = CudaMemCopyProps(cudaMemcpyHostToDevice, cudaStream_);
	//auto copyProps = CudaMemCopyProps(cudaMemcpyDeviceToHost, cudaStream_);
	auto copy = boost::shared_ptr<Module>(new CudaMemCopy(copyProps));
	//fileReader->setNext(colorConvt);
	fileReader->setNext(copy);

	H264EncoderNVCodecProps encProps(bitRateKbps, cuContext, gopLength, frameRate, profile, enableBFrames);
	auto encoder = boost::shared_ptr<H264EncoderNVCodec>(new H264EncoderNVCodec(encProps));
	copy->setNext(encoder);

	//auto valve = boost::shared_ptr<ValveModule>(new ValveModule(ValveModuleProps(0)));
	//encoder->setNext(valve);
	MultimediaQueueXformProps multProps(1500, 2000, false);
	auto multiQueue = boost::shared_ptr<MultimediaQueueXform>(new MultimediaQueueXform(multProps));
	encoder->setNext(multiQueue);

	std::string outFolderPath_1 = "./data/testOutput/mp4_videos/24bpp/";
	auto mp4WriterSinkProps_1 = Mp4WriterSinkProps(1, 1, 5, outFolderPath_1);
	mp4WriterSinkProps_1.logHealth = true;
	mp4WriterSinkProps_1.logHealthFrequency = 10;
	auto mp4Writer_1 = boost::shared_ptr<Mp4WriterSink>(new Mp4WriterSink(mp4WriterSinkProps_1));
	auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/h264images/Raw_YUV420_640x360????.h264")));
	multiQueue->setNext(mp4Writer_1);
	//auto mControl = boost::shared_ptr<NVRControlModule>(new NVRControlModule(NVRControlModuleProps()));

	PipeLine p("test");
	//std::thread inp(key_func, mControl);
	p.appendModule(fileReader);
	//p.addControlModule(mControl);
	//mControl->enrollModule("webcamera", fileReader);
	//mControl->enrollModule("multimediaQueue", multiQueue);
	//mControl->enrollModule("writer", mp4Writer_1);

	p.init();
	//mControl->init();
	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(180));
	p.stop();
	p.term();
	p.wait_for_all();
	BOOST_LOG_TRIVIAL(info) << "The first thread has stopped";
	//inp.join();
}

BOOST_AUTO_TEST_SUITE_END()