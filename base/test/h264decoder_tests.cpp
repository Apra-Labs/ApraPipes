#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include "FileReaderModule.h"
#include "FileWriterModule.h"
#include "Logger.h"
#include "H264Decoder.h"
#include "test_utils.h"
#include "DMAFDToHostCopy.h"
#include "PipeLine.h"
#include "ExternalSinkModule.h"
#include "H264Metadata.h"
#include "Mp4ReaderSource.h"
#include "Mp4VideoMetadata.h"
#include "StatSink.h"
#include "CudaStreamSynchronize.h"
#include "RTSPClientSrc.h"
#ifdef ARM64
#include "EglRenderer.h"
#include "AffineTransform.h"
#include "NvTransform.h"

#else
#include "CudaMemCopy.h"
#include "nv_test_utils.h"
#endif

BOOST_AUTO_TEST_SUITE(h264decoder_tests)

#ifdef ARM64

// struct rtsp_client_tests_data {
//     rtsp_client_tests_data()
//     {
//         outFile = string("./data/testOutput/bunny.h264");
//         Test_Utils::FileCleaner fc;
//         fc.pathsOfFiles.push_back(outFile); //clear any occurance before starting the tests
//     }
//     string outFile;
//     string empty;
// };
// BOOST_AUTO_TEST_CASE(rtsp_case3)
// {
//     rtsp_client_tests_data d;
//     auto url=string("rtsp://evo-dev-apra.blub0xSecurity.com:5544/76174d56-fdc0-4a2e-aad6-981f4f7f71ea");
//     auto rtspProps = RTSPClientSrcProps(url, d.empty, d.empty);
//     rtspProps.fps = 18;
//     auto mLive = boost::shared_ptr<Module>(new RTSPClientSrc(rtspProps));
//     auto meta = framemetadata_sp(new H264Metadata());
//     mLive->addOutputPin(meta);
//     auto Decoder = boost::shared_ptr<Module>(new H264Decoder(H264DecoderProps()));
//     std::vector<std::string> mImagePin;
//     mImagePin = mLive->getAllOutputPinsByType(FrameMetadata::FrameType::H264_DATA);
//     mLive->setNext(Decoder, mImagePin);
//     auto copySource = boost::shared_ptr<Module>(new DMAFDToHostCopy);
//     Decoder->setNext(copySource);
//     auto writer1 = boost::shared_ptr<FileWriterModule>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/1/webrtcFrame_????.raw")));
//     copySource->setNext(writer1);
//     boost::shared_ptr<PipeLine> p;
//     p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
//     p->appendModule(mLive);
//     if (!p->init())
//     {
//         throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
//     }
//     p->run_all_threaded();
//     Test_Utils::sleep_for_seconds(5);
//     p->stop();
//     p->term();
//     p->wait_for_all();
//     p.reset();
// }
BOOST_AUTO_TEST_CASE(atl_test_pipeline,* boost::unit_test::disabled())
{
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);
	std::string videoPath = "/home/developer/workspace/ApraPipes/1684307720.mp4";
	auto stream = cudastream_sp(new ApraCudaStream);

	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, false);
	auto m_reviewSource = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	m_reviewSource->addOutPutPin(h264ImageMetadata);

	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	m_reviewSource->addOutPutPin(mp4Metadata);

	StatSinkProps sinkProps;
	sinkProps.logHealth = true;
	sinkProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new StatSink(sinkProps));
	m_reviewSource->setNext(sink);

	auto m_h264Decode = boost::shared_ptr<H264Decoder>(new H264Decoder(H264DecoderProps()));
	std::vector<std::string> mImagePin;
	mImagePin = m_reviewSource->getAllOutputPinsByType(FrameMetadata::FrameType::H264_DATA);
	// m_reviewSource->setNext(m_h264Decode, mImagePin);

    auto m_nv12_to_yuv444Transform = boost::shared_ptr<NvTransform>(new NvTransform(NvTransformProps(ImageMetadata::RGBA)));
	m_h264Decode->setNext(m_nv12_to_yuv444Transform);

	AffineTransformProps affineProps(AffineTransformProps::CUBIC, stream, 0,4096, 0, 0, 1);
	affineProps.qlen = 1;
	affineProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto m_reviewAffineTransform = boost::shared_ptr<AffineTransform>(new AffineTransform(affineProps));
	m_nv12_to_yuv444Transform->setNext(m_reviewAffineTransform);

	auto sync = boost::shared_ptr<CudaStreamSynchronize>(new CudaStreamSynchronize(CudaStreamSynchronizeProps(stream)));
	m_reviewAffineTransform->setNext(sync);

	EglRendererProps eglProps(455, 38, 1000, 1000);
	eglProps.qlen = 2;
	eglProps.fps = 20;
	eglProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto m_review_renderer = boost::shared_ptr<EglRenderer>(new EglRenderer(eglProps));
	sync->setNext(m_review_renderer);

	// m_playbackPipeline.appendModule(m_reviewSource);
	// m_playbackPipeline.init();
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(m_reviewSource);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	p->run_all_threaded();
}


BOOST_AUTO_TEST_CASE(memory_leak_check,* boost::unit_test::disabled())
{

	boost::shared_ptr<PipeLine> m_playbackPipeline;
	m_playbackPipeline = boost::shared_ptr<PipeLine>(new PipeLine("test"));

	std::vector<std::string> mediaList = {
		"/media/developer/1250328450326F1B/Videos_to_test/2023-10-31/D1/P1/2023-10-31_12-42-39-913.mp4",
		"/media/developer/1250328450326F1B/Videos_to_test/2023-10-31/D1/P1/2023-10-31_12-42-55-845.mp4",
		"/media/developer/1250328450326F1B/Videos_to_test/2023-10-31/D1/P1/2023-10-31_12-50-53-781.mp4",
		"/media/developer/1250328450326F1B/Videos_to_test/2023-10-31/D1/P1/2023-10-31_12-51-12-751.mp4",
		"/media/developer/1250328450326F1B/Videos_to_test/2023-10-31/D1/P1/2023-10-31_12-51-34-695.mp4"
	};

	auto stream = cudastream_sp(new ApraCudaStream);
	auto mp4ReaderProps = Mp4ReaderSourceProps(mediaList[0], false);
	boost::shared_ptr<Mp4ReaderSource> m_reviewSource = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	m_reviewSource->addOutPutPin(h264ImageMetadata);

	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	m_reviewSource->addOutPutPin(mp4Metadata);

	
	StatSinkProps sinkProps;
	sinkProps.logHealth = true;
	sinkProps.logHealthFrequency = 100;
	auto m3 = boost::shared_ptr<Module>(new StatSink(sinkProps));
	m_reviewSource->setNext(m3);
	
	auto decoderProps = H264DecoderProps();
	boost::shared_ptr<H264Decoder> m_h264Decode = boost::shared_ptr<H264Decoder>(new H264Decoder(decoderProps));
	std::vector<std::string> mImagePin;
	mImagePin = m_reviewSource->getAllOutputPinsByType(FrameMetadata::FrameType::H264_DATA);
	m_reviewSource->setNext(m_h264Decode, mImagePin);

	EglRendererProps eglProps(455, 800, 400, 400);
	boost::shared_ptr<EglRenderer> m_review_renderer = boost::shared_ptr<EglRenderer>(new EglRenderer(eglProps));
	m_h264Decode->setNext(m_review_renderer);

	m_playbackPipeline->appendModule(m_reviewSource);
	m_playbackPipeline->init();
	m_playbackPipeline->run_all_threaded();

	while (true)
	{

		LOG_ERROR << "TOTAL NUMBER OF FILE TO PLAY" << mediaList.size(); 
		for (int i = 0; i < mediaList.size(); i++)
		{
			LOG_ERROR << "<==========================    PLAYING =================================>>>>>>>>>>>>>>>>>>>>" << mediaList[i];
			auto currMediaProps = m_reviewSource->getProps();
			currMediaProps.videoPath = mediaList[i]; //"/home/developer/workspace/ApraPipes/1684824632.mp4";
			m_reviewSource->setProps(currMediaProps);
			// m_review_renderer->createWindow(1000, 1000);
			boost::this_thread::sleep_for(boost::chrono::seconds(15));
			// m_review_renderer->closeWindow();
			// boost::this_thread::sleep_for(boost::chrono::seconds(1));
		}
	}

	Test_Utils::sleep_for_seconds(15000);
}

BOOST_AUTO_TEST_CASE(fastPlayback,* boost::unit_test::disabled())
{
	boost::shared_ptr<PipeLine> m_playbackPipeline;
	m_playbackPipeline = boost::shared_ptr<PipeLine>(new PipeLine("test"));

	std::vector<std::string> mediaList = {
		"/home/vivek/2023-12-12_16-34-08-799.mp4"};

	auto stream = cudastream_sp(new ApraCudaStream);
	auto mp4ReaderProps = Mp4ReaderSourceProps(mediaList[0], false);
	// mp4ReaderProps.fps = 30;
	mp4ReaderProps.logHealth = true;
	mp4ReaderProps.logHealthFrequency = 100;
	mp4ReaderProps.quePushStrategyType = QuePushStrategy::BLOCKING;
	boost::shared_ptr<Mp4ReaderSource> m_reviewSource = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	m_reviewSource->addOutPutPin(h264ImageMetadata);

	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	m_reviewSource->addOutPutPin(mp4Metadata);

	auto decoderProps = H264DecoderProps();
	decoderProps.quePushStrategyType = QuePushStrategy::BLOCKING;
	boost::shared_ptr<H264Decoder> m_h264Decode = boost::shared_ptr<H264Decoder>(new H264Decoder(decoderProps));
	std::vector<std::string> mImagePin;
	mImagePin = m_reviewSource->getAllOutputPinsByType(FrameMetadata::FrameType::H264_DATA);
	m_reviewSource->setNext(m_h264Decode, mImagePin);

	EglRendererProps eglProps(0, 0, 1000, 1000);
	eglProps.fps = 30;
	eglProps.quePushStrategyType = QuePushStrategy::BLOCKING;
	boost::shared_ptr<EglRenderer> m_review_renderer = boost::shared_ptr<EglRenderer>(new EglRenderer(eglProps));
	m_h264Decode->setNext(m_review_renderer);
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);
	m_playbackPipeline->appendModule(m_reviewSource);
	m_playbackPipeline->init();
	m_playbackPipeline->run_all_threaded();
	Logger::setLogLevel(boost::log::trivial::severity_level::debug);
	
	boost::this_thread::sleep_for(boost::chrono::seconds(10));
	m_review_renderer->play(false, true);
	m_reviewSource->play(false, true);

	boost::this_thread::sleep_for(boost::chrono::seconds(10));
    // auto rederProps = m_reviewSource->getProps();
	// m_reviewSource->setProps(rederProps);
	m_review_renderer->play(true, true);
	m_reviewSource->play(true, true);
	boost::this_thread::sleep_for(boost::chrono::seconds(10));

	m_review_renderer->play(false, true);
	m_reviewSource->play(false, true);

	boost::this_thread::sleep_for(boost::chrono::seconds(10));
	m_review_renderer->play(true, true);
	m_reviewSource->play(true, true);

	Test_Utils::sleep_for_seconds(15000);
}

void myCallbackFunction()
{
    // Your callback logic here
    LOG_ERROR << "Callback function triggered!";
}

BOOST_AUTO_TEST_CASE(rotateRecordedClip,* boost::unit_test::disabled())
{

	boost::shared_ptr<PipeLine> m_playbackPipeline;
	m_playbackPipeline = boost::shared_ptr<PipeLine>(new PipeLine("test"));

	std::vector<std::string> mediaList = {
		"/home/vivek/apra_test/Videos_to_test/2023-10-31/D1/P1/2023-10-31_12-42-39-913.mp4"
	};

	auto stream = cudastream_sp(new ApraCudaStream);
	auto mp4ReaderProps = Mp4ReaderSourceProps(mediaList[0], false);
	boost::shared_ptr<Mp4ReaderSource> m_reviewSource = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	m_reviewSource->registerCallback(myCallbackFunction);
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	m_reviewSource->addOutPutPin(h264ImageMetadata);

	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	m_reviewSource->addOutPutPin(mp4Metadata);
	
	auto decoderProps = H264DecoderProps();
	boost::shared_ptr<H264Decoder> m_h264Decode = boost::shared_ptr<H264Decoder>(new H264Decoder(decoderProps));
	std::vector<std::string> mImagePin;
	mImagePin = m_reviewSource->getAllOutputPinsByType(FrameMetadata::FrameType::H264_DATA);
	m_reviewSource->setNext(m_h264Decode, mImagePin);

	auto nv_transform = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::RGBA))); // DMA
	m_h264Decode->setNext(nv_transform);

	AffineTransformProps affineProps(AffineTransformProps::CUBIC, stream, 0, 4096, 0, 0, 1.0f);
	affineProps.qlen = 1;
	affineProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	auto affine = boost::shared_ptr<AffineTransform>(new AffineTransform(affineProps));
	nv_transform->setNext(affine);

	EglRendererProps eglProps(0, 0, 1000, 1000);
	boost::shared_ptr<EglRenderer> m_review_renderer = boost::shared_ptr<EglRenderer>(new EglRenderer(eglProps));
	affine->setNext(m_review_renderer);

	m_playbackPipeline->appendModule(m_reviewSource);
	m_playbackPipeline->init();
	m_playbackPipeline->run_all_threaded();

	// while (true)
	// {

	// 	LOG_ERROR << "TOTAL NUMBER OF FILE TO PLAY" << mediaList.size(); 
	// 	// for (int i = 0; i < mediaList.size(); i++)
	// 	// {
	// 	// 	LOG_ERROR << "<==========================    PLAYING =================================>>>>>>>>>>>>>>>>>>>>" << mediaList[i];
	// 	// 	auto currMediaProps = m_reviewSource->getProps();
	// 	// 	currMediaProps.videoPath = mediaList[i]; //"/home/developer/workspace/ApraPipes/1684824632.mp4";
	// 	// 	m_reviewSource->setProps(currMediaProps);
	// 	// 	// m_review_renderer->createWindow(1000, 1000);
	// 	// 	boost::this_thread::sleep_for(boost::chrono::seconds(1));
	// 	// 	auto affineProps = affine->getProps();
	// 	// 	affineProps.angle =90;
	// 	// 	affine->setProps(affineProps);
	// 	// 	boost::this_thread::sleep_for(boost::chrono::seconds(1));
	// 	// 	affineProps = affine->getProps();
	// 	// 	affineProps.angle =180;
	// 	// 	affine->setProps(affineProps);
	// 	// 	boost::this_thread::sleep_for(boost::chrono::seconds(1));
	// 	// 	affineProps = affine->getProps();
	// 	// 	affineProps.angle =270;
	// 	// 	affine->setProps(affineProps);
	// 	// 	boost::this_thread::sleep_for(boost::chrono::seconds(1));
	// 	// 	affineProps = affine->getProps();
	// 	// 	affineProps.angle =0;
	// 	// 	affine->setProps(affineProps);
	// 	// 	boost::this_thread::sleep_for(boost::chrono::seconds(1));
	// 	// 	// m_review_renderer->closeWindow();
	// 	// 	// boost::this_thread::sleep_for(boost::chrono::seconds(1));
	// 	// }
	// }

	Test_Utils::sleep_for_seconds(15000);
}

// BOOST_AUTO_TEST_CASE(mp4reader_decoder_eglrenderer,* boost::unit_test::disabled())
// {
// 	Logger::setLogLevel("info");

// 	// metadata is known
// 	std::string videoPath = "/home/developer/mp4_data/newatl.mp4";
// 	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, false);
// 	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
// 	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
// 	mp4Reader->addOutPutPin(h264ImageMetadata);

// 	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
// 	mp4Reader->addOutPutPin(mp4Metadata);

// 	auto Decoder = boost::shared_ptr<Module>(new H264Decoder(H264DecoderProps()));
// 	std::vector<std::string> mImagePin;
// 	mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::FrameType::H264_DATA);
// 	mp4Reader->setNext(Decoder, mImagePin);

// 	auto nv_transform = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::RGBA))); // DMA
// 	Decoder->setNext(nv_transform);

// 	auto stream = cudastream_sp(new ApraCudaStream);
// 	AffineTransformProps affineProps(AffineTransformProps::CUBIC, stream, 10, 45, 0, 1.0f);
// 	affineProps.qlen = 1;
// 	affineProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
// 	auto affine = boost::shared_ptr<AffineTransform>(new AffineTransform(affineProps));
// 	nv_transform->setNext(affine);

// 	auto sync = boost::shared_ptr<Module>(new CudaStreamSynchronize(CudaStreamSynchronizeProps(stream)));
// 	affine->setNext(sync);	

// 	auto sink = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(0, 0)));
// 	sync->setNext(sink);

// 	boost::shared_ptr<PipeLine> p;
// 	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
// 	p->appendModule(mp4Reader);

// 	if (!p->init())
// 	{
// 		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
// 	}

// 	p->run_all_threaded();

// 	 Test_Utils::sleep_for_seconds(15000);

// 	// p->stop();
// 	// p->term();
// 	// p->wait_for_all();
// 	// p.reset();
// }

BOOST_AUTO_TEST_CASE(mp4reader_decoder_eglrenderer,* boost::unit_test::disabled())
{
	Logger::setLogLevel("info");
	auto stream = cudastream_sp(new ApraCudaStream);
	// metadata is known
	std::string videoPath = "/home/developer/workspace/ApraPipes/1684824632.mp4";
	// std::string videoPath = "/media/developer/7C3B-7A0B/2023-07-11/DOCTOR/PATIENT/2023-07-11_16-14-50-880.mp4";
	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, false);
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	mp4Reader->addOutPutPin(h264ImageMetadata);

	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	// StatSinkProps sinkProps;
	// sinkProps.logHealth = true;
	// sinkProps.logHealthFrequency = 100;
	// auto sink2 = boost::shared_ptr<Module>(new StatSink(sinkProps));
	// mp4Reader->setNext(sink2);

	auto Decoder = boost::shared_ptr<H264Decoder>(new H264Decoder(H264DecoderProps()));
	std::vector<std::string> mImagePin;
	mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::FrameType::H264_DATA);
	mp4Reader->setNext(Decoder, mImagePin);

	//Adding transform

	auto nv_transform = boost::shared_ptr<Module>(new NvTransform(NvTransformProps(ImageMetadata::RGBA))); // DMA
	Decoder->setNext(nv_transform);

	// AffineTransformProps affineProps(AffineTransformProps::CUBIC, stream, 15 ,4096, 0, 0, 1);
	// affineProps.qlen = 1;
	// affineProps.quePushStrategyType = QuePushStrategy::NON_BLOCKING_ANY;
	// auto m_reviewAffineTransform = boost::shared_ptr<AffineTransform>(new AffineTransform(affineProps));
	// nv_transform->setNext(m_reviewAffineTransform);


	auto sink = boost::shared_ptr<EglRenderer>(new EglRenderer(EglRendererProps(0,0)));
	nv_transform->setNext(sink);

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(mp4Reader);


	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	p->run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(2));
	LOG_ERROR << "Play @nd Video";


	// auto currMediaProps = mp4Reader->getProps();
	// currMediaProps.videoPath = "/home/developer/workspace/ApraPipes/1684824653.mp4";
	// mp4Reader->setProps(currMediaProps);
	// boost::this_thread::sleep_for(boost::chrono::seconds(20));


	auto currMediaProps = mp4Reader->getProps();
	currMediaProps.videoPath = "/home/developer/workspace/ApraPipes/1684824632.mp4";//"/home/developer/workspace/ApraPipes/1684824632.mp4";
	mp4Reader->setProps(currMediaProps);
	boost::this_thread::sleep_for(boost::chrono::seconds(2));
	// Decoder->decoderEos();
	// mp4Reader->closeOpenFile();
	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
}

BOOST_AUTO_TEST_CASE(mp4reader_decoder_extsink)
{
	Logger::setLogLevel("info");

	// metadata is known
	std::string videoPath = "/home/developer/mp4_data/newatl.mp4";
	auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, false);
	auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	mp4Reader->addOutPutPin(h264ImageMetadata);

	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader->addOutPutPin(mp4Metadata);

	auto Decoder = boost::shared_ptr<Module>(new H264Decoder(H264DecoderProps()));
	std::vector<std::string> mImagePin;
	mImagePin = mp4Reader->getAllOutputPinsByType(FrameMetadata::FrameType::H264_DATA);
	mp4Reader->setNext(Decoder, mImagePin);

	auto m3 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	Decoder->setNext(m3);

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(mp4Reader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	p->run_all_threaded();

	Test_Utils::sleep_for_seconds(15);

	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
}

#else
BOOST_AUTO_TEST_CASE(h264_to_yuv420)
{
	Logger::setLogLevel("info");

	// metadata is known
	auto props = FileReaderModuleProps("./data/h264_data/FVDO_Freeway_4cif_???.H264", 0, -1);
	props.readLoop = false;
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(props));

	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));

	auto rawImagePin = fileReader->addOutputPin(h264ImageMetadata);

	auto Decoder = boost::shared_ptr<Module>(new H264Decoder(H264DecoderProps()));
	fileReader->setNext(Decoder);

	auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/yuv420Frames/Yuv420_704x576????.raw")));
	Decoder->setNext(fileWriter);
	fileReader->play(true);

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(fileReader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	p->run_all_threaded();

	Test_Utils::sleep_for_seconds(6);

	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();

}

BOOST_AUTO_TEST_CASE(encoder_to_decoder)
{
	Logger::setLogLevel("info");
	auto cuContext = apracucontext_sp(new ApraCUcontext());

	auto width = 640;
	auto height = 360;
	uint32_t gopLength = 25;
	uint32_t bitRateKbps = 1000;
	uint32_t frameRate = 30;
	H264EncoderNVCodecProps::H264CodecProfile profile = H264EncoderNVCodecProps::BASELINE;
	bool enableBFrames = true;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/Raw_YUV420_640x360/Image???_YUV420.raw", 0, -1)));
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));

	fileReader->addOutputPin(metadata);

	auto cudaStream_ = boost::shared_ptr<ApraCudaStream>(new ApraCudaStream());
	auto copyProps = CudaMemCopyProps(cudaMemcpyKind::cudaMemcpyHostToDevice, cudaStream_);
	copyProps.sync = true;
	auto copy = boost::shared_ptr<Module>(new CudaMemCopy(copyProps));
	fileReader->setNext(copy);

	auto encoder = boost::shared_ptr<Module>(new H264EncoderNVCodec(H264EncoderNVCodecProps(bitRateKbps, cuContext, gopLength, frameRate, profile, enableBFrames)));
	copy->setNext(encoder);

	auto Decoder = boost::shared_ptr<Module>(new H264Decoder(H264DecoderProps()));
	encoder->setNext(Decoder);

	auto m2 = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	Decoder->setNext(m2);

	fileReader->play(true);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(copy->init());
	BOOST_TEST(encoder->init());
	BOOST_TEST(Decoder->init());
	BOOST_TEST(m2->init());

	int index = 0;
	for (auto i = 0; i <= 43; i++)
	{

		fileReader->step();
		copy->step();
		encoder->step();
		Decoder->step();

		if (i >= 3)
		{
			auto frames = m2->pop();
			BOOST_TEST(frames.size() == 1);
			auto outputFrame = frames.cbegin()->second;
			BOOST_TEST(outputFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE_PLANAR);

			std::string fileName;

			if (index <= 9)
			{
				fileName = "/data/Raw_YUV420_640x360/Image00" + std::to_string(index) + "_YUV420.raw";
			}
			else
			{
				fileName = "/data/Raw_YUV420_640x360/Image0" + std::to_string(index) + "_YUV420.raw";
			}

			Test_Utils::saveOrCompare(fileName.c_str(), const_cast<const uint8_t*>(static_cast<uint8_t*>(outputFrame->data())), outputFrame->size(), 0);
			index++;
		}
	}
}

BOOST_AUTO_TEST_CASE(mp4reader_to_decoder_extSink)
{
	Logger::setLogLevel("info");

	std::string startingVideoPath_2 = "./data/Mp4_videos/h264_video/20221010/0012/1668064027062.mp4";
	auto mp4ReaderProps_2 = Mp4ReaderSourceProps(startingVideoPath_2, false);
	mp4ReaderProps_2.logHealth = true;
	mp4ReaderProps_2.logHealthFrequency = 100;
	mp4ReaderProps_2.fps = 30;
	auto mp4Reader_2 = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps_2));
	auto h264ImageMetadata_2 = framemetadata_sp(new H264Metadata(0, 0));
	mp4Reader_2->addOutPutPin(h264ImageMetadata_2);
	auto mp4Metadata_2 = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader_2->addOutPutPin(mp4Metadata_2);
	// metadata is known

	auto Decoder = boost::shared_ptr<Module>(new H264Decoder(H264DecoderProps()));
	mp4Reader_2->setNext(Decoder);

	StatSinkProps sinkProps;
	sinkProps.logHealth = true;
	sinkProps.logHealthFrequency = 100;
	auto sink = boost::shared_ptr<Module>(new StatSink(sinkProps));
	Decoder->setNext(sink);

	boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));

	p->appendModule(mp4Reader_2);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	p->run_all_threaded();
	Test_Utils::sleep_for_seconds(10);
	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();
}

#endif

BOOST_AUTO_TEST_SUITE_END()