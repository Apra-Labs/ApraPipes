#include <boost/test/unit_test.hpp>

#include "PipeLine.h"
#include "H264EncoderV4L2.h"
#include "FileReaderModule.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "StatSink.h"
#include "CudaMemCopy.h"
#include "ExternalSinkModule.h"
#include "RTSPPusher.h"
#include "Overlay.h"
#include "OverlayModule.h"
#include "H264Metadata.h"
#include "FramesMuxer.h"

#include "test_utils.h"

BOOST_AUTO_TEST_SUITE(h264encoderv4l2_tests)

BOOST_AUTO_TEST_CASE(yuv420_640x360)
{
	// metadata is known
	auto width = 640;
	auto height = 360;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/Raw_YUV420_640x360/Image???_YUV420.raw")));
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));
	auto rawImagePin = fileReader->addOutputPin(metadata);

	H264EncoderV4L2Props encoderProps;
	encoderProps.targetKbps = 1024;
	auto encoder = boost::shared_ptr<Module>(new H264EncoderV4L2(encoderProps));
	fileReader->setNext(encoder);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	encoder->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(encoder->init());
	BOOST_TEST(sink->init());

	fileReader->play(true);


	for (auto i = 0; i < 42; i++)
	{
		fileReader->step();
		encoder->step();
		auto frames = sink->pop();
		auto outputFrame = frames.begin()->second;
		std::string fileName = "./data/testOutput/h264EncoderV4l2/Raw_YUV420_640x360_" + to_string(i) + ".h264";
		Test_Utils::saveOrCompare(fileName.c_str(),  const_cast<const uint8_t*>(static_cast<uint8_t*>(outputFrame->data())), outputFrame->size(), 0);
	}
}

BOOST_AUTO_TEST_CASE(rgb24_1280x720, *boost::unit_test::disabled())
{
	// metadata is known
	auto width = 1280;
	auto height = 720;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/Raw_RGB24_1280x720")));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::RGB, CV_8UC3, size_t(0), CV_8U, FrameMetadata::HOST, true));
	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copyProps = CudaMemCopyProps(cudaMemcpyHostToDevice, stream);
	copyProps.sync = true;
	auto copy = boost::shared_ptr<Module>(new CudaMemCopy(copyProps));
	fileReader->setNext(copy);

	H264EncoderV4L2Props encoderProps;
	encoderProps.targetKbps = 1024;
	auto encoder = boost::shared_ptr<Module>(new H264EncoderV4L2(encoderProps));
	copy->setNext(encoder);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	encoder->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(copy->init());
	BOOST_TEST(encoder->init());
	BOOST_TEST(sink->init());

	fileReader->play(true);


	for (auto i = 0; i < 42; i++)
	{
		fileReader->step();
		copy->step();
		encoder->step();
		auto frames = sink->pop();
		auto outputFrame = frames.begin()->second;
		Test_Utils::saveOrCompare("./data/testOutput/Raw_RGB24_1280x720.h264",  const_cast<const uint8_t*>(static_cast<uint8_t*>(outputFrame->data())), outputFrame->size(), 0);
	}
	
}

BOOST_AUTO_TEST_CASE(yuv420_640x360_profiling, *boost::unit_test::disabled())
{
	// metadata is known
	auto width = 640;
	auto height = 360;

	FileReaderModuleProps fileReaderProps("./data/Raw_YUV420_640x360/Image???_YUV420.raw");
	fileReaderProps.fps = 1000;
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));
	auto rawImagePin = fileReader->addOutputPin(metadata);

	H264EncoderV4L2Props encoderProps;
	encoderProps.targetKbps = 1024;
	auto encoder = boost::shared_ptr<Module>(new H264EncoderV4L2(encoderProps));
	fileReader->setNext(encoder);

	StatSinkProps sinkProps;
	sinkProps.logHealth = true;
	auto sink = boost::shared_ptr<Module>(new StatSink(sinkProps));
	encoder->setNext(sink);

	PipeLine p("test");
	p.appendModule(fileReader);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(100));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(rgb24_1280x720_profiling, *boost::unit_test::disabled())
{
	// metadata is known
	auto width = 1280;
	auto height = 720;

	FileReaderModuleProps fileReaderProps("./data/Raw_RGB24_1280x720");
	fileReaderProps.fps = 1000;
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
	auto metadata = framemetadata_sp(new RawImageMetadata(width, height, ImageMetadata::ImageType::RGB, CV_8UC3, size_t(0), CV_8U, FrameMetadata::HOST, true));
	auto rawImagePin = fileReader->addOutputPin(metadata);

	auto stream = cudastream_sp(new ApraCudaStream);
	auto copyProps = CudaMemCopyProps(cudaMemcpyHostToDevice, stream);
	copyProps.sync = true;
	auto copy = boost::shared_ptr<Module>(new CudaMemCopy(copyProps));
	fileReader->setNext(copy);

	H264EncoderV4L2Props encoderProps;
	encoderProps.targetKbps = 1024;
	auto encoder = boost::shared_ptr<Module>(new H264EncoderV4L2(encoderProps));
	copy->setNext(encoder);

	StatSinkProps sinkProps;
	sinkProps.logHealth = true;
	auto sink = boost::shared_ptr<Module>(new StatSink(sinkProps));
	encoder->setNext(sink);

	PipeLine p("test");
	p.appendModule(fileReader);
	BOOST_TEST(p.init());

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(100));
	Logger::setLogLevel(boost::log::trivial::severity_level::error);

	p.stop();
	p.term();

	p.wait_for_all();
}


BOOST_AUTO_TEST_CASE(encodepush, *boost::unit_test::disabled())
{
	// metadata is known
	auto width = 640;
	auto height = 360;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/Raw_YUV420_640x360/Image???_YUV420.raw")));
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));
	auto rawImagePin = fileReader->addOutputPin(metadata);

	H264EncoderV4L2Props encoderProps;
	encoderProps.targetKbps = 1024;
	auto encoder = boost::shared_ptr<Module>(new H264EncoderV4L2(encoderProps));
	fileReader->setNext(encoder);

	auto sink = boost::shared_ptr<Module>(new RTSPPusher(RTSPPusherProps("rtsp://10.102.10.129:5544", "aprapipes_h264")));
	encoder->setNext(sink);

	PipeLine p("test");
	p.appendModule(fileReader);
	BOOST_TEST(p.init());
	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(5));

	LOG_DEBUG << "STOPPING";

	p.stop();
	p.term();
	LOG_DEBUG << "WAITING";
	p.wait_for_all();
	LOG_INFO << "TEST DONE";
}

BOOST_AUTO_TEST_CASE(encode_and_extract_motion_vectors)
{
	// metadata is known
	auto width = 640;
	auto height = 360;

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/Raw_YUV420_640x360/Image???_YUV420.raw")));
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));
	auto rawImagePin = fileReader->addOutputPin(metadata);

	H264EncoderV4L2Props encoderProps;
	encoderProps.targetKbps = 1024;
	encoderProps.enableMotionVectors = true;
	auto encoder = boost::shared_ptr<Module>(new H264EncoderV4L2(encoderProps));
	fileReader->setNext(encoder);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	encoder->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(encoder->init());
	BOOST_TEST(sink->init());

	fileReader->play(true);

	int motionVectorFramesCount  = 0;
	for (auto i = 0; i < 40; i++)
	{
		fileReader->step();
		encoder->step();
		auto frames = sink->pop();
		for (auto it = frames.cbegin(); it != frames.cend(); it++)
		{
			auto metadata = it->second->getMetadata();
			auto frameType = metadata->getFrameType();
			auto outputFrame = it->second;
			if (frameType == FrameMetadata::H264_DATA)
			{
				std::string fileName = "./data/testOutput/h264EncoderH264Frames/frame_640x360" +  to_string(i) + ".h264";
				Test_Utils::saveOrCompare(fileName.c_str(), const_cast<const uint8_t*>(static_cast<uint8_t*>(outputFrame->data())), outputFrame->size(), 0);
			}
			else if(frameType == FrameMetadata::OVERLAY_INFO_IMAGE)
			{
				DrawingOverlay drawOverlay;
				drawOverlay.deserialize(outputFrame);
				auto list = drawOverlay.getList();
				motionVectorFramesCount++;
				for (auto primitive1 : list)
				{
					if (primitive1->primitiveType == Primitive::COMPOSITE)
					{
						CompositeOverlay *mCompositeOverlay1 = static_cast<CompositeOverlay *>(primitive1);

						auto compositeList1 = mCompositeOverlay1->getList();

						for (auto primitive2 : compositeList1)
						{
							BOOST_TEST(primitive2->primitiveType == Primitive::CIRCLE);
						}
					}
				}
			}
		}
	}
	bool condition = (motionVectorFramesCount == 32 || motionVectorFramesCount == 33);
    BOOST_TEST(condition);
}
BOOST_AUTO_TEST_SUITE_END()
