#include "stdafx.h"
#include <boost/test/unit_test.hpp>

#include "FileReaderModule.h"
#include "FrameMetadata.h"
#include "MotionVectorExtractor.h"
#include "OverlayMotionVectors.h"
#include "ExternalSinkModule.h"
#include "test_utils.h"
#include "PipeLine.h"
#include "H264Metadata.h"
#include "FileWriterModule.h"

BOOST_AUTO_TEST_SUITE(overlay_motion_vectors_tests)

BOOST_AUTO_TEST_CASE(basic_extract_motion_vector)
{
	FileReaderModuleProps fileReaderProps("./data/h264_data/FVDO_Freeway_4cif_???.H264");
	fileReaderProps.fps = 30;
	fileReaderProps.readLoop = false;
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
	auto metadata = framemetadata_sp(new H264Metadata(0, 0));
	fileReader->addOutputPin(metadata);

	auto motionExtractor = boost::shared_ptr<Module>(new MotionVectorExtractor(MotionVectorExtractorProps()));
	fileReader->setNext(motionExtractor);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	motionExtractor->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(motionExtractor->init());
	BOOST_TEST(sink->init());

	fileReader->play(true);

	for (int i = 0; i < 231; i++)
	{
		fileReader->step();
		motionExtractor->step();
		auto frames = sink->pop();
		auto outFrame = frames.begin()->second;
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::MOTION_VECTOR_DATA);
		std::string fileName = "./data/testOutput/motionVector" + std::to_string(outFrame->fIndex2) + ".raw";
		Test_Utils::saveOrCompare(fileName.c_str(), const_cast<const uint8_t*>(static_cast<uint8_t*>(outFrame->data())), outFrame->size(), 0);
	}
}

BOOST_AUTO_TEST_CASE(extract_vectors_and_overlay)
{
	bool enableOverlay = true;

	FileReaderModuleProps fileReaderProps("./data/h264_data/FVDO_Freeway_4cif_???.H264");
	fileReaderProps.fps = 30;
	fileReaderProps.readLoop = true;
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	fileReader->addOutputPin(h264ImageMetadata);

	auto motionExtractor = boost::shared_ptr<MotionVectorExtractor>(new MotionVectorExtractor(MotionVectorExtractorProps(enableOverlay)));
	fileReader->setNext(motionExtractor);

	auto overlayMotionVector = boost::shared_ptr<Module>(new OverlayMotionVector(OverlayMotionVectorProps()));
	motionExtractor->setNext(overlayMotionVector);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	overlayMotionVector->setNext(sink);


	BOOST_TEST(fileReader->init());
	BOOST_TEST(motionExtractor->init());
	BOOST_TEST(overlayMotionVector->init());
	BOOST_TEST(sink->init());

	fileReader->play(true);

	for (int i = 0; i < 231; i++)
	{
		fileReader->step();
		motionExtractor->step();
		overlayMotionVector->step();
		auto frames = sink->pop();
		auto outFrame = frames.begin()->second;
		BOOST_TEST(outFrame->getMetadata()->getFrameType() == FrameMetadata::RAW_IMAGE);
		std::string fileName = "./data/testOutput/MotionVectorOverlayFrame" + std::to_string(outFrame->fIndex2) + ".raw";
		Test_Utils::saveOrCompare(fileName.c_str(), const_cast<const uint8_t*>(static_cast<uint8_t*>(outFrame->data())), outFrame->size(), 0);
	}
}

BOOST_AUTO_TEST_CASE(extract_vectors_and_overlay_setprops) // overlay can be shown only when requested by user but motion vectors are for every frame.
{
	FileReaderModuleProps fileReaderProps("./data/h264_data/FVDO_Freeway_4cif_???.H264");
	fileReaderProps.fps = 30;
	fileReaderProps.readLoop = true;
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	fileReader->addOutputPin(h264ImageMetadata);

	auto motionExtractor = boost::shared_ptr<MotionVectorExtractor>(new MotionVectorExtractor(MotionVectorExtractorProps()));
	fileReader->setNext(motionExtractor);

	auto overlayMotionVector = boost::shared_ptr<Module>(new OverlayMotionVector(OverlayMotionVectorProps()));
	motionExtractor->setNext(overlayMotionVector);

	auto sink = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/motionVectorOverlay/frame_??.raw")));
	overlayMotionVector->setNext(sink);

	PipeLine p("test");
	p.appendModule(fileReader);
	p.init();

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(10));

	MotionVectorExtractorProps propsChange(true);
	motionExtractor->setProps(propsChange);

	boost::this_thread::sleep_for(boost::chrono::seconds(10));

	LOG_INFO << "profiling done - stopping the pipeline";
	p.stop();
	p.term();
	p.wait_for_all();
}


BOOST_AUTO_TEST_SUITE_END()
