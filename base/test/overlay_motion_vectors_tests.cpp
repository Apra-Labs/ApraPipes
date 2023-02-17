#include "stdafx.h"
#include <boost/test/unit_test.hpp>

#include "FileReaderModule.h"
#include "FrameMetadata.h"
#include "H264MotionExtractorXForm.h"
#include "OverlayMotionVectors.h"
#include "ExternalSinkModule.h"
#include "FramesMuxer.h"
#include "Logger.h"
#include "test_utils.h"
#include "PipeLine.h"
#include "H264Metadata.h"

BOOST_AUTO_TEST_SUITE(overlay_motion_vectors_tests)

BOOST_AUTO_TEST_CASE(basic)
{
	FileReaderModuleProps fileReaderProps("./data/h264_data/FVDO_Freeway_4cif_???.H264");
	fileReaderProps.fps = 30;
	fileReaderProps.readLoop = true;
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(704, 576));
	fileReader->addOutputPin(h264ImageMetadata);

	auto motionExtractor = boost::shared_ptr<MotionExtractor>(new MotionExtractor(MotionExtractorProps()));
	fileReader->setNext(motionExtractor);

	auto overlayMotionVector = boost::shared_ptr<Module>(new OverlayMotionVector(OverlayMotionVectorProps()));
	motionExtractor->setNext(overlayMotionVector);

	PipeLine p("test");
	p.appendModule(fileReader);
	p.init();

	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(10));

	MotionExtractorProps propsChange(true);
	motionExtractor->setProps(propsChange);

	boost::this_thread::sleep_for(boost::chrono::seconds(10));

	LOG_INFO << "profiling done - stopping the pipeline";
	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_SUITE_END()
