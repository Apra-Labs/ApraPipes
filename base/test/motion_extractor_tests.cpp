#include "stdafx.h"
#include <boost/test/unit_test.hpp>

#include "FileReaderModule.h"
#include "FrameMetadata.h"
#include "H264MotionExtractorXForm.h"
#include "ExternalSinkModule.h"
#include "Logger.h"
#include "test_utils.h"
#include "PipeLine.h"

BOOST_AUTO_TEST_SUITE(motion_vectors_tests)

BOOST_AUTO_TEST_CASE(basic, *boost::unit_test::disabled())
{
	FileReaderModuleProps fileReaderProps("./data/h264_data/FVDO_Freeway_4cif_230.H264");
	fileReaderProps.fps = 30;
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::H264_DATA));
	fileReader->addOutputPin(metadata);

	auto m1 = boost::shared_ptr<Module>(new MotionExtractor(MotionExtractorProps()));
	fileReader->setNext(m1);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	m1->setNext(sink);

	PipeLine p("test");
	p.appendModule(fileReader);
	BOOST_TEST(p.init());
	p.run_all_threaded();

	boost::this_thread::sleep_for(boost::chrono::seconds(50));

	LOG_INFO << "STOPPING";

	p.stop();
	p.term();
	LOG_INFO << "WAITING";
	p.wait_for_all();
	LOG_INFO << "TEST DONE";
}

BOOST_AUTO_TEST_SUITE_END()
