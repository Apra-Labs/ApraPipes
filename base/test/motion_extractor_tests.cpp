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
	FileReaderModuleProps fileReaderProps("./data/h264_data/FVDO_Freeway_4cif_???.H264");
	fileReaderProps.fps = 30;
	fileReaderProps.readLoop = false;
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::H264_DATA));
	fileReader->addOutputPin(metadata);

	auto m1 = boost::shared_ptr<Module>(new MotionExtractor(MotionExtractorProps()));
	fileReader->setNext(m1);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	m1->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(m1->init());
	BOOST_TEST(sink->init());

	fileReader->play(true);
	int i;
	for (i = 0; i < 231; i++)
	{
		fileReader->step();
		m1->step();
		sink->step();
	}
	m1->step();
	//m1->step();
	LOG_INFO << "completed";
}

BOOST_AUTO_TEST_SUITE_END()
