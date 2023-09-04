#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include "Logger.h"
#include "test_utils.h"
#include "ExternalSourceModule.h"
#include "STLRendererSink.h"
#include "ApraData.h"

BOOST_AUTO_TEST_SUITE(stlrenderer_tests)

BOOST_AUTO_TEST_CASE(basic)
{
	auto m1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new RawImageMetadata(100, 100, 1, CV_8UC1, 100, CV_8U));
	auto rawImagePinId = m1->addOutputPin(metadata);
	auto stlMod = boost::shared_ptr<STLRendererSink>(new STLRendererSink(STLRendererSinkProps()));
	m1->setNext(stlMod);

	BOOST_TEST(m1->init());
	BOOST_TEST(stlMod->init());
	
	auto frame = m1->makeFrame(10, rawImagePinId);
	ApraData* data = new ApraData(frame->data(), 10, 0);
	m1->produceExternalFrame(data);
	stlMod->step();
	boost::this_thread::sleep_for(boost::chrono::seconds(30));
}

BOOST_AUTO_TEST_SUITE_END()
