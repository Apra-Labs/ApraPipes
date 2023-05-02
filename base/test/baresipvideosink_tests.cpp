#include <stdafx.h>
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include "ExternalSourceModule.h"
#include "BaresipVideoSink.h"
#include "Logger.h"
#include "WebCamSource.h"
#include "FileReaderModule.h"
#include "ColorConversionXForm.h"
#include "AIPExceptions.h"
#include "test_utils.h"
#include "PipeLine.h"
#include "Module.h"

BOOST_AUTO_TEST_SUITE(baresipvideosink_tests)

BOOST_AUTO_TEST_CASE(basic, * boost::unit_test::disabled())
{
    WebCamSourceProps webCamSourceprops(0, 640, 360);
    auto webCam = boost::shared_ptr<WebCamSource>(new WebCamSource(webCamSourceprops));

	auto colorConvt = boost::shared_ptr<ColorConversion>(new ColorConversion(ColorConversionProps(ColorConversionProps::ConversionType::RGB_TO_YUV420PLANAR)));
 	webCam->setNext(colorConvt);

    auto sink = boost::shared_ptr<BaresipVideoSink>(new BaresipVideoSink(BaresipVideoSinkProps()));
    colorConvt->setNext(sink);

    boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(webCam);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	p->run_all_threaded();
	
	Test_Utils::sleep_for_seconds(999);

	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();

}

BOOST_AUTO_TEST_CASE(webcam_source, * boost::unit_test::disabled())
{
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/Raw_YUV420_640x360/Image???_YUV420.raw", 0, -1)));
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(640, 360, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));

	fileReader->addOutputPin(metadata);
    auto sink = boost::shared_ptr<BaresipVideoSink>(new BaresipVideoSink(BaresipVideoSinkProps()));
    fileReader->setNext(sink);
    boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(fileReader);

	if (!p->init())
	{
		throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
	}

	p->run_all_threaded();
	
	Test_Utils::sleep_for_seconds(999);

	p->stop();
	p->term();
	p->wait_for_all();
	p.reset();

}


BOOST_AUTO_TEST_CASE(adapter_test, * boost::unit_test::disabled())
{
	void * buffer = calloc(1,(640*360*3)>>1);
    auto baresip = boost::shared_ptr<BaresipVideoAdapter>(new BaresipVideoAdapter(BaresipVideoAdapterProps()));
	BOOST_TEST(baresip->init());
	while (true)
	{
		BOOST_TEST(baresip->process(buffer));
		Test_Utils::sleep_for_milliseconds(33);
		
	}
	BOOST_TEST(baresip->term());

}



BOOST_AUTO_TEST_SUITE_END()