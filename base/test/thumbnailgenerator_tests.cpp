#include "ThumbnailListGenerator.h"
#include "FileReaderModule.h"
#include <boost/test/unit_test.hpp>
#include "RTSPClientSrc.h"
#include "PipeLine.h"
#include "H264Decoder.h"
#include "H264Metadata.h"
#include "test_utils.h"

BOOST_AUTO_TEST_SUITE(thumbnailgenerator_tests)

struct rtsp_client_tests_data {
	rtsp_client_tests_data()
	{
		outFile = string("./data/testOutput/bunny.h264");
		Test_Utils::FileCleaner fc;
		fc.pathsOfFiles.push_back(outFile); //clear any occurance before starting the tests
	}
	string outFile;
	string empty;
};

BOOST_AUTO_TEST_CASE(basic)
{
    auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/YUV_420_planar.raw")));
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(1280, 720, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));
	auto rawImagePin = fileReader->addOutputPin(metadata);

    auto m_thumbnailGenerator = boost::shared_ptr<ThumbnailListGenerator>(new ThumbnailListGenerator(ThumbnailListGeneratorProps(180, 180, "./data/thumbnail.jpg")));
	fileReader->setNext(m_thumbnailGenerator);

    fileReader->init();
    m_thumbnailGenerator->init();

    fileReader->play(true);

    fileReader->step();
    m_thumbnailGenerator->step();

    fileReader->term();
}

BOOST_AUTO_TEST_CASE(basic_)
{
   rtsp_client_tests_data d;

	//drop bunny/mp4 into evostream folder, 
	//also set it up for RTSP client authentication as shown here: https://sites.google.com/apra.in/development/home/evostream/rtsp-authentication?authuser=1
	auto url=string("rtsp://10.102.10.77/axis-media/media.amp"); 
	
	auto m = boost::shared_ptr<Module>(new RTSPClientSrc(RTSPClientSrcProps(url, d.empty, d.empty)));
	auto meta = framemetadata_sp(new H264Metadata());
	m->addOutputPin(meta);

    auto Decoder = boost::shared_ptr<Module>(new H264Decoder(H264DecoderProps()));
    m->setNext(Decoder);

    auto m_thumbnailGenerator = boost::shared_ptr<ThumbnailListGenerator>(new ThumbnailListGenerator(ThumbnailListGeneratorProps(180, 180, "./data/thumbnail.jpg")));
	Decoder->setNext(m_thumbnailGenerator);

   boost::shared_ptr<PipeLine> p;
	p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
	p->appendModule(m);

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



BOOST_AUTO_TEST_SUITE_END()