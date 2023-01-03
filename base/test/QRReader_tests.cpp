#include "stdafx.h"
#include <boost/test/unit_test.hpp>

#include "FileReaderModule.h"
#include "ExternalSinkModule.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "AIPExceptions.h"
#include "QRReader.h"

BOOST_AUTO_TEST_SUITE(QRReader_tests)

void testQRResult(std::string inPutFileName, framemetadata_sp metadata)
{	
	Logger::setLogLevel(boost::log::trivial::severity_level::trace);
    auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps(inPutFileName)));
	fileReader->addOutputPin(metadata);
	
	auto QRData = boost::shared_ptr<QRReader>(new QRReader());
	fileReader->setNext(QRData);
				
	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	QRData->setNext(sink);

	BOOST_TEST(fileReader->init());
	BOOST_TEST(QRData->init());
	BOOST_TEST(sink->init());	
	
	BOOST_TEST(fileReader->step());
	BOOST_TEST(QRData->step());
	auto frames = sink->pop();
	BOOST_TEST(frames.size() == 1);
	auto outputFrame = frames.cbegin()->second;
	BOOST_TEST(outputFrame.get()!=nullptr);
	std::string expectedOutput = "0005100788";
	auto actualOutput = std::string(const_cast<const char*>( static_cast<char*>(outputFrame->data()) ), outputFrame->size() );
	BOOST_TEST(expectedOutput == actualOutput);
}


BOOST_AUTO_TEST_CASE(rgb)
{	
	auto metadata = framemetadata_sp(new RawImageMetadata(935,935,ImageMetadata::ImageType::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));
	testQRResult("./data/QR.raw",metadata);
}

BOOST_AUTO_TEST_CASE(yuv420)
{	
	auto metadata = framemetadata_sp(new RawImagePlanarMetadata(935, 935, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));
	testQRResult("./data/QR_yuv420.raw",metadata);
}


BOOST_AUTO_TEST_SUITE_END()
