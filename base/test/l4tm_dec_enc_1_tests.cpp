#include "stdafx.h"
#include <boost/test/unit_test.hpp>

#include "FileReaderModule.h"
// #include "JPEGDecoderIM.h"
#include "JPEGDecoderL4TM.h"
#include "JPEGEncoderL4TM.h"
// #include "JPEGEncoderIM.h"
#include "StatSink.h"
#include "FileWriterModule.h"
#include "PipeLine.h"
#include "Logger.h"
#include <iostream>

BOOST_AUTO_TEST_SUITE(l4tm_dec_enc_1_tests)

string folderPath = "/home/apra/Downloads/re3_filtered";
string outputFilePattern = "/home/apra/Downloads/re3_filtered_out/frame_????.jpg";
int startIndex = 0;
int endIndex = -1;
bool sw = false;

BOOST_AUTO_TEST_CASE(sample)
{		
	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	Logger::initLogger(loggerProps);
	auto fileReaderModuleProps = FileReaderModuleProps(folderPath, startIndex, endIndex, 200 * 1024);
	//fileReaderModuleProps.fps = 30;
	auto fileReader = boost::shared_ptr<Module>(new FileReaderModule(fileReaderModuleProps));
	auto encodedImageMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	fileReader->addOutputPin(encodedImageMetadata);

	auto jpegDecoderProps = JPEGDecoderL4TMProps();
	jpegDecoderProps.logHealth = true;
	auto decoder = boost::shared_ptr<Module>(new JPEGDecoderL4TM(jpegDecoderProps));
	fileReader->setNext(decoder);
	auto rawImageMetadata = framemetadata_sp(new RawImageMetadata());
	decoder->addOutputPin(rawImageMetadata);

	JPEGEncoderL4TMProps encoderProps;
	encoderProps.logHealth = true;
	auto encoder = boost::shared_ptr<JPEGEncoderL4TM>(new JPEGEncoderL4TM(encoderProps));
	decoder->setNext(encoder);
	auto outMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
	encoder->addOutputPin(outMetadata);
	
	auto fileWriter = boost::shared_ptr<FileWriterModule>(new FileWriterModule(FileWriterModuleProps(outputFilePattern)));
	encoder->setNext(fileWriter);

	auto sink3 = boost::shared_ptr<Module>(new StatSink());
	//decoder->setNext(sink3);
	encoder->setNext(sink3);

	PipeLine p("test");
	p.appendModule(fileReader);

	p.init();
	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(5));
	p.stop();
	p.term();

	p.wait_for_all();

}

BOOST_AUTO_TEST_SUITE_END()