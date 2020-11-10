#include "stdafx.h"
#include <boost/test/unit_test.hpp>

#include "FileReaderModule.h"
#include "JPEGDecoderIM.h"
#include "JPEGEncoderIM.h"
#include "StatSink.h"
#include "FileWriterModule.h"
#include "PipeLine.h"
#include "Logger.h"

BOOST_AUTO_TEST_SUITE(multiple_pipeline_tests)

string folderPath = "C:/Users/developer/Downloads/Sai/data/07";
string outputFilePattern = "C:/Users/developer/Downloads/Sai/temp/enc/frame_????.jpg";
int startIndex = 2500;
int endIndex = 3000;
bool sw = false;

BOOST_AUTO_TEST_CASE(sample)
{	
	PipeLine p1("test1");
	PipeLine p2("test2");
	PipeLine p3("test3");

	for (auto i = 0; i < 3; i++)
	{
		LoggerProps loggerProps;
		loggerProps.logLevel = boost::log::trivial::severity_level::info;
		Logger::initLogger(loggerProps);
		auto fileReaderModuleProps = FileReaderModuleProps(folderPath, startIndex, endIndex, 200 * 1024);
		//fileReaderModuleProps.fps = 30;
		auto fileReader = boost::shared_ptr<Module>(new FileReaderModule(fileReaderModuleProps));
		auto srcMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
		fileReader->addOutputPin(srcMetadata);

		auto jpegDecoderProps = JPEGDecoderIMProps();
		jpegDecoderProps.logHealth = true;
		auto decoder = boost::shared_ptr<Module>(new JPEGDecoderIM(jpegDecoderProps));
		fileReader->setNext(decoder);
		auto rawImageMetadata = framemetadata_sp(new RawImageMetadata());
		decoder->addOutputPin(rawImageMetadata);

		JPEGEncoderIMProps encoderProps;
		encoderProps.logHealth = true;
		auto encoder = boost::shared_ptr<JPEGEncoderIM>(new JPEGEncoderIM(encoderProps));
		decoder->setNext(encoder);
		auto sinkMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
		encoder->addOutputPin(sinkMetadata);

		auto fileWriter = boost::shared_ptr<FileWriterModule>(new FileWriterModule(FileWriterModuleProps(outputFilePattern)));
		encoder->setNext(fileWriter);

		auto sink3 = boost::shared_ptr<Module>(new StatSink());
		//decoder->setNext(sink3);
		encoder->setNext(sink3);

		PipeLine* p;
		if (i == 0)
		{
			p = &p1;
		}
		else if (i == 1)
		{
			p = &p2;
		}
		else
		{
			p = &p3;
		}
		p->appendModule(fileReader);

		p->init();
		p->run_all_threaded();
	}
	
	

	boost::this_thread::sleep_for(boost::chrono::seconds(5));

	for (auto i = 0; i < 3; i++)
	{
		PipeLine* p;
		if (i == 0)
		{
			p = &p1;
		}
		else if (i == 1)
		{
			p = &p2;
		}
		else
		{
			p = &p3;
		}
		p->stop();
		p->term();
	}
	
	for (auto i = 0; i < 3; i++)
	{
		PipeLine* p;
		if (i == 0)
		{
			p = &p1;
		}
		else if (i == 1)
		{
			p = &p2;
		}
		else
		{
			p = &p3;
		}
		p->wait_for_all();
	}
}

BOOST_AUTO_TEST_SUITE_END()