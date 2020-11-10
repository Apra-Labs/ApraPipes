#include "stdafx.h"
#include <boost/test/unit_test.hpp>

#include "FileReaderModule.h"
#include "JPEGDecoderIM.h"
#include "JPEGEncoderIM.h"
#include "StatSink.h"
#include "FileWriterModule.h"
#include "PipeLine.h"
#include "Logger.h"
#include <boost/algorithm/string/replace.hpp>
#include "ExternalSourceModule.h"

BOOST_AUTO_TEST_SUITE(logger_stress_tests)

string folderPath = "C:/Users/developer/Downloads/Sai/data/07";
string outputFilePattern = "C:/Users/developer/Downloads/Sai/temp/enc/frame_????.jpg";
int startIndex = 2500;
int endIndex = 3000;
bool sw = false;

class LoggerTestModule : public ExternalSourceModule
{
public:
	LoggerTestModule(ExternalSourceModuleProps props) : ExternalSourceModule(props) 
	{
		counter = 0;
	}
	~LoggerTestModule() {}

protected:
	bool produce() 
	{
		LOG_INFO << "hola how are you <>" << counter++;
		return true;
	}

	uint64_t counter;
};

BOOST_AUTO_TEST_CASE(sample)
{		
	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	loggerProps.enableFileLog = true;
	ostringstream ss;
	ss << boost::posix_time::microsec_clock::universal_time();
	std::string tsstr = ss.str();	
	boost::replace_all(tsstr, " ", "");
	boost::replace_all(tsstr, ".", "");
	boost::replace_all(tsstr, ":", "");
	boost::replace_all(tsstr, "-", "");

	loggerProps.fileLogPath = "hello_" + tsstr + ".log";
	Logger::initLogger(loggerProps);
	auto fileReaderModuleProps = FileReaderModuleProps(folderPath, startIndex, endIndex, 200 * 1024);
	fileReaderModuleProps.fps = 2000;
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

	PipeLine p("test");
	p.appendModule(fileReader);

	for (auto i = 0; i < 3; i++)
	{
		ExternalSourceModuleProps props;
		props.fps = 100;
		auto logger1 = boost::shared_ptr<Module>(new LoggerTestModule(props));
		auto genMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
		logger1->addOutputPin(genMetadata);
		p.appendModule(logger1);
	}


	p.init();
	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(2));
	p.stop();
	p.term();

	p.wait_for_all();

}

BOOST_AUTO_TEST_CASE(sample2)
{
	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	loggerProps.enableFileLog = true;
	ostringstream ss;
	ss << boost::posix_time::microsec_clock::universal_time();
	std::string tsstr = ss.str();
	boost::replace_all(tsstr, " ", "");
	boost::replace_all(tsstr, ".", "");
	boost::replace_all(tsstr, ":", "");
	boost::replace_all(tsstr, "-", "");

	loggerProps.fileLogPath = "hello_" + tsstr + ".log";
	Logger::initLogger(loggerProps);
	
	PipeLine p("test");
	

	ExternalSourceModuleProps props;
	props.fps = 100;
	auto logger1 = boost::shared_ptr<Module>(new LoggerTestModule(props));
	auto genMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::GENERAL));
	logger1->addOutputPin(genMetadata);
	p.appendModule(logger1);
	   
	p.init();

	for (auto i = 0; i < 100; i++)
	{
		logger1->step();
	}

	p.stop();
	p.term();

}

BOOST_AUTO_TEST_CASE(sample3)
{
	LOG_ERROR << "HOLA SAMPLE3";
	boost::this_thread::sleep_for(boost::chrono::seconds(2));
}

BOOST_AUTO_TEST_SUITE_END()