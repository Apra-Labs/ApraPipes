#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include <fstream>
#include "VADTransform.h"
#include "AudioCaptureSrc.h"
#include "FileWriterModule.h"
#include "PipeLine.h"
#include "Logger.h"
#include "test_utils.h"

BOOST_AUTO_TEST_SUITE(vad_tests)

BOOST_AUTO_TEST_CASE(vad_basic_test, *boost::unit_test::disabled())
{
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	LOG_INFO << "Starting VAD Basic Test...";

	int sampleRate = 16000;
	int channels = 1;  
	int processingInterval = 10;  
	
	AudioCaptureSrcProps audioProps(sampleRate, channels, 0, processingInterval);
	auto audioSrc = boost::shared_ptr<AudioCaptureSrc>(new AudioCaptureSrc(audioProps));

	VADTransformProps vadProps(
		16000,  
		VADTransformProps::QUALITY, 
		VADTransformProps::FRAME_10MS  
	);
	auto vad = boost::shared_ptr<VADTransform>(new VADTransform(vadProps));

	std::string filename = "./data/vad_output.raw";
	FileWriterModuleProps sinkProps(filename, true);
	auto sink = boost::shared_ptr<FileWriterModule>(new FileWriterModule(sinkProps));

	audioSrc->setNext(vad);
	vad->setNext(sink);

	PipeLine p("VADTestPipeline");
	p.appendModule(audioSrc);

	LOG_INFO << "Initializing Pipeline...";
	BOOST_TEST(p.init());

	LOG_INFO << "Running for 10 seconds...";
	LOG_INFO << "First 5 seconds: STAY SILENT (expect mostly 0s)";
	LOG_INFO << "Last 5 seconds: SPEAK INTO MIC (expect mostly 1s)";
	
	p.run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(10));

	LOG_INFO << "Stopping Pipeline...";
	p.stop();
	p.term();
	p.wait_for_all();
}

BOOST_AUTO_TEST_CASE(vad_aggressiveness_test, *boost::unit_test::disabled())
{
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	LOG_INFO << "Testing different aggressiveness modes...";

	// Test all 4 aggressiveness modes
	VADTransformProps::AggressivenessMode modes[] = {
		VADTransformProps::QUALITY,
		VADTransformProps::LOW_BITRATE,
		VADTransformProps::AGGRESSIVE,
		VADTransformProps::VERY_AGGRESSIVE
	};

	for (int i = 0; i < 4; i++) {
		LOG_INFO << "Testing mode " << i << "...";
		
		VADTransformProps props(16000, modes[i], VADTransformProps::FRAME_10MS);
		auto vad = boost::shared_ptr<VADTransform>(new VADTransform(props));
		
		// Just verify it initializes
		BOOST_TEST(vad->init());
		vad->term();
	}
	
	LOG_INFO << "All aggressiveness modes initialized successfully";
}

BOOST_AUTO_TEST_SUITE_END()
