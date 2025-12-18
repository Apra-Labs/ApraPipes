#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include <fstream>
#include "VoiceActivityDetector.h"
#include "AudioCaptureSrc.h"
#include "FileWriterModule.h"
#include "PipeLine.h"
#include "Logger.h"
#include "test_utils.h"

BOOST_AUTO_TEST_SUITE(voice_activity_detector_tests)

BOOST_AUTO_TEST_CASE(voice_activity_detector_basic_test)
{
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	LOG_INFO << "Starting Voice Activity Detector Basic Test...";

	int sampleRate = 16000;
	int channels = 1;  
	int processingInterval = 10;  
	
	AudioCaptureSrcProps audioProps(sampleRate, channels, 0, processingInterval);
	auto audioSrc = boost::shared_ptr<AudioCaptureSrc>(new AudioCaptureSrc(audioProps));

	VoiceActivityDetectorProps vadProps(
		16000,  
		VoiceActivityDetectorProps::VERY_AGGRESSIVE, 
		VoiceActivityDetectorProps::FRAME_10MS  
	);
	auto vad = boost::shared_ptr<VoiceActivityDetector>(new VoiceActivityDetector(vadProps));

	// Sink 1: Audio Passthrough
	std::string fileAudio = "./data/basic_test_audio.raw";
	FileWriterModuleProps sinkPropsAudio(fileAudio, true);
	auto sinkAudio = boost::shared_ptr<FileWriterModule>(new FileWriterModule(sinkPropsAudio));

	// Sink 2: VAD Result
	std::string fileVad = "./data/basic_test_vad.raw";
	FileWriterModuleProps sinkPropsVad(fileVad, true);
	auto sinkVad = boost::shared_ptr<FileWriterModule>(new FileWriterModule(sinkPropsVad));

	audioSrc->setNext(vad);
	
	auto audioPins = vad->getAllOutputPinsByType(FrameMetadata::FrameType::AUDIO);
	auto vadPins = vad->getAllOutputPinsByType(FrameMetadata::FrameType::GENERAL);

	vad->setNext(sinkAudio, audioPins);
	vad->setNext(sinkVad, vadPins);

	PipeLine p("VoiceActivityDetectorTestPipeline");
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

BOOST_AUTO_TEST_CASE(voice_activity_detector_aggressiveness_test)
{
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	LOG_INFO << "Testing different aggressiveness modes...";

	// Test all 4 aggressiveness modes
	VoiceActivityDetectorProps::AggressivenessMode modes[] = {
		VoiceActivityDetectorProps::QUALITY,
		VoiceActivityDetectorProps::LOW_BITRATE,
		VoiceActivityDetectorProps::AGGRESSIVE,
		VoiceActivityDetectorProps::VERY_AGGRESSIVE
	};

	for (int i = 0; i < 4; i++) {
		LOG_INFO << "Testing mode " << i << "...";
		
		AudioCaptureSrcProps audioProps(16000, 1, 0, 10);
		auto source = boost::shared_ptr<AudioCaptureSrc>(new AudioCaptureSrc(audioProps));

		VoiceActivityDetectorProps props(16000, modes[i], VoiceActivityDetectorProps::FRAME_10MS);
		auto vad = boost::shared_ptr<VoiceActivityDetector>(new VoiceActivityDetector(props));
		
		source->setNext(vad);

		BOOST_TEST(source->init());
		BOOST_TEST(vad->init());

		// Verify the mode was set correctly
		auto currentProps = vad->getProps();
		BOOST_TEST(currentProps.mode == modes[i]);

		vad->term(); 
		source->term();
	}
	
	LOG_INFO << "All aggressiveness modes initialized successfully";
}

BOOST_AUTO_TEST_CASE(voice_activity_detector_props_test)
{
	Logger::setLogLevel(boost::log::trivial::severity_level::info);
	LOG_INFO << "Testing getProps and setProps...";

	// 1. Setup helper source to satisfy input pin requirement
	AudioCaptureSrcProps audioProps(16000, 1, 0, 10);
	auto source = boost::shared_ptr<AudioCaptureSrc>(new AudioCaptureSrc(audioProps));

	VoiceActivityDetectorProps initialProps(
		16000,
		VoiceActivityDetectorProps::QUALITY,
		VoiceActivityDetectorProps::FRAME_10MS
	);
	auto vad = boost::shared_ptr<VoiceActivityDetector>(new VoiceActivityDetector(initialProps));

	// Connect source to vad so validation passes (expects 1 input pin)
	source->setNext(vad);

	// Create two sinks for the two output pins (Audio, VAD)
	std::string filenameAudio = "./data/test_audio.raw";
	FileWriterModuleProps sinkPropsAudio(filenameAudio, true);
	auto sinkAudio = boost::shared_ptr<FileWriterModule>(new FileWriterModule(sinkPropsAudio));

	std::string filenameVad = "./data/test_vad.raw";
	FileWriterModuleProps sinkPropsVad(filenameVad, true);
	auto sinkVad = boost::shared_ptr<FileWriterModule>(new FileWriterModule(sinkPropsVad));

	// Connect VAD outputs
	// This ensures both output pins are connected
	vad->setNext(sinkAudio); 
	vad->setNext(sinkVad);

	// Initialize
	if (!source->init()) 
	{
		BOOST_ERROR("Source init failed");
		return;
	}
	if (!sinkAudio->init())
	{
		BOOST_ERROR("SinkAudio init failed");
		return;
	}
	if (!sinkVad->init())
	{
		BOOST_ERROR("SinkVad init failed");
		return;
	}
	if (!vad->init())
	{
		BOOST_ERROR("VAD init failed");
		return;
	}

	// Verify initial props
	auto currentProps = vad->getProps();
	BOOST_TEST(currentProps.sampleRate == 16000);
	BOOST_TEST(currentProps.mode == VoiceActivityDetectorProps::QUALITY);
	BOOST_TEST(currentProps.frameLength == VoiceActivityDetectorProps::FRAME_10MS);

	// Change props
	VoiceActivityDetectorProps newProps(
		16000,
		VoiceActivityDetectorProps::VERY_AGGRESSIVE,
		VoiceActivityDetectorProps::FRAME_20MS
	);
	
	// setProps is async - it queues a command
	vad->setProps(newProps);

	vad->step(); 

	// Verify updated props
	currentProps = vad->getProps();
	BOOST_TEST(currentProps.mode == VoiceActivityDetectorProps::VERY_AGGRESSIVE);
	BOOST_TEST(currentProps.frameLength == VoiceActivityDetectorProps::FRAME_20MS);
	
	vad->term();
	source->term();
	sinkAudio->term();
	sinkVad->term();
}

BOOST_AUTO_TEST_SUITE_END()
