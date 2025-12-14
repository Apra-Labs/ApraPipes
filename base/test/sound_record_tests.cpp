#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "test_utils.h"
#include "PipeLine.h"
#include "FileWriterModule.h"
#include "AudioCaptureSrc.h"
#include "ExternalSinkModule.h"
#include "Module.h"
#include<iostream>
#include<fstream>
#include<vector>

BOOST_AUTO_TEST_SUITE(sound_record_tests)

BOOST_AUTO_TEST_CASE(recordMono, *boost::unit_test::disabled())
{
	std::vector<std::string> audioFiles = { "./data/AudiotestMono.wav" };
	Test_Utils::FileCleaner f(audioFiles);
    // Manual test, listen to the file on audacity to for sanity check
    Logger::setLogLevel(boost::log::trivial::severity_level::info);

    auto time_to_run = Test_Utils::getArgValue("s", "10");
    auto n_seconds = atoi(time_to_run.c_str());
    auto sampling_rate = 48000;
    auto channels = 1;
    auto sample_size_byte = 2;

    AudioCaptureSrcProps sourceProps(sampling_rate,channels,0,200);
    auto source = boost::shared_ptr<Module>(new AudioCaptureSrc(sourceProps));

    auto outputFile = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps(audioFiles[0], true)));
    source->setNext(outputFile);

    PipeLine p("test");
    p.appendModule(source);
    p.init();
    p.run_all_threaded();
    std::this_thread::sleep_for(std::chrono::seconds(n_seconds));

    ifstream in_file_mono(audioFiles[0], ios::binary);
    in_file_mono.seekg(0, ios::end);
    int file_size_mono = in_file_mono.tellg();
    BOOST_TEST((channels * sampling_rate * sample_size_byte * n_seconds >= file_size_mono - (file_size_mono * 0.02) &&
                channels * sampling_rate * sample_size_byte * n_seconds <= file_size_mono + (file_size_mono * 0.02)));
    p.stop();
    p.term();
    p.wait_for_all();
	in_file_mono.close();
}

BOOST_AUTO_TEST_CASE(recordStereo, *boost::unit_test::disabled())
{
	std::vector<std::string> audioFiles = { "./data/AudiotestStereo.wav" };
	Test_Utils::FileCleaner f(audioFiles);
    Logger::setLogLevel(boost::log::trivial::severity_level::info);

    auto time_to_run = Test_Utils::getArgValue("s", "10");
    auto n_seconds = atoi(time_to_run.c_str());
    auto sampling_rate = 48000;
    auto channels = 2;
    auto sample_size_byte = 2;

    AudioCaptureSrcProps sourceProps(sampling_rate,channels,0,200);
    auto source = boost::shared_ptr<AudioCaptureSrc>(new AudioCaptureSrc(sourceProps));

    auto outputFile = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps(audioFiles[0], true)));
    source->setNext(outputFile);

    PipeLine p("test");
    p.appendModule(source);
    p.init();
    p.run_all_threaded();
    std::this_thread::sleep_for(std::chrono::seconds(n_seconds));

    ifstream in_file_stereo(audioFiles[0], ios::binary);
    in_file_stereo.seekg(0, ios::end);
    int file_size_stereo = in_file_stereo.tellg();
    BOOST_TEST((channels * sampling_rate * sample_size_byte * n_seconds >= file_size_stereo - (file_size_stereo * 0.02) &&
                channels * sampling_rate * sample_size_byte * n_seconds <= file_size_stereo + (file_size_stereo * 0.02)));
    p.stop();
    p.term();
    p.wait_for_all();
	in_file_stereo.close();
}
BOOST_AUTO_TEST_CASE(recordMonoStereo, *boost::unit_test::disabled())
{
	std::vector<std::string> audioFiles = { "./data/AudiotestMono.wav", "./data/AudiotestStereo.wav" };
	Test_Utils::FileCleaner f(audioFiles);
    Logger::setLogLevel(boost::log::trivial::severity_level::info);

    auto time_to_run = Test_Utils::getArgValue("s", "10");
    auto n_seconds = atoi(time_to_run.c_str());
    auto sampling_rate = 44000;
    auto channels = 1;
    auto sample_size_byte = 2;
    AudioCaptureSrcProps sourceProps(sampling_rate,channels,0,200);
    auto source = boost::shared_ptr<AudioCaptureSrc>(new AudioCaptureSrc(sourceProps));

    auto outputFile = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps(audioFiles[0], true)));
    source->setNext(outputFile);

    auto outputFile2 = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps(audioFiles[1], true)));
    source->setNext(outputFile2,false);


    PipeLine p("test");
    p.appendModule(source);
    p.init();
    p.run_all_threaded();
    std::this_thread::sleep_for(std::chrono::seconds(n_seconds));

    ifstream in_file_mono(audioFiles[0], ios::binary);
    in_file_mono.seekg(0, ios::end);
    int file_size_mono = in_file_mono.tellg();
    BOOST_TEST((channels * sampling_rate * sample_size_byte * n_seconds >= file_size_mono - (file_size_mono * 0.02) &&
                channels * sampling_rate * sample_size_byte * n_seconds <= file_size_mono + (file_size_mono * 0.02)));

    auto currentProps = source->getProps();
    currentProps.channels = 2;
    currentProps.sampleRate = 48000;
    source->setProps(currentProps);
    source->relay(outputFile,false);
    source->relay(outputFile2,true);
    std::this_thread::sleep_for(std::chrono::seconds(n_seconds));
	in_file_mono.close();

    ifstream in_file_stereo(audioFiles[1], ios::binary);
    in_file_stereo.seekg(0, ios::end);
    int file_size_stereo = in_file_stereo.tellg();
    BOOST_TEST((channels * sampling_rate * sample_size_byte * n_seconds >= file_size_stereo - (file_size_stereo * 0.02) &&
                channels * sampling_rate * sample_size_byte * n_seconds <= file_size_stereo + (file_size_stereo * 0.02)));
    p.stop();
    p.term();
    p.wait_for_all();
	in_file_stereo.close();
}
BOOST_AUTO_TEST_SUITE_END()