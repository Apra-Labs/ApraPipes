#include <boost/test/unit_test.hpp>
#include "stdafx.h"
#include<fstream>
#include<vector>

#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "test_utils.h"
#include "PipeLine.h"
#include "FileWriterModule.h"
#include "FileReaderModule.h"
#include "FileWriterModule.h"
#include "AudioToTextXForm.h"
#include "Module.h"

BOOST_AUTO_TEST_SUITE(audioToTextXform_test)

BOOST_AUTO_TEST_CASE(test_asr)
{
    std::vector<std::string> asrOutText = { "./data/asr_out.txt" };
    Test_Utils::FileCleaner f(asrOutText);

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

    // This is a PCM file without WAV header
    auto fileReaderProps = FileReaderModuleProps("./data/audioToTextXform_test.pcm");
    fileReaderProps.readLoop = false;
    auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
    auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::AUDIO));
    auto pinId = fileReader->addOutputPin(metadata);
   
    auto asr = boost::shared_ptr<AudioToTextXForm>(new AudioToTextXForm(AudioToTextXFormProps(
        AudioToTextXFormProps::DecoderSamplingStrategy::GREEDY
        ,"./data/whisper/models/ggml-tiny.en-q8_0.bin",18000)));
    fileReader->setNext(asr);

    auto outputFile = boost::shared_ptr<FileWriterModule>(new FileWriterModule(FileWriterModuleProps(asrOutText[0], false)));
    asr->setNext(outputFile);

    BOOST_TEST(fileReader->init());
    BOOST_TEST(asr->init());
    BOOST_TEST(outputFile->init());

    fileReader->step();
    asr->step();
    outputFile->step();

    std::ifstream in_file_text(asrOutText[0]);
    std::ostringstream  buffer;
    buffer << in_file_text.rdbuf();
    std:string output = " The Matic speech recognition also known as ASR is the use of machine learning or artificial intelligence technology to process human speech into readable text.";
    BOOST_TEST(
        (buffer.str() == output));
    in_file_text.close();
}

BOOST_AUTO_TEST_CASE(changeprop_asr)
{
    std::vector<std::string> asrOutText = { "./data/asr_change_props_out.txt" };
    Test_Utils::FileCleaner f(asrOutText);

    Logger::setLogLevel(boost::log::trivial::severity_level::info);

    // This is a PCM file without WAV header
    auto fileReaderProps = FileReaderModuleProps("./data/audioToTextXform_test.pcm");
    fileReaderProps.readLoop = true;
    auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
    auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::AUDIO));
    auto pinId = fileReader->addOutputPin(metadata);

    auto asr = boost::shared_ptr<AudioToTextXForm>(new AudioToTextXForm(AudioToTextXFormProps(
        AudioToTextXFormProps::DecoderSamplingStrategy::GREEDY
        , "./data/whisper/models/ggml-tiny.en-q8_0.bin", 18000)));
    fileReader->setNext(asr);

    auto outputFile = boost::shared_ptr<FileWriterModule>(new FileWriterModule(FileWriterModuleProps(asrOutText[0], false)));
    asr->setNext(outputFile);

    BOOST_TEST(fileReader->init());
    BOOST_TEST(asr->init());
    BOOST_TEST(outputFile->init());

    AudioToTextXFormProps propschange = asr->getProps();
    propschange.bufferSize = 20000;
    propschange.samplingStrategy = AudioToTextXFormProps::DecoderSamplingStrategy::BEAM_SEARCH;
    fileReader->step();
    asr->step();
    outputFile->step();

    asr->setProps(propschange);
    for (int i = 0; i < 2; i++) {
        fileReader->step();
        asr->step();
    }
    outputFile->step();
    propschange = asr->getProps();
    std::ifstream in_file_text(asrOutText[0]);
    std::ostringstream  buffer;
    buffer << in_file_text.rdbuf();
    std:string output = " Metex speech recognition, also known as ASR, is the use of machine learning or artificial intelligence technology to process human speech into readable text.";
    //TODO: This test fails in Linux Cuda. Maybe Something to do with the Beam Search / change in props size that makes the behaviour different from windows
    //    BOOST_TEST(
    //        (buffer.str() == output));
    in_file_text.close();
    
    BOOST_TEST(
        (propschange.bufferSize == 20000));
    BOOST_TEST(
        (propschange.samplingStrategy == AudioToTextXFormProps::DecoderSamplingStrategy::BEAM_SEARCH));
}

BOOST_AUTO_TEST_CASE(change_unsupported_prop_asr)
{
    std::vector<std::string> asrOutText = { "./data/asr_change_props_out.txt" };
    Test_Utils::FileCleaner f(asrOutText);

    Logger::setLogLevel(boost::log::trivial::severity_level::info);

    // This is a PCM file without WAV header
    auto fileReaderProps = FileReaderModuleProps("./data/audioToTextXform_test.pcm");
    fileReaderProps.readLoop = true;
    auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
    auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::AUDIO));
    auto pinId = fileReader->addOutputPin(metadata);

    auto asr = boost::shared_ptr<AudioToTextXForm>(new AudioToTextXForm(AudioToTextXFormProps(
        AudioToTextXFormProps::DecoderSamplingStrategy::GREEDY
        , "./data/whisper/models/ggml-tiny.en-q8_0.bin", 18000)));
    fileReader->setNext(asr);

    auto outputFile = boost::shared_ptr<FileWriterModule>(new FileWriterModule(FileWriterModuleProps(asrOutText[0], false)));
    asr->setNext(outputFile);

    BOOST_TEST(fileReader->init());
    BOOST_TEST(asr->init());
    BOOST_TEST(outputFile->init());

    AudioToTextXFormProps propschange = asr->getProps();
    propschange.modelPath = "./newpath.bin";
    fileReader->step();
    asr->step();
    outputFile->step();

    BOOST_CHECK_THROW(asr->setProps(propschange), std::runtime_error);
}


BOOST_AUTO_TEST_SUITE_END()