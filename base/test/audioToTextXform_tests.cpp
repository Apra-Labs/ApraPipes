#include <boost/test/unit_test.hpp>
#include "stdafx.h"
#include<sstream>
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

    auto outputFile = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps(asrOutText[0], false)));
    asr->setNext(outputFile);

    PipeLine p("test");
    p.appendModule(fileReader);
    p.init();
    p.run_all_threaded();
    boost::this_thread::sleep_for(boost::chrono::milliseconds(10000));  // giving time to call step 
    std::ifstream in_file_text(asrOutText[0]);
    std::ostringstream buffer;
    buffer << in_file_text.rdbuf();
    p.stop();
    p.term();
    p.wait_for_all();
    BOOST_TEST(
        (buffer.str() == " The Matic speech recognition also known as ASR is the use of machine learning"
            "or artificial intelligence technology to process human speech into readable text."));
    in_file_text.close();
}

BOOST_AUTO_TEST_CASE(changeprop_asr)
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
        , "./data/whisper/models/ggml-tiny.en-q8_0.bin", 18000)));
    fileReader->setNext(asr);

    auto outputFile = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps(asrOutText[0], false)));
    asr->setNext(outputFile);

    PipeLine p("test");
    p.appendModule(fileReader);
    p.init();
    p.run_all_threaded();
    Test_Utils::sleep_for_seconds(5);

    AudioToTextXFormProps propschange = asr->getProps();
    propschange.bufferSize = 2000;
    asr->setProps(propschange);

    Test_Utils::sleep_for_seconds(5);
    std::ifstream in_file_text(asrOutText[0]);
    std::ostringstream buffer;
    buffer << in_file_text.rdbuf();
    p.stop();
    p.term();
    p.wait_for_all();
    BOOST_TEST(
        (buffer.str() == " The Matic speech recognition also known as ASR is the use of machine learning"
            "or artificial intelligence technology to process human speech into readable text."));
    in_file_text.close();
}

BOOST_AUTO_TEST_SUITE_END()