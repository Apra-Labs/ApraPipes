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
#include "ExternalSinkModule.h"

#include <unordered_map>
#include <string>
#include <cmath>

// Function to calculate the frequency of each word in a string
std::unordered_map<string, int> calculateWordFrequency(const std::string &str)
{
    std::unordered_map<std::string, int> frequencyMap;
    std::string word = "";
    for (char c : str)
    {
        if (c == ' ' || c == '.' || c == ',' || c == ';' || c == ':' || c == '!' || c == '?')
        {
            if (!word.empty())
            {
                frequencyMap[word]++;
                word = "";
            }
        }
        else
        {
            word += std::tolower(c);
        }
    }
    if (!word.empty())
    {
        frequencyMap[word]++;
    }
    return frequencyMap;
}

// Function to calculate dot product of two vectors
double dotProduct(const std::unordered_map<std::string, int> &vec1, const std::unordered_map<std::string, int> &vec2)
{
    double dotProduct = 0.0;
    for (const auto &pair : vec1)
    {
        if (vec2.count(pair.first) > 0)
        {
            dotProduct += pair.second * vec2.at(pair.first);
        }
    }
    return dotProduct;
}

// Function to calculate magnitude of a vector
double magnitude(const std::unordered_map<std::string, int> &vec)
{
    double mag = 0.0;
    for (const auto &pair : vec)
    {
        mag += std::pow(pair.second, 2);
    }
    return std::sqrt(mag);
}

// Function to calculate cosine similarity between two strings
double cosineSimilarity(const std::string &str1, const std::string &str2)
{
    unordered_map<string, int> vec1 = calculateWordFrequency(str1);
    unordered_map<string, int> vec2 = calculateWordFrequency(str2);

    double dotProd = dotProduct(vec1, vec2);
    double magVec1 = magnitude(vec1);
    double magVec2 = magnitude(vec2);

    if (magVec1 == 0 || magVec2 == 0)
    {
        return 0; // Handle division by zero
    }

    return dotProd / (magVec1 * magVec2);
}

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
    double thres = 0.95;
    BOOST_TEST(cosineSimilarity(buffer.str(), output) >= thres);
    // BOOST_TEST(buffer.str() == output);
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
    double thres = 0.95;
    BOOST_TEST(cosineSimilarity(buffer.str(), output) >= thres);
    // BOOST_TEST(buffer.str() == output);

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

BOOST_AUTO_TEST_CASE(check_eos_frame_asr)
{
    std::vector<std::string> asrOutText = { "./data/asr_check_eos_frame.txt" };
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
        ,"./data/whisper/models/ggml-tiny.en-q8_0.bin",160000)));
    fileReader->setNext(asr);

    auto outputFile = boost::shared_ptr<FileWriterModule>(new FileWriterModule(FileWriterModuleProps(asrOutText[0], false)));
    asr->setNext(outputFile);

    auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
    asr->setNext(sink);

    BOOST_TEST(fileReader->init());
    BOOST_TEST(asr->init());
    BOOST_TEST(outputFile->init());
    BOOST_TEST(sink->init());

    fileReader->step();
    asr->step();

    auto frames = sink->pop();
    auto eosframe = frames.begin()->second;
    BOOST_TEST(eosframe->isEOS());
    
    outputFile->step();

    std::ifstream in_file_text(asrOutText[0]);
    std::ostringstream  buffer;
    buffer << in_file_text.rdbuf();
    std:string output = " The Matic speech recognition also known as ASR is the use of machine learning or artificial intelligence technology to process human speech into readable text.";
    double thres = 0;
    BOOST_TEST(cosineSimilarity(buffer.str(), output) == thres);
    // BOOST_TEST(buffer.str() == output);
    in_file_text.close();
}

BOOST_AUTO_TEST_CASE(check_flushed_buffer_asr)
{
    std::vector<std::string> asrOutText = { "./data/asr_flushed_buffer.txt" };
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
        ,"./data/whisper/models/ggml-tiny.en-q8_0.bin",160000)));
    fileReader->setNext(asr);

    auto outputFile = boost::shared_ptr<FileWriterModule>(new FileWriterModule(FileWriterModuleProps(asrOutText[0], false)));
    asr->setNext(outputFile);

    auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
    asr->setNext(sink);

    BOOST_TEST(fileReader->init());
    BOOST_TEST(asr->init());
    BOOST_TEST(outputFile->init());
    BOOST_TEST(sink->init());

    fileReader->step();
    asr->step();

    auto frames = sink->pop();
    auto eosframe = frames.begin()->second;
    BOOST_TEST(eosframe->isEOS());
    
    outputFile->step();

    AudioToTextXFormProps propschange = asr->getProps();
    propschange.bufferSize = 18000;
    asr->setProps(propschange);

    for (int i = 0; i < 2; i++) {
        fileReader->step();
        asr->step();
    }
    outputFile->step();

    std::ifstream in_file_text(asrOutText[0]);
    std::ostringstream  buffer;
    buffer << in_file_text.rdbuf();
    std:string output = " The Matic speech recognition also known as ASR is the use of machine learning or artificial intelligence technology to process human speech into readable text.";
    double thres = 0.95;
    BOOST_TEST(cosineSimilarity(buffer.str(), output) >= thres);
    // BOOST_TEST(buffer.str() == output);
    in_file_text.close();
}

BOOST_AUTO_TEST_SUITE_END()