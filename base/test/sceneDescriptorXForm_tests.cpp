#include <boost/test/unit_test.hpp>
#include "stdafx.h"
#include <fstream>
#include <vector>

#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "test_utils.h"
#include "PipeLine.h"
#include "FileWriterModule.h"
#include "FileReaderModule.h"
#include "SceneDescriptorXForm.h"
#include "ModelStrategy.h"
#include "Module.h"
#include "ExternalSinkModule.h"

BOOST_AUTO_TEST_SUITE(sceneDescriptorXForm_tests)

BOOST_AUTO_TEST_CASE(testing)
{
    std::vector<std::string> sceneDescriptorOutText = {
        "./data/sceneDescriptor_out.txt"};
    Test_Utils::FileCleaner f(sceneDescriptorOutText);

    Logger::setLogLevel(boost::log::trivial::severity_level::info);

    auto fileReaderProps = FileReaderModuleProps("./data/theft/Image_???.jpeg");
    fileReaderProps.readLoop = false;
    auto fileReader = boost::shared_ptr<FileReaderModule>(
        new FileReaderModule(fileReaderProps));
    auto metadata =
        framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
    auto pinId = fileReader->addOutputPin(metadata);

    auto sceneDescriptorProps = SceneDescriptorXFormProps(
        SceneDescriptorXFormProps::ModelStrategyType::LLAVA,
        "./data/llm/llava/llava-v1.6-7b/mmproj-model-f16.gguf",
        "./data/llm/llava/llava-v1.6-7b/llava-v1.6-mistral-7b.Q8_0.gguf",
        "You are a Spy AI. Your task is to analyze the provided images and "
        "identify any potential suspicious activity. Look for anomalies, unusual "
        "behavior, or anything that raises concerns. Describe what you find and "
        "explain why it might be considered suspicious. Your analysis should "
        "consider various factors such as context, environmental cues, and human "
        "behavior patterns. Be detailed and provide insights into your thought "
        "process as you assess the image and don't hallucinate.\nUSER:",
        "Tell me the percentage of suspicious activity based on the context of the images", 10);
    auto sceneDescriptor = boost::shared_ptr<SceneDescriptorXForm>(
        new SceneDescriptorXForm(sceneDescriptorProps));
    fileReader->setNext(sceneDescriptor);

    auto outputFile = boost::shared_ptr<FileWriterModule>(new FileWriterModule(
        FileWriterModuleProps(sceneDescriptorOutText[0], false)));
    sceneDescriptor->setNext(outputFile);

    BOOST_TEST(fileReader->init());
    BOOST_TEST(sceneDescriptor->init());
    BOOST_TEST(outputFile->init());

    for(int i = 0; i < 4; i++){
        fileReader->step();
        sceneDescriptor->step();
        outputFile->step();
    }

    std::ifstream in_file_text(sceneDescriptorOutText[0]);
    std::ostringstream buffer;
    buffer << in_file_text.rdbuf();
    in_file_text.close();
}

BOOST_AUTO_TEST_SUITE_END()