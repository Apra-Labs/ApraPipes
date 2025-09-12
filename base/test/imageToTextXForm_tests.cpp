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
#include "ImageToTextXForm.h"
#include "ModelStrategy.h"
#include "Module.h"
#include "ExternalSinkModule.h"

BOOST_AUTO_TEST_SUITE(imageToTextXForm_tests, *boost::unit_test::disabled())

BOOST_AUTO_TEST_CASE(testing)
{
    std::vector<std::string> imageToTextOutText = {
        "./data/imageToText_out.txt"};
    Test_Utils::FileCleaner f(imageToTextOutText);

    Logger::setLogLevel(boost::log::trivial::severity_level::info);

    auto fileReaderProps = FileReaderModuleProps("./data/1280x960.jpg");
    fileReaderProps.readLoop = false;
    auto fileReader = boost::shared_ptr<FileReaderModule>(
        new FileReaderModule(fileReaderProps));
    auto metadata =
        framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
    auto pinId = fileReader->addOutputPin(metadata);

    auto imageToTextProps = ImageToTextXFormProps(
        ImageToTextXFormProps::ModelStrategyType::LLAVA,
        "./data/llm/llava/llava-v1.6-7b/mmproj-model-f16.gguf",
        "./data/llm/llava/llava-v1.6-7b/llava-v1.6-mistral-7b.Q8_0.gguf",
        "A chat between a curious human and an artificial intelligence assistant.  The assistant gives helpful, detailed, and polite answers to the human's questions.\nUSER:",
        "<image> Describe the image", 10);
    auto imageToText = boost::shared_ptr<ImageToTextXForm>(
        new ImageToTextXForm(imageToTextProps));
    fileReader->setNext(imageToText);

    auto outputFile = boost::shared_ptr<FileWriterModule>(new FileWriterModule(
        FileWriterModuleProps(imageToTextOutText[0], false)));
    imageToText->setNext(outputFile);
    fileReader->play(true);
    BOOST_TEST(fileReader->init());
    BOOST_TEST(imageToText->init());
    BOOST_TEST(outputFile->init());

    fileReader->step();
    imageToText->step();
    outputFile->step();

    std::ifstream in_file_text(imageToTextOutText[0]);
    std::ostringstream buffer;
    buffer << in_file_text.rdbuf();
    in_file_text.close();
}

BOOST_AUTO_TEST_SUITE_END()
