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
#include "SceneDescriptorXForm.h"
#include "ModelStrategy.h"
#include "Module.h"
#include "ExternalSinkModule.h"

BOOST_AUTO_TEST_SUITE(sceneDescriptorXForm_tests)

BOOST_AUTO_TEST_CASE(testing)
{
  std::vector<std::string> sceneDescriptorOutText = { "./data/sceneDescriptor_out.txt" };
  Test_Utils::FileCleaner f(sceneDescriptorOutText);

	Logger::setLogLevel(boost::log::trivial::severity_level::info);

  auto fileReaderProps = FileReaderModuleProps("./data/1280x960.jpg");
  fileReaderProps.readLoop = false;
  auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
  auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::ENCODED_IMAGE));
  auto pinId = fileReader->addOutputPin(metadata);
  
  auto sceneDescriptorProps = SceneDescriptorXFormProps(SceneDescriptorXFormProps::SceneDescriptorStrategy::LLAVA);
  auto sceneDescriptor = boost::shared_ptr<SceneDescriptorXForm>(new SceneDescriptorXForm(sceneDescriptorProps));
  fileReader->setNext(sceneDescriptor);

  auto outputFile = boost::shared_ptr<FileWriterModule>(new FileWriterModule(FileWriterModuleProps(sceneDescriptorOutText[0], false)));
  sceneDescriptor->setNext(outputFile);

  BOOST_TEST(fileReader->init());
  BOOST_TEST(sceneDescriptor->init());
  BOOST_TEST(outputFile->init());

  fileReader->step();
  sceneDescriptor->step();
  outputFile->step();

  std::ifstream in_file_text(sceneDescriptorOutText[0]);
  std::ostringstream  buffer;
  buffer << in_file_text.rdbuf();
  in_file_text.close();
}

BOOST_AUTO_TEST_SUITE_END()