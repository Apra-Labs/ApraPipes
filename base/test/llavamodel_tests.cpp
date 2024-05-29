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
#include "LlmModelAbstract.h"
#include "Llava.h"
#include "Module.h"
#include "ExternalSinkModule.h"

BOOST_AUTO_TEST_SUITE(llavamodel_test)

BOOST_AUTO_TEST_CASE(llava_init)
{
  auto llavaProps = LlavaProps(
      "./data/llm/llava/llava-v1.6-7b/llava-v1.6-mistral-7b.Q8_0.gguf",
      "A chat between a curious human and an artificial intelligence "
      "assistant.  The assistant gives helpful, detailed, and polite answers "
      "to the human's questions.\nUSER:",
      "Describe the image", 2048, 512, 0.8, 10, 256);
  auto llavaModel = boost::shared_ptr<LlmModelAbstract>(new Llava(llavaProps));
  llavaModel->modelInit();
  llavaModel->modelTerm();
}

BOOST_AUTO_TEST_SUITE_END()