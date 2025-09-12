#include "LlmModelAbstract.h"

LlmModelAbstractProps::LlmModelAbstractProps()
{
  modelArchitecture = ModelArchitectureType::TRANSFORMER;
  inputTypes = {FrameMetadata::FrameType::TEXT};
  outputTypes = {FrameMetadata::FrameType::TEXT};
  useCases = {UseCase::TEXT_TO_TEXT};
  qlen = 20;
}

LlmModelAbstractProps::LlmModelAbstractProps(
    ModelArchitectureType _modelArchitecture,
    std::vector<FrameMetadata::FrameType> _inputTypes,
    std::vector<FrameMetadata::FrameType> _outputTypes,
    std::vector<UseCase> _useCases)
{
  modelArchitecture = _modelArchitecture;
  inputTypes = _inputTypes;
  outputTypes = _outputTypes;
  useCases = _useCases;
  qlen = 20;
}

LlmModelAbstract::LlmModelAbstract(std::string _modelName,
                                   LlmModelAbstractProps _props)
    : modelName(_modelName)
{
  mQue.reset(new FrameContainerQueue(_props.qlen));
  mProps.reset(new LlmModelAbstractProps(_props));
}

LlmModelAbstract::~LlmModelAbstract() {}

bool LlmModelAbstract::init()
{
  mQue->accept();
  return true;
}

bool LlmModelAbstract::term()
{
  mQue->clear();
  return true;
}

bool LlmModelAbstract::step(frame_container &outputFrameContainer,
                            std::function<frame_sp(size_t)> makeFrame)
{
  auto inputFrameContainer = mQue->pop();
  if (inputFrameContainer.size() == 0)
  {
    return true;
  }
  bool ret =
      modelInference(inputFrameContainer, outputFrameContainer, makeFrame);
  return ret;
}

bool LlmModelAbstract::push(frame_container &inputFrameContainer,
                            frame_container &outputFrameContainer,
                            std::function<frame_sp(size_t)> makeFrame)
{
  mQue->push(inputFrameContainer);
  while (mQue->size() != 0)
  {
    if (!step(outputFrameContainer, makeFrame))
    {
      LOG_ERROR << "Step failed";
      return false;
    }
  }
  return true;
}