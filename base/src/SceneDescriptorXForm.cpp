#include "SceneDescriptorXForm.h"
#include "ModelStrategy.h"

SceneDescriptorXFormProps::SceneDescriptorXFormProps(
    ModelStrategyType _modelStrategyType, std::string _encoderModelPath,
    std::string _llmModelPath, std::string _systemPrompt,
    std::string _userPrompt, int _gpuLayers)
{
  modelStrategyType = _modelStrategyType;
  encoderModelPath = _encoderModelPath;
  llmModelPath = _llmModelPath;
  systemPrompt = _systemPrompt;
  userPrompt = _userPrompt;
  gpuLayers = _gpuLayers;
}

class SceneDescriptorXForm::Detail
{
public:
  Detail(SceneDescriptorXFormProps &_props) : mProps(_props)
  {
    setModelStrategy(_props);
  }
  ~Detail() {}

  void setProps(SceneDescriptorXFormProps &_props) { mProps = _props; }

  void setModelStrategy(SceneDescriptorXFormProps &_props)
  {
    switch (_props.modelStrategyType)
    {
    case SceneDescriptorXFormProps::ModelStrategyType::LLAVA:
      modelStrategyType =
          ModelStrategy::ModelStrategyType::LLAVA_SCENE_DESCRIPTOR;
      break;
    default:
      LOG_ERROR << "Please choose a valid model strategy!!!\n";
      break;
    }

    modelStrategy = ModelStrategy::create(modelStrategyType, _props);
  }

public:
  framemetadata_sp mOutputMetadata;
  std::string mOutputPinId;
  SceneDescriptorXFormProps mProps;
  ModelStrategy::ModelStrategyType modelStrategyType;
  boost::shared_ptr<ModelStrategy> modelStrategy;
};

SceneDescriptorXForm::SceneDescriptorXForm(SceneDescriptorXFormProps _props)
    : Module(TRANSFORM, "SceneDescriptorXForm", _props)
{
  mDetail.reset(new Detail(_props));
}

SceneDescriptorXForm::~SceneDescriptorXForm() {}

bool SceneDescriptorXForm::validateInputPins()
{
  if (getNumberOfInputPins() != 1)
  {
    LOG_ERROR << "<" << getId()
              << ">::validateInputPins size is expected to be 1. Actual<"
              << getNumberOfInputPins() << ">";
    return false;
  }

  framemetadata_sp metadata = getFirstInputMetadata();

  FrameMetadata::FrameType frameType = metadata->getFrameType();

  if (frameType != FrameMetadata::FrameType::ENCODED_IMAGE)
  {
    LOG_ERROR << "<" << getId()
              << ">::validateInputPins input frameType is expected to be "
                 "Audio. Actual<"
              << frameType << ">";
    return false;
  }

  FrameMetadata::MemType memType = metadata->getMemType();
  if (memType != FrameMetadata::MemType::HOST)
  {
    LOG_ERROR
        << "<" << getId()
        << ">::validateInputPins input memType is expected to be HOST. Actual<"
        << memType << ">";
    return false;
  }
  return true;
}

bool SceneDescriptorXForm::validateOutputPins()
{
  if (getNumberOfOutputPins() != 1)
  {
    LOG_ERROR << "<" << getId()
              << ">::validateOutputPins size is expected to be 1. Actual<"
              << getNumberOfOutputPins() << ">";
    return false;
  }

  framemetadata_sp metadata = getFirstOutputMetadata();
  FrameMetadata::FrameType frameType = metadata->getFrameType();
  if (frameType != FrameMetadata::FrameType::TEXT)
  {
    LOG_ERROR << "<" << getId()
              << ">::validateOutputPins input frameType is expected to be "
                 "TEXT. Actual<"
              << frameType << ">";
    return false;
  }

  return true;
}

void SceneDescriptorXForm::addInputPin(framemetadata_sp &metadata,
                                       string &pinId)
{
  Module::addInputPin(metadata, pinId);
  mDetail->mOutputMetadata =
      framemetadata_sp(new FrameMetadata(FrameMetadata::FrameType::TEXT));
  mDetail->mOutputMetadata->copyHint(*metadata.get());
  mDetail->mOutputPinId = addOutputPin(mDetail->mOutputMetadata);
}

bool SceneDescriptorXForm::init()
{
  bool ret = mDetail->modelStrategy->initStrategy();
  if (!ret)
  {
    return false;
  }
  return Module::init();
}

bool SceneDescriptorXForm::term()
{
  bool ret = mDetail->modelStrategy->termStrategy();
  if (!ret)
  {
    return false;
  }
  return Module::term();
}

bool SceneDescriptorXForm::process(frame_container &frames)
{
  /*Encoder Model*/
  frame_container clipFrames;
  mDetail->modelStrategy->encoderModel->push(frames, clipFrames, [&](size_t size) -> frame_sp {return makeFrame(size, mDetail->mOutputPinId); });

  /*LLM Model*/
  frame_container llavaFrames;
  mDetail->modelStrategy->llmModel->push(clipFrames, llavaFrames, [&](size_t size) -> frame_sp {return makeFrame(size, mDetail->mOutputPinId); });

  auto outFrame = llavaFrames.begin()->second;
  frames.insert(make_pair(mDetail->mOutputPinId, outFrame));
  send(frames);
  return true;
}

void SceneDescriptorXForm::setMetadata(framemetadata_sp &metadata)
{
  if (!metadata->isSet())
  {
    return;
  }
}

bool SceneDescriptorXForm::processSOS(frame_sp &frame)
{
  auto metadata = frame->getMetadata();
  setMetadata(metadata);
  return true;
}

SceneDescriptorXFormProps SceneDescriptorXForm::getProps()
{
  fillProps(mDetail->mProps);
  return mDetail->mProps;
}

bool SceneDescriptorXForm::handlePropsChange(frame_sp &frame)
{
  SceneDescriptorXFormProps props(
      mDetail->mProps.modelStrategyType, mDetail->mProps.encoderModelPath,
      mDetail->mProps.llmModelPath, mDetail->mProps.systemPrompt,
      mDetail->mProps.userPrompt, mDetail->mProps.gpuLayers);
  auto ret = Module::handlePropsChange(frame, props);
  mDetail->setProps(props);
  return ret;
}

void SceneDescriptorXForm::setProps(SceneDescriptorXFormProps &props)
{
  if (props.modelStrategyType != mDetail->mProps.modelStrategyType)
  {
    throw AIPException(AIP_FATAL,
                       "Model Strategy Type dynamic change not handled");
  }
  Module::addPropsToQueue(props);
}