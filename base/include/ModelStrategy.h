#pragma once

#include "SceneDescriptorXForm.h"
#include "EncoderModelAbstract.h"
#include "LlmModelAbstract.h"

class ModelStrategy
{
public:
  enum ModelStrategyType
  {
    LLAMA_TEXT_TO_TEXT = 0,
    LLAVA_TEXT_TO_TEXT,
    LLAVA_SCENE_DESCRIPTOR
  };

  template <typename T>
  static boost::shared_ptr<ModelStrategy> create(ModelStrategyType type,
                                                 T &props);

  ModelStrategy();
  virtual ~ModelStrategy();

  virtual bool initStrategy() = 0;
  virtual bool termStrategy() = 0;

public:
  boost::shared_ptr<EncoderModelAbstract> encoderModel;
  boost::shared_ptr<LlmModelAbstract> llmModel;
};

class SceneDescriptorModelStrategy : public ModelStrategy
{
public:
  SceneDescriptorModelStrategy(SceneDescriptorXFormProps props);
  ~SceneDescriptorModelStrategy();

  bool initStrategy() override;
  bool termStrategy() override;
};

class LlavaTextToTextModelStrategy : public ModelStrategy
{
public:
  LlavaTextToTextModelStrategy();
  ~LlavaTextToTextModelStrategy();

  bool initStrategy() override;
  bool termStrategy() override;
};

template <typename T>
boost::shared_ptr<ModelStrategy> ModelStrategy::create(ModelStrategyType type,
                                                       T &props)
{
  switch (type)
  {
  case ModelStrategyType::LLAVA_SCENE_DESCRIPTOR:
    return boost::make_shared<SceneDescriptorModelStrategy>(props);
  case ModelStrategyType::LLAVA_TEXT_TO_TEXT:
    return boost::make_shared<LlavaTextToTextModelStrategy>();
  default:
    return boost::make_shared<LlavaTextToTextModelStrategy>();
    break;
  }
}