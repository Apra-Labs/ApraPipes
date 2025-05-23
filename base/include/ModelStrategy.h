#pragma once

#include "ImageToTextXForm.h"
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

  // Model configuration constants
  static constexpr int DEFAULT_CONTEXT_SIZE = 4096;
  static constexpr int DEFAULT_BATCH_SIZE = 512;
  static constexpr float DEFAULT_TEMPERATURE = 0.8f;
  static constexpr int DEFAULT_MAX_TOKENS = 256;

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

class ImageToTextModelStrategy : public ModelStrategy
{
public:
  ImageToTextModelStrategy(ImageToTextXFormProps props);
  ~ImageToTextModelStrategy();

  bool initStrategy() override;
  bool termStrategy() override;
};

class LlavaTextToTextModelStrategy : public ModelStrategy
{
public:
  LlavaTextToTextModelStrategy(ImageToTextXFormProps props);
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
    return boost::make_shared<ImageToTextModelStrategy>(props);
  case ModelStrategyType::LLAVA_TEXT_TO_TEXT:
    return boost::make_shared<LlavaTextToTextModelStrategy>(props);
  default:
    return boost::make_shared<LlavaTextToTextModelStrategy>(props);
    break;
  }
}
