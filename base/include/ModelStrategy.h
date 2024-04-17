# pragma once

#include "Module.h"
#include "ClipEncoder.h"
#include "Llava.h"

class ModelStrategy {
public:
  enum ModelStrategyType {
    LLAMA_TEXT_TO_TEXT = 0,
    LLAVA_TEXT_TO_TEXT,
    LLAVA_SCENE_DESCRIPTOR
  };

  static boost::shared_ptr<ModelStrategy> create(ModelStrategyType type);

  ModelStrategy();
  virtual ~ModelStrategy();

  virtual bool initStrategy() = 0;
  virtual bool termStrategy() = 0;
public:
  boost::shared_ptr<EncoderModelAbstract> encoderModel;
	boost::shared_ptr<LlmModelAbstract> llmModel;
};

class SceneDescriptorModelStrategy : public ModelStrategy {
public:
  SceneDescriptorModelStrategy();
  ~SceneDescriptorModelStrategy();

  bool initStrategy() override;
  bool termStrategy() override;
};

class LlavaTextToTextModelStrategy : public ModelStrategy {
public:
  LlavaTextToTextModelStrategy();
  ~LlavaTextToTextModelStrategy();

  bool initStrategy() override;
  bool termStrategy() override;
};