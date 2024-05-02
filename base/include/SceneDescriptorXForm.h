#pragma once

#include "Module.h"

class SceneDescriptorXFormProps : public ModuleProps
{
public:
  enum ModelStrategyType
  {
    LLAVA = 0
  };

  SceneDescriptorXFormProps(ModelStrategyType _modelStrategyType,
                            std::string _encoderModelPath,
                            std::string _llmModelPath,
                            std::string _systemPrompt, std::string _userPrompt,
                            int _gpuLayers);

  size_t getSerializeSize()
  {
    return ModuleProps::getSerializeSize() + sizeof(modelStrategyType) +
           sizeof(encoderModelPath) + sizeof(llmModelPath) +
           sizeof(systemPrompt) + sizeof(userPrompt) + sizeof(gpuLayers);
  }

  ModelStrategyType modelStrategyType;
  std::string encoderModelPath;
  std::string llmModelPath;
  std::string systemPrompt;
  std::string userPrompt;
  int gpuLayers;

private:
  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive &ar, const unsigned int version)
  {
    ar &boost::serialization::base_object<ModuleProps>(*this);
    ar & modelStrategyType;
    ar & encoderModelPath & llmModelPath;
    ar & systemPrompt & userPrompt;
    ar & gpuLayers;
  }
};

class SceneDescriptorXForm : public Module
{
public:
  SceneDescriptorXForm(SceneDescriptorXFormProps _props);
  virtual ~SceneDescriptorXForm();
  bool init();
  bool term();
  void setProps(SceneDescriptorXFormProps &props);
  SceneDescriptorXFormProps getProps();

protected:
  bool process(frame_container &frames);
  bool processSOS(frame_sp &frame);
  bool validateInputPins();
  bool validateOutputPins();
  void addInputPin(framemetadata_sp &metadata, string &pinId);
  bool handlePropsChange(frame_sp &frame);

private:
  void setMetadata(framemetadata_sp &metadata);
  class Detail;
  boost::shared_ptr<Detail> mDetail;
};