#pragma once

#include "LlmModelAbstract.h"

class LlavaProps : public LlmModelAbstractProps {
public:
  LlavaProps(std::string _modelPath, std::string _prompt,
             int _contextSize, int _batchSize, float _degreeOfRandomness, int _gpuLayers, int _predictionLength) {
    
    /* Set LLM Model Base Class Properties for each model*/
    modelArchitecture = ModelArchitectureType::TRANSFORMER;
    inputTypes = {DataType::TEXT, DataType::IMAGE_EMBEDDING};
    outputTypes = {DataType::TEXT};
    useCases = {UseCase::TEXT_TO_TEXT, UseCase::OCR, UseCase::SCENE_DESCRIPTOR};

    /*Unique Model Properties*/
    modelPath = _modelPath;
    prompt = _prompt;
    degreeOfRandomness = _degreeOfRandomness;
    contextSize = _contextSize;
    batchSize = _batchSize;
    gpuLayers = _gpuLayers;
    predictionLength = _predictionLength;
  }

  std::string modelPath;
  std::string prompt;
  int contextSize;
  int batchSize;
  float degreeOfRandomness;
  int gpuLayers;
  int predictionLength;

  size_t getSerializeSize() {
    return LlmModelAbstractProps::getSerializeSize() + sizeof(modelPath) +
           sizeof(prompt) + sizeof(float) + 4 * sizeof(int);
  }

private:
  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive &ar, const unsigned int version) {
    ar &boost::serialization::base_object<ModuleProps>(*this);
    ar & modelPath & prompt;
    ar & degreeOfRandomness;
    ar & contextSize & batchSize & gpuLayers & predictionLength;
  }
};

class Llava : public LlmModelAbstract {
public:
  Llava(LlavaProps _props);
  virtual ~Llava();
  bool modelInit() override;
  bool modelTerm() override;
  bool modelInference(frame_container& frames) override;
  bool validateUseCase(LlmModelAbstractProps::UseCase useCase) override;
  size_t getFrameSize() override;
  void getFrames(frame_sp& frame) override; 

private:
  class Detail;
  boost::shared_ptr<Detail> mDetail;
};