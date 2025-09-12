#pragma once

#include "LlmModelAbstract.h"

class LlavaProps : public LlmModelAbstractProps {
public:
  LlavaProps(std::string _modelPath, std::string _systemPrompt,
             std::string _userPrompt, int _contextSize, int _batchSize,
             float _degreeOfRandomness, int _gpuLayers, int _predictionLength);

  std::string modelPath;
  std::string systemPrompt;
  std::string userPrompt;
  int contextSize;
  int batchSize;
  float degreeOfRandomness;
  int gpuLayers;
  int predictionLength;

  size_t getSerializeSize() {
    return LlmModelAbstractProps::getSerializeSize() + sizeof(modelPath) +
           sizeof(systemPrompt) + sizeof(userPrompt) + sizeof(float) +
           4 * sizeof(int);
  }

private:
  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive &ar, const unsigned int version) {
    ar &boost::serialization::base_object<LlmModelAbstractProps>(*this);
    ar & modelPath & systemPrompt & userPrompt;
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
  bool modelInference(frame_container &inputFrameContainer,
                      frame_container &outputFrameContainer,
                      std::function<frame_sp(size_t)> makeFrame) override;
  bool validateUseCase(UseCase useCase) override;
  size_t getFrameSize() override;
  void storeFrames(frame_sp &frame);

private:
  class Detail;
  boost::shared_ptr<Detail> mDetail;
};