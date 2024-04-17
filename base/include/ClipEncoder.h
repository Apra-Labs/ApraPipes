#pragma once

#include "EncoderModelAbstract.h"

class ClipEncoderProps : public EncoderModelAbstractProps {
public:
  ClipEncoderProps(std::string _modelPath) {
    /* Set LLM Model Base Class Properties for each model*/
    modelArchitecture = ModelArchitectureType::VIT;
    inputTypes = {DataType::TEXT, DataType::IMAGE};
    outputTypes = {DataType::IMAGE_EMBEDDING};
    useCases = {UseCase::OCR, UseCase::SCENE_DESCRIPTOR};

    /*Unique Model Properties*/
    modelPath = _modelPath;
  }

  std::string modelPath;

  size_t getSerializeSize() {
    return EncoderModelAbstractProps::getSerializeSize() + sizeof(modelPath);
  }

private:
  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive &ar, const unsigned int version) {
    ar &boost::serialization::base_object<ModuleProps>(*this);
    ar & modelPath;
  }
};

class ClipEncoder : public EncoderModelAbstract {
public:
  ClipEncoder(ClipEncoderProps _props);
  virtual ~ClipEncoder();
  bool modelInit() override;
  bool modelTerm() override;
  bool modelInference(frame_container& frames) override;
  bool validateUseCase(EncoderModelAbstractProps::UseCase useCase) override;
  size_t getFrameSize() override;
  void getFrames(frame_sp& frame) override;

private:
  class Detail;
  boost::shared_ptr<Detail> mDetail;
};