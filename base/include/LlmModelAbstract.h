#pragma once

#include "Module.h"

class LlmModelAbstractProps {
public:
  enum ModelArchitectureType {
    TRANSFORMER = 0,
    ENCODERDECODER,
    CASUALDECODER,
    PREFIXDECODER
  };

  enum DataType { TEXT = 0, IMAGE, AUDIO, TEXT_EMBEDDING, IMAGE_EMBEDDING, AUDIO_EMBEDDING };

  enum UseCase { TEXT_TO_TEXT = 0, SCENE_DESCRIPTOR, OCR };

  LlmModelAbstractProps() {
    modelArchitecture = ModelArchitectureType::TRANSFORMER;
    inputTypes = {DataType::TEXT};
    outputTypes = {DataType::TEXT};
    useCases = {UseCase::TEXT_TO_TEXT};
    qlen = 20;
  }

  LlmModelAbstractProps(ModelArchitectureType _modelArchitecture,
                std::vector<DataType> _inputTypes,
                std::vector<DataType> _outputTypes,
                std::vector<UseCase> _useCases) {
    modelArchitecture = _modelArchitecture;
    inputTypes = _inputTypes;
    outputTypes = _outputTypes;
    useCases = _useCases;
    qlen = 20;
  }

  size_t getSerializeSize() {
    return sizeof(modelArchitecture) + sizeof(inputTypes) + sizeof(outputTypes) + sizeof(useCases) + sizeof(qlen);
  }

  ModelArchitectureType modelArchitecture;
  std::vector<DataType> inputTypes;
  std::vector<DataType> outputTypes;
  std::vector<UseCase> useCases;
  size_t qlen;

private:
  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive &ar, const unsigned int version) {
    ar &boost::serialization::base_object<ModuleProps>(*this);
    ar & modelArchitecture;
    ar & inputTypes;
    ar & outputTypes;
    ar & useCases;
    ar & qlen;
  }
};

class LlmModelAbstract {
public:
  LlmModelAbstract(std::string name, LlmModelAbstractProps props);
  ~LlmModelAbstract();

  std::string getMyName() {
    return myName;
  }

  boost::shared_ptr<FrameContainerQueue> getQue() {
    return mQue;
  }

	virtual bool modelInit() = 0;
	virtual bool modelTerm() = 0;
  virtual bool modelInference(frame_container& frameContainer) {return false;}
  virtual size_t getFrameSize() = 0;
  virtual void getFrames(frame_sp& frame) = 0;

  virtual bool validateUseCase(LlmModelAbstractProps::UseCase useCase) = 0;

	bool init();
	bool term();
  bool step();
  bool push(frame_container& frameContainer);

private:
  std::string myName;
  boost::shared_ptr<FrameContainerQueue> mQue;
  boost::shared_ptr<LlmModelAbstractProps> mProps;
};