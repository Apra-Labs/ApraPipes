#pragma once
#include "stdafx.h"
#include <boost/shared_ptr.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/stream.hpp>
#include "Frame.h"
#include <boost/function.hpp>
#include "BoundBuffer.h"
#include "FrameFactory.h"
#include "CommonDefs.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Command.h"
#include "BufferMaker.h"
#include "ModelEnums.h"
#include "FrameContainerQueue.h"

class EncoderModelAbstractProps
{
public:
  EncoderModelAbstractProps();

  EncoderModelAbstractProps(ModelArchitectureType _modelArchitecture,
                            std::vector<FrameMetadata::FrameType> _inputTypes,
                            std::vector<FrameMetadata::FrameType> _outputTypes,
                            std::vector<UseCase> _useCases);

  size_t getSerializeSize()
  {
    return sizeof(modelArchitecture) + sizeof(inputTypes) +
           sizeof(outputTypes) + sizeof(useCases) + sizeof(qlen);
  }

  ModelArchitectureType modelArchitecture;
  std::vector<FrameMetadata::FrameType> inputTypes;
  std::vector<FrameMetadata::FrameType> outputTypes;
  std::vector<UseCase> useCases;
  size_t qlen;

private:
  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive &ar, const unsigned int version)
  {
    ar &boost::serialization::base_object<EncoderModelAbstractProps>(*this);
    ar & modelArchitecture;
    ar & inputTypes;
    ar & outputTypes;
    ar & useCases;
    ar & qlen;
  }
};

class EncoderModelAbstract
{
public:
  EncoderModelAbstract(std::string _modelName, EncoderModelAbstractProps props);
  ~EncoderModelAbstract();

  std::string getMyName() { return modelName; }

  boost::shared_ptr<FrameContainerQueue> getQue() { return mQue; }

  virtual bool modelInit() = 0;
  virtual bool modelTerm() = 0;
  virtual bool modelInference(frame_container &inputFrameContainer,
                              frame_container &outputFrameContainer, std::function<frame_sp(size_t)> makeFrame)
  {
    return false;
  }
  virtual size_t getFrameSize() = 0;

  virtual bool validateUseCase(UseCase useCase) = 0;

  bool init();
  bool term();
  bool step(frame_container &outputFrameContaine, std::function<frame_sp(size_t)> makeFrame);
  bool push(frame_container &inputFrameContainer,
            frame_container &outputFrameContainer, std::function<frame_sp(size_t)> _makeFrame);

private:
  std::string modelName;
  boost::shared_ptr<FrameContainerQueue> mQue;
  boost::shared_ptr<EncoderModelAbstractProps> mProps;
};