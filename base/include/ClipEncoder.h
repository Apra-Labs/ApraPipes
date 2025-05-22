#pragma once

#include "EncoderModelAbstract.h"

class ClipEncoderProps : public EncoderModelAbstractProps
{
public:
  ClipEncoderProps(std::string _modelPath, int _numThreads = 8);

  std::string modelPath;
  int numThreads; // Removed default value since it's set in constructor

  size_t getSerializeSize()
  {
    return EncoderModelAbstractProps::getSerializeSize() + sizeof(modelPath) + sizeof(numThreads);
  }

private:
  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive &ar, const unsigned int version)
  {
    ar &boost::serialization::base_object<EncoderModelAbstractProps>(*this);
    ar & modelPath;
    ar & numThreads;
  }
};

class ClipEncoder : public EncoderModelAbstract
{
public:
  ClipEncoder(ClipEncoderProps _props);
  virtual ~ClipEncoder();
  bool modelInit() override;
  bool modelTerm() override;
  bool modelInference(frame_container &inputFrameContainer,
                      frame_container &outputFrameContainer, std::function<frame_sp(size_t)> makeFrame) override;
  bool validateUseCase(UseCase useCase) override;
  size_t getFrameSize() override;
  void storeFrames(frame_sp &frame);

private:
  class Detail;
  boost::shared_ptr<Detail> mDetail;
};