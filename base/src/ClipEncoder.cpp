#include "ClipEncoder.h"
#include "Frame.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Logger.h"
#include "Utils.h"

#include <llama/common.h>
#include <llama/clip.h>
#include <llama/llava.h>

ClipEncoderProps::ClipEncoderProps(std::string _modelPath, int _numThreads)
{
  /* Set LLM Model Base Class Properties for each model*/
  modelArchitecture = ModelArchitectureType::VIT;
  inputTypes = {FrameMetadata::FrameType::TEXT,
                FrameMetadata::FrameType::ENCODED_IMAGE};
  outputTypes = {FrameMetadata::FrameType::IMAGE_EMBEDDING};
  useCases = {UseCase::OCR, UseCase::SCENE_DESCRIPTOR};

  /*Unique Model Properties*/
  modelPath = _modelPath;
  numThreads = _numThreads;
}

class ClipEncoder::Detail
{
public:
  Detail(ClipEncoderProps &_props) : mProps(_props) { mClipContext = NULL; }

  ~Detail() {}

  void setProps(ClipEncoderProps &_props) { mProps = _props; }

public:
  ClipEncoderProps mProps;
  clip_ctx *mClipContext;
  llava_image_embed *storedData = nullptr;
};

ClipEncoder::ClipEncoder(ClipEncoderProps _props)
    : EncoderModelAbstract("ClipEncoder", _props)
{
  mDetail.reset(new Detail(_props));
}

ClipEncoder::~ClipEncoder() {}

bool ClipEncoder::validateUseCase(UseCase useCase)
{
  for (auto validUseCase : mDetail->mProps.useCases)
  {
    if (validUseCase == useCase)
    {
      return true;
    }
  }
  throw AIPException(AIP_FATAL, "Model cannot be used for the this use case");
  return false;
}

bool ClipEncoder::modelInit()
{
  int verbosity = 1;
  mDetail->mClipContext =
      clip_model_load(mDetail->mProps.modelPath.c_str(), verbosity);
  if (!mDetail->mClipContext)
  {
    LOG_ERROR << "Cannot Load Clip Model";
    return false;
  }
  return EncoderModelAbstract::init();
}

bool ClipEncoder::modelTerm()
{
  if (mDetail->mClipContext)
  {
    clip_free(mDetail->mClipContext);
    mDetail->mClipContext = NULL;
  }
  return EncoderModelAbstract::term();
}

bool ClipEncoder::modelInference(frame_container &inputFrameContainer,
                                 frame_container &outputFrameContainer, std::function<frame_sp(size_t)> makeFrame)
{
   auto outputPinId = inputFrameContainer.begin()->first;
  auto inputFrame = inputFrameContainer.begin()->second;
  mDetail->storedData = llava_image_embed_make_with_bytes(
      mDetail->mClipContext, mDetail->mProps.numThreads,
      static_cast<unsigned char *>(inputFrame->data()), inputFrame->size());

  auto outputFrame = makeFrame(getFrameSize());
  auto metaData = boost::shared_ptr<FrameMetadata>(
      new FrameMetadata(FrameMetadata::FrameType::IMAGE_EMBEDDING));
  outputFrame->setMetadata(metaData);
  storeFrames(outputFrame);
  outputFrameContainer.insert(make_pair(outputPinId, outputFrame));
  mDetail->storedData = nullptr;
  return true;
}

size_t ClipEncoder::getFrameSize()
{
  return (clip_embd_nbytes(mDetail->mClipContext) + sizeof(int));
}

void ClipEncoder::storeFrames(frame_sp &frame)
{
  memcpy(frame->data(), mDetail->storedData, sizeof(llava_image_embed));
  float *char_buffer = (float *)frame->data();
  char_buffer += sizeof(llava_image_embed);
  memcpy(char_buffer, mDetail->storedData->embed,
         clip_embd_nbytes(mDetail->mClipContext));
}