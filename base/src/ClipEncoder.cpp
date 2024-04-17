#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"

#include "llama/common.h"
#include "llama/llama.h"
#include "llama/clip.h"
#include "llama/llava.h"
#include "ClipEncoder.h"

class ClipEncoder::Detail {
public:
  Detail(ClipEncoderProps &_props) : mProps(_props) {
    
  }

  ~Detail() { }

  void setProps(ClipEncoderProps &_props) {
    mProps = _props;
  }

public:
  ClipEncoderProps mProps;
  clip_ctx * mClipContext = NULL;
  llava_image_embed * storedData;
};

ClipEncoder::ClipEncoder(ClipEncoderProps _props) : EncoderModelAbstract("ClipEncoder", _props) {
  mDetail.reset(new Detail(_props));
}

ClipEncoder::~ClipEncoder() {}

bool ClipEncoder::validateUseCase(ClipEncoderProps::UseCase useCase) {
  for(auto validUseCase: mDetail->mProps.useCases) {
    if(validUseCase == useCase) {
      return true;
    }
  }
  return false;
}

bool ClipEncoder::modelInit() {
  mDetail->mClipContext = clip_model_load(mDetail->mProps.modelPath.c_str(), 1);
  return EncoderModelAbstract::init();
}

bool ClipEncoder::modelTerm() {
  if(mDetail->mClipContext){
    clip_free(mDetail->mClipContext);
    mDetail->mClipContext = NULL;
  }
  return EncoderModelAbstract::term();
}

bool ClipEncoder::modelInference(frame_container& frames) {
  auto frame = frames.begin()->second;
  auto imageBytes = static_cast<unsigned char*>(frame->data());
  auto imageBytesLength = frame->size();
  mDetail->storedData = llava_image_embed_make_with_bytes(mDetail->mClipContext, 8, imageBytes, imageBytesLength);
  return true;
}

size_t ClipEncoder::getFrameSize() {
  return (clip_embd_nbytes(mDetail->mClipContext) + sizeof(int));
}

void ClipEncoder::getFrames(frame_sp& frame) {
  memcpy(frame->data(), mDetail->storedData, sizeof(llava_image_embed));
  float *char_buffer = (float *)frame->data();
  char_buffer += sizeof(llava_image_embed);
  memcpy(char_buffer, mDetail->storedData->embed, clip_embd_nbytes(mDetail->mClipContext));
}