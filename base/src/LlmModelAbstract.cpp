#include "LlmModelAbstract.h"
#include "FrameContainerQueue.h"

LlmModelAbstract::LlmModelAbstract(std::string name, LlmModelAbstractProps _props) : myName(name) {
  mQue.reset(new FrameContainerQueue(_props.qlen));
	mProps.reset(new LlmModelAbstractProps(_props));
}

LlmModelAbstract::~LlmModelAbstract() {}

bool LlmModelAbstract::init() {
  mQue->accept();
  return true;
}

bool LlmModelAbstract::term() {
  mQue->clear();
  return true;
}

bool LlmModelAbstract::step() {
  auto frames = mQue->pop();
  if (frames.size() == 0) {
    return true;
  }
  bool ret = modelInference(frames);
  return ret;
}

bool LlmModelAbstract::push(frame_container& frameContainer) {
  mQue->push(frameContainer);
  return true;
}