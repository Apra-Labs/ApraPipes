#include "EncoderModelAbstract.h"
#include "FrameContainerQueue.h"
#include "Logger.h"

EncoderModelAbstract::EncoderModelAbstract(std::string name, EncoderModelAbstractProps _props) : myName(name) {
	mQue.reset(new FrameContainerQueue(_props.qlen));
	mProps.reset(new EncoderModelAbstractProps(_props));
}

EncoderModelAbstract::~EncoderModelAbstract() { }

bool EncoderModelAbstract::init() {
  mQue->accept();
  return true;
}

bool EncoderModelAbstract::term() {
  mQue->clear();
  return true;
}

bool EncoderModelAbstract::step() {
  auto frames = mQue->pop();
  if (frames.size() == 0) {
    return true;
  }
  bool ret = modelInference(frames);
  return ret;
}

bool EncoderModelAbstract::push(frame_container& frameContainer) {
  mQue->push(frameContainer);
  return true;
}