#include "ModelStrategy.h"

ModelStrategy::ModelStrategy() {

}

ModelStrategy::~ModelStrategy() {
  
}

boost::shared_ptr<ModelStrategy> ModelStrategy::create(ModelStrategyType type) {
  switch (type) {
    case ModelStrategyType::LLAVA_SCENE_DESCRIPTOR:
        return boost::make_shared<SceneDescriptorModelStrategy>();
    case ModelStrategyType::LLAVA_TEXT_TO_TEXT:
        return boost::make_shared<LlavaTextToTextModelStrategy>();
    default:
        return boost::make_shared<LlavaTextToTextModelStrategy>();
        break;
  }
}

/*LLAVA SCENE-DESCRIPTOR STRATEGY*/
SceneDescriptorModelStrategy::SceneDescriptorModelStrategy() : ModelStrategy() {
  auto clipProps = ClipEncoderProps("./data/llm/llava/llava-v1.6-7b/mmproj-model-f16.gguf");
  auto llavaProps = LlavaProps("./data/llm/llava/llava-v1.6-7b/llava-v1.6-mistral-7b.Q8_0.gguf", "Describe the image", 2048, 512, 0.8, 10, 256);
  
  encoderModel = boost::shared_ptr<EncoderModelAbstract>(new ClipEncoder(clipProps));
  llmModel = boost::shared_ptr<LlmModelAbstract>(new Llava(llavaProps));
}

SceneDescriptorModelStrategy::~SceneDescriptorModelStrategy() {

}

bool SceneDescriptorModelStrategy::initStrategy() {
  encoderModel->modelInit();
  llmModel->modelInit();
  return true;
}

bool SceneDescriptorModelStrategy::termStrategy() {
  encoderModel->modelTerm();
  llmModel->modelTerm();
  return true;
}

/*LLAVE TEXT-TO-TEXT STRATEGY*/
LlavaTextToTextModelStrategy::LlavaTextToTextModelStrategy() : ModelStrategy() {
  auto llavaProps = LlavaProps("./data/llm/llava/llava-v1.6-7b/llava-v1.6-mistral-7b.Q8_0.gguf", "Tell me a story", 2048, 512, 0.8, 10, 256);
  llmModel = boost::shared_ptr<LlmModelAbstract>(new Llava(llavaProps));
}

LlavaTextToTextModelStrategy::~LlavaTextToTextModelStrategy() {

}

bool LlavaTextToTextModelStrategy::initStrategy() {
  llmModel->modelInit();
  return true;
}

bool LlavaTextToTextModelStrategy::termStrategy() {
  llmModel->modelTerm();
  return true;
}