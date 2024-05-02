#include "ModelStrategy.h"
#include "ClipEncoder.h"
#include "Llava.h"

ModelStrategy::ModelStrategy() {}

ModelStrategy::~ModelStrategy() {}

/*LLAVA SCENE-DESCRIPTOR STRATEGY*/
SceneDescriptorModelStrategy::SceneDescriptorModelStrategy(
    SceneDescriptorXFormProps props)
    : ModelStrategy()
{
  auto clipProps = ClipEncoderProps(props.encoderModelPath);
  auto llavaProps =
      LlavaProps(props.llmModelPath, props.systemPrompt, props.userPrompt, 4096,
                 512, 0.8, props.gpuLayers, 256);

  encoderModel =
      boost::shared_ptr<EncoderModelAbstract>(new ClipEncoder(clipProps));
  llmModel = boost::shared_ptr<LlmModelAbstract>(new Llava(llavaProps));
}

SceneDescriptorModelStrategy::~SceneDescriptorModelStrategy() {}

bool SceneDescriptorModelStrategy::initStrategy()
{
  encoderModel->modelInit();
  llmModel->modelInit();
  return true;
}

bool SceneDescriptorModelStrategy::termStrategy()
{
  encoderModel->modelTerm();
  llmModel->modelTerm();
  return true;
}

/*LLAVE TEXT-TO-TEXT STRATEGY*/
LlavaTextToTextModelStrategy::LlavaTextToTextModelStrategy() : ModelStrategy()
{
  auto llavaProps = LlavaProps(
      "./data/llm/llava/llava-v1.6-7b/llava-v1.6-mistral-7b.Q8_0.gguf",
      "A chat between a curious human and an artificial intelligence "
      "assistant.  The assistant gives helpful, detailed, and polite answers "
      "to the human's questions.\nUSER:",
      "Tell me a story", 2048, 512, 0.8, 10, 256);
  ;
  llmModel = boost::shared_ptr<LlmModelAbstract>(new Llava(llavaProps));
}

LlavaTextToTextModelStrategy::~LlavaTextToTextModelStrategy() {}

bool LlavaTextToTextModelStrategy::initStrategy()
{
  llmModel->modelInit();
  return true;
}

bool LlavaTextToTextModelStrategy::termStrategy()
{
  llmModel->modelTerm();
  return true;
}