#include "ModelStrategy.h"
#include "ClipEncoder.h"
#include "Llava.h"

ModelStrategy::ModelStrategy() {}

ModelStrategy::~ModelStrategy() {}

/*LLAVA SCENE-DESCRIPTOR STRATEGY*/
ImageToTextModelStrategy::ImageToTextModelStrategy(
    ImageToTextXFormProps props)
    : ModelStrategy()
{
  auto clipProps = ClipEncoderProps(props.encoderModelPath);
  auto llavaProps =
      LlavaProps(props.llmModelPath, props.systemPrompt, props.userPrompt, 
                 ModelStrategy::DEFAULT_CONTEXT_SIZE, 
                 ModelStrategy::DEFAULT_BATCH_SIZE, 
                 ModelStrategy::DEFAULT_TEMPERATURE, 
                 props.gpuLayers, 
                 ModelStrategy::DEFAULT_MAX_TOKENS);

  encoderModel =
      boost::shared_ptr<EncoderModelAbstract>(new ClipEncoder(clipProps));
  llmModel = boost::shared_ptr<LlmModelAbstract>(new Llava(llavaProps));
}

ImageToTextModelStrategy::~ImageToTextModelStrategy() {}

bool ImageToTextModelStrategy::initStrategy()
{
  encoderModel->modelInit();
  llmModel->modelInit();
  return true;
}

bool ImageToTextModelStrategy::termStrategy()
{
  encoderModel->modelTerm();
  llmModel->modelTerm();
  return true;
}

/*LLAVE TEXT-TO-TEXT STRATEGY*/
LlavaTextToTextModelStrategy::LlavaTextToTextModelStrategy(ImageToTextXFormProps props) : ModelStrategy()
{
  auto llavaProps = LlavaProps(
      props.llmModelPath,
      props.systemPrompt,
      props.userPrompt,
      ModelStrategy::DEFAULT_CONTEXT_SIZE,
      ModelStrategy::DEFAULT_BATCH_SIZE,
      ModelStrategy::DEFAULT_TEMPERATURE,
      props.gpuLayers,
      ModelStrategy::DEFAULT_MAX_TOKENS
  );
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