#include "Llava.h"
#include "Frame.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Logger.h"
#include "Utils.h"

#include <llama/common.h>
#include <llama/llama.h>
#include <llama/llava.h>

LlavaProps::LlavaProps(std::string _modelPath, std::string _systemPrompt, std::string _userPrompt, int _contextSize,
                       int _batchSize, float _degreeOfRandomness, int _gpuLayers,
                       int _predictionLength)
{

  /* Set LLM Model Base Class Properties for each model*/
  modelArchitecture = ModelArchitectureType::TRANSFORMER;
  inputTypes = {FrameMetadata::FrameType::TEXT,
                FrameMetadata::FrameType::IMAGE_EMBEDDING};
  outputTypes = {FrameMetadata::FrameType::TEXT};
  useCases = {UseCase::TEXT_TO_TEXT, UseCase::OCR, UseCase::SCENE_DESCRIPTOR};

  /*Unique Model Properties*/
  modelPath = _modelPath;
  systemPrompt = _systemPrompt;
  userPrompt = _userPrompt;
  degreeOfRandomness = _degreeOfRandomness;
  contextSize = _contextSize;
  batchSize = _batchSize;
  gpuLayers = _gpuLayers;
  predictionLength = _predictionLength;
}

class Llava::Detail
{
public:
  Detail(LlavaProps &_props) : mProps(_props)
  {
    mLlavaContext = NULL;
    systemPromptFlag = true;
    nPast = 0;
  }

  ~Detail() {}

  void setProps(LlavaProps &_props)
  {
    mProps = _props;
    setModelProps(_props);
  }

  void setModelProps(LlavaProps &_props)
  {
    mLlavaModelParams = llama_model_default_params();
    mLlavaContextParams = llama_context_default_params();
    setContextSize(_props);
    setBatchSize(_props);
    setDegreeOfRandomness(_props);
    setGpuLayers(_props);
  }

  void setContextSize(LlavaProps &_props)
  {
    mLlavaContextParams.n_ctx = _props.contextSize;
  }

  void setBatchSize(LlavaProps &_props)
  {
    mLlavaContextParams.n_batch = _props.batchSize;
  }

  void setDegreeOfRandomness(LlavaProps &_props)
  {
    mLlavaSamplingParams.temp = _props.degreeOfRandomness;
  }

  void setGpuLayers(LlavaProps &_props)
  {
    mLlavaModelParams.n_gpu_layers = _props.gpuLayers;
  }

  void compute(llama_context *llamaContext, std::vector<llama_token> tokens,
               int nBatch, int *nPast)
  {
    int N = (int)tokens.size();
    for (int i = 0; i < N; i += nBatch)
    {
      int nEval = (int)tokens.size() - i;
      if (nEval > nBatch)
      {
        nEval = nBatch;
      }
      if (llama_decode(llamaContext,
                       llama_batch_get_one(&tokens[i], nEval, *nPast, 0)))
      {
        LOG_ERROR << "LLAMA DECODE ERROR";
        break;
      }
      *nPast += nEval;
    }
  }

public:
  LlavaProps mProps;
  llama_model *mLlavaModel;
  llama_context *mLlavaContext;
  llama_model_params mLlavaModelParams;
  llama_context_params mLlavaContextParams;
  llama_sampling_params mLlavaSamplingParams;
  std::string storedData;
  bool systemPromptFlag;
  int nPast;
};

Llava::Llava(LlavaProps _props) : LlmModelAbstract("Llava", _props)
{
  mDetail.reset(new Detail(_props));
}

Llava::~Llava() {}

bool Llava::validateUseCase(UseCase useCase)
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

bool Llava::modelInit()
{
  llama_backend_init(false /*NUMA Architecure set to false*/);
  mDetail->setModelProps(mDetail->mProps);
  mDetail->mLlavaModel = llama_load_model_from_file(
      mDetail->mProps.modelPath.c_str(), mDetail->mLlavaModelParams);
  mDetail->mLlavaContext = llama_new_context_with_model(
      mDetail->mLlavaModel, mDetail->mLlavaContextParams);

  if (!mDetail->mLlavaContext)
  {
    LOG_ERROR << "Cannot Load Llava Model";
    return false;
  }
  return LlmModelAbstract::init();
}

bool Llava::modelTerm()
{
  llama_free(mDetail->mLlavaContext);
  llama_free_model(mDetail->mLlavaModel);
  llama_backend_free();
  return LlmModelAbstract::term();
}

bool Llava::modelInference(frame_container &inputFrameContainer, frame_container &outputFrameContainer, std::function<frame_sp(size_t)> makeFrame)
{
  /*Parameter Declaration*/
  auto outputPinId = inputFrameContainer.begin()->first;
  auto inputFrame = inputFrameContainer.begin()->second;
  auto frameType = inputFrame->getMetadata()->getFrameType();

  std::string systemPrompt = mDetail->mProps.systemPrompt;
  std::string userPrompt = mDetail->mProps.userPrompt;
  const bool add_bos =
      llama_should_add_bos_token(llama_get_model(mDetail->mLlavaContext));
  int nPredict = mDetail->mProps.predictionLength;
  int nBatch = mDetail->mProps.batchSize;

  /*System Prompt Tokenization*/
  if(mDetail->systemPromptFlag){
    std::vector<llama_token> systemPromptTokens =
      ::llama_tokenize(mDetail->mLlavaContext, systemPrompt, add_bos);
    mDetail->compute(mDetail->mLlavaContext, systemPromptTokens, nBatch, &mDetail->nPast);
    mDetail->systemPromptFlag = false;
    LOG_ERROR << "Loaded System Prompt";
  }
  
  if (frameType == FrameMetadata::FrameType::IMAGE_EMBEDDING)
  {
    /*Image Embed Tokenization*/
    auto imageEmbedding = static_cast<llava_image_embed *>(inputFrame->data());
    llava_eval_image_embed(mDetail->mLlavaContext, imageEmbedding, nBatch, &mDetail->nPast);
  }
  else if (frameType == FrameMetadata::FrameType::TEXT)
  {
    /*Text Embed Tokenization*/
    auto textEmbed = static_cast<char *>(inputFrame->data());
    std::string textEmbedPrompt(textEmbed);
    std::vector<llama_token> textEmbedTokens =
        ::llama_tokenize(mDetail->mLlavaContext, textEmbedPrompt, false);
    mDetail->compute(mDetail->mLlavaContext, textEmbedTokens, nBatch, &mDetail->nPast);
  }

  /*User Prompt Tokenization*/
  std::vector<llama_token> userPromptTokens = ::llama_tokenize(
      mDetail->mLlavaContext, (userPrompt + "\nASSISTANT:").c_str(), false);
  mDetail->compute(mDetail->mLlavaContext, userPromptTokens, nBatch, &mDetail->nPast);

  std::string output = "";

  std::cout << "\n";

  /*Prediction token by token*/
  for (int i = 0; i < nPredict; i++)
  {
    llama_token id = 0;
    auto logits = llama_get_logits(mDetail->mLlavaContext);
    auto nVocab = llama_n_vocab(llama_get_model(mDetail->mLlavaContext));

    std::vector<llama_token_data> candidates;
    candidates.reserve(nVocab);
    for (llama_token tokenId = 0; tokenId < nVocab; tokenId++)
    {
      candidates.emplace_back(
          llama_token_data{tokenId, logits[tokenId], 0.0f});
    }

    llama_token_data_array candidatesP = {candidates.data(), candidates.size(),
                                           false};
    id = llama_sample_token_greedy(mDetail->mLlavaContext, &candidatesP);

    if (id == llama_token_eos(llama_get_model(mDetail->mLlavaContext)))
    {
      break;
    }

    std::string ret = llama_token_to_piece(mDetail->mLlavaContext, id);
    output += ret;

    std::cout << ret << std::flush;

    std::vector<llama_token> tokens;
    tokens.push_back(id);
    mDetail->compute(mDetail->mLlavaContext, tokens, 1, &mDetail->nPast);
  }

  mDetail->storedData = output;
  auto outputFrame = makeFrame(getFrameSize());
  auto metaData = boost::shared_ptr<FrameMetadata>(
      new FrameMetadata(FrameMetadata::FrameType::TEXT));
  outputFrame->setMetadata(metaData);
  storeFrames(outputFrame);
  outputFrameContainer.insert(make_pair(outputPinId, outputFrame));
  return true;
}

size_t Llava::getFrameSize()
{
  return (mDetail->storedData.length() + 1); /* Add 1 more byte for /0 for conversion from std::string to char* */
}

void Llava::storeFrames(frame_sp &frame)
{
  memcpy(frame->data(), mDetail->storedData.c_str(), frame->size());
}