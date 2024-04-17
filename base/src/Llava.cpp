#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"

#include "llama/common.h"
#include "llama/llama.h"
#include "llama/llava.h"
#include "Llava.h"

class Llava::Detail {
public:
  Detail(LlavaProps &_props) : mProps(_props) {
    setContextSize(_props);
    setBatchSize(_props);
    setDegreeOfRandomness(_props);
    setGpuLayers(_props);
  }

  ~Detail() {
    
  }

  void setProps(LlavaProps &_props) {
    mProps = _props;
    updateProps(_props);
  }

  void updateProps(LlavaProps &_props) {
    setContextSize(_props);
    setBatchSize(_props);
    setDegreeOfRandomness(_props);
    setGpuLayers(_props);
  }

  void setContextSize(LlavaProps &_props) {
    mLlavaContextParams.n_ctx = _props.contextSize;
  }

  void setBatchSize(LlavaProps &_props) {
    mLlavaContextParams.n_batch = _props.batchSize;
  }

  void setDegreeOfRandomness(LlavaProps &_props) {
    mLlavaSamplingParams.temp = _props.degreeOfRandomness;
  }

  void setGpuLayers(LlavaProps &_props) {
    mLlavaModelParams.n_gpu_layers = _props.gpuLayers;
  }

  void compute(llama_context * llamaContext, std::vector<llama_token> tokens, int nBatch, int * nPast) {
    int N = (int) tokens.size();
    for (int i = 0; i < N; i += nBatch) {
        int nEval = (int) tokens.size() - i;
        if (nEval > nBatch) {
            nEval = nBatch;
        }
        if (llama_decode(llamaContext, llama_batch_get_one(&tokens[i], nEval, *nPast, 0))) {
            LOG_ERROR << "LLAMA DECODE ERROR";
            break;
        }
        *nPast += nEval;
    }
  }

public:
  LlavaProps mProps;
  llama_model *mLlavaModel;
  llama_context *mLlavaContext = NULL;
  llama_model_params mLlavaModelParams;
  llama_context_params mLlavaContextParams;
  llama_sampling_params mLlavaSamplingParams;
  std::string storedData;
};

Llava::Llava(LlavaProps _props) : LlmModelAbstract("Llava", _props) {
  mDetail.reset(new Detail(_props));
}

Llava::~Llava() {}

bool Llava::validateUseCase(LlavaProps::UseCase useCase) {
  for(auto validUseCase: mDetail->mProps.useCases) {
    if(validUseCase == useCase) {
      return true;
    }
  }
  return false;
}

bool Llava::modelInit() {
  llama_backend_init(false /*NUMA Architecure set to false*/);
   
  mDetail->mLlavaModelParams = llama_model_default_params();
  mDetail->mLlavaContextParams = llama_context_default_params();
  mDetail->updateProps(mDetail->mProps);

  mDetail->mLlavaModel = llama_load_model_from_file(
      mDetail->mProps.modelPath.c_str(), mDetail->mLlavaModelParams);
  mDetail->mLlavaContext = llama_new_context_with_model(
      mDetail->mLlavaModel, mDetail->mLlavaContextParams);
  return LlmModelAbstract::init();
}

bool Llava::modelTerm() {
  llama_free(mDetail->mLlavaContext);
  llama_free_model(mDetail->mLlavaModel);
  llama_backend_free();
  return LlmModelAbstract::term();
}

bool Llava::modelInference(frame_container& frames) {
  /*Parameter Declaration*/
  auto frame = frames.begin()->second;
  auto frameType = frame->getMetadata()->getFrameType();
  int nPast = 0;
  std::string systemPrompt = "A chat between a curious human and an artificial intelligence assistant.  The assistant gives helpful, detailed, and polite answers to the human's questions.\nUSER:";
  std::string userPrompt = mDetail->mProps.prompt;
  const bool add_bos = llama_should_add_bos_token(llama_get_model(mDetail->mLlavaContext));
  int nPredict = mDetail->mProps.predictionLength;
  int nBatch = mDetail->mProps.batchSize;

  /*System Prompt Tokenization*/
  std::vector<llama_token> systemPromptTokens = ::llama_tokenize(mDetail->mLlavaContext, systemPrompt, add_bos);
  mDetail->compute(mDetail->mLlavaContext, systemPromptTokens, nBatch, &nPast);

  if(frameType == FrameMetadata::FrameType::IMAGE_EMBEDDING){
    /*Image Embed Tokenization*/
    auto imageEmbed = static_cast<llava_image_embed*>(frame->data());
    llava_eval_image_embed(mDetail->mLlavaContext, imageEmbed, nBatch, &nPast);
  }
  else if(frameType == FrameMetadata::FrameType::TEXT){
    /*Text Embed Tokenization*/
    auto textEmbed = static_cast<char*>(frame->data());
    std::string textEmbedPrompt(textEmbed);
    std::vector<llama_token> textEmbedTokens = ::llama_tokenize(mDetail->mLlavaContext, textEmbedPrompt, false);
    mDetail->compute(mDetail->mLlavaContext, textEmbedTokens, nBatch, &nPast);
  }
  
  /*User Prompt Tokenization*/
  std::vector<llama_token> userPromptTokens = ::llama_tokenize(mDetail->mLlavaContext, (userPrompt + "\nASSISTANT:").c_str(), false);
  mDetail->compute(mDetail->mLlavaContext, userPromptTokens, nBatch, &nPast);

  std::string output = "";
  
  std::cout << "\n";

  /*Prediction token by token*/
  for(int i = 0; i < nPredict; i++) {
    llama_token id = 0;
    auto logits  = llama_get_logits(mDetail->mLlavaContext);
    auto n_vocab = llama_n_vocab(llama_get_model(mDetail->mLlavaContext));

    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);
    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
    }

    llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
    id = llama_sample_token_greedy(mDetail->mLlavaContext, &candidates_p);

    if (id == llama_token_eos(llama_get_model(mDetail->mLlavaContext))) {
        break;
    }

    std::string ret = llama_token_to_piece(mDetail->mLlavaContext, id);
    output += ret;

    std::cout << ret;

    std::vector<llama_token> tokens;
    tokens.push_back(id);
    mDetail->compute(mDetail->mLlavaContext, tokens, 1, &nPast);
  }

  mDetail->storedData = output;
  return true;
}

size_t Llava::getFrameSize() {
  return (mDetail->storedData.length() + 1); /* Add 1 more byte for /0 for conversion from std::string to char* */
}

void Llava::getFrames(frame_sp& frame) {
  memcpy(frame->data(), mDetail->storedData.c_str(), frame->size());
}