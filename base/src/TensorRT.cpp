#include <fstream>
#include <NvInfer.h>
#include <memory>
#include <vector>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <numeric>

#include "FrameMetadata.h"
#include "RawImageMetadata.h"
#include "TensorRT.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "AIPExceptions.h"

class TensorRT::Detail
{
public:
	size_t getSizeByDim(const nvinfer1::Dims& dims)
	{
		size_t size = 1;
		for (size_t i = 0; i < dims.nbDims; ++i)
		{
			size *= dims.d[i];
		}
		return size;
	}
	Detail(TensorRTProps &_props): props(_props)
	{
        // initialize TensorRT engine
        std::vector<char> trtModelStream_;
        size_t size{ 0 };

        std::ifstream file(_props.enginePath, std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream_.resize(size);
            file.read(trtModelStream_.data(), size);
            file.close();
        }
        nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
        assert(runtime != nullptr);
        nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream_.data(), size, nullptr);
        context = engine->createExecutionContext();
        batch_size = 1;
	}

	~Detail()
	{
	}

	bool compute()
	{
        context->enqueue(batch_size, buffers.data(), props.stream, nullptr);
		return true;
	}

	bool term(){
		context->destroy();
	}
    std::vector<void*> buffers;
private:
    class gLogger : public nvinfer1::ILogger
    {
    public:
        void log(Severity severity, const char* msg) override {
            if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) {
                LOG_ERROR << msg;
            }
        }
    } gLogger;

    int batch_size;
    nvinfer1::IExecutionContext* context;
	TensorRTProps props;
};

TensorRT::TensorRT(TensorRTProps _props) : Module(TRANSFORM, "TensorRT", _props)
{
	mDetail.reset(new Detail(_props));	
}

bool TensorRT::validateInputPins()
{
	if (getNumberOfInputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstInputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::RAW_IMAGE)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be RAW_IMAGE. Actual<" << frameType << ">";
		return false;
	}

	FrameMetadata::MemType memType = metadata->getMemType();
	if (memType != FrameMetadata::MemType::CUDA_DEVICE)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is expected to be CUDA_DEVICE. Actual<" << memType << ">";
		return false;
	}
 
	return true;
}

bool TensorRT::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstOutputMetadata();
	auto mOutputFrameType = metadata->getFrameType();
	if (mOutputFrameType != FrameMetadata::RAW_IMAGE)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input frameType is expected to be RAW_IMAGE. Actual<" << mOutputFrameType << ">";
		return false;
	}

	FrameMetadata::MemType memType = metadata->getMemType();
	if (memType != FrameMetadata::MemType::CUDA_DEVICE)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input memType is expected to be CUDA_DEVICE. Actual<" << memType << ">";
		return false;
	}	

	return true;
}

void TensorRT::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);
	auto inputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
    mOutputMetadata = framemetadata_sp(new RawImageMetadata(FrameMetadata::CUDA_DEVICE));
	mOutputMetadata->copyHint(*metadata.get());
	mOutputPinId = addOutputPin(mOutputMetadata);	
}

bool TensorRT::init()
{
	if (!Module::init())
	{
		return false;
	}

	return true;
}

bool TensorRT::term()
{
	return Module::term() && mDetail->term();
}

bool TensorRT::process(frame_container &frames)
{
	auto frame = frames.cbegin()->second;	
    auto outFrame = makeFrame();
    auto tempFrame = makeFrame();
	if(!frame.get() || !frame->size())
	{
		LOG_ERROR << "frame EMTPY";
	}
	if(!outFrame.get() || !outFrame->size())
	{
		LOG_ERROR << "outFrame EMTPY";
	}
	if(!tempFrame.get() || !tempFrame->size())
	{
		LOG_ERROR << "TEMPFRAME EMTPY";
	}
    mDetail->buffers.push_back(frame->data());
    mDetail->buffers.push_back(outFrame->data());
    mDetail->buffers.push_back(tempFrame->data());

    mDetail->compute();

	frames.insert(make_pair(mOutputPinId, outFrame));
	send(frames);

	mDetail->buffers.clear();

	return true;
}

bool TensorRT::processSOS(frame_sp &frame)
{
	auto metadata = frame->getMetadata();
	setMetadata(metadata);

	return true;
}

void TensorRT::setMetadata(framemetadata_sp& metadata)
{
	auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
	auto width = rawMetadata->getWidth();
	auto height = rawMetadata->getHeight();
	auto type = rawMetadata->getType();
	auto depth = rawMetadata->getDepth();
	auto inputImageType = rawMetadata->getImageType();

	if (!(inputImageType == ImageMetadata::RGB))
	{
		throw AIPException(AIP_NOTIMPLEMENTED, "Expected RGB Image");
	}
    auto rawOutMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata);
    RawImageMetadata outputMetadata(width, height, ImageMetadata::MONO, CV_32FC1, 512, CV_32F, FrameMetadata::CUDA_DEVICE, true);		
    rawOutMetadata->setData(outputMetadata);
}