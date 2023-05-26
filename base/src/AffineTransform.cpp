#include <npp.h>
#include <opencv2/core.hpp> 
#include <CuCtxSynchronize.h>
#include  <nppdefs.h> 
#include "AffineTransform.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "AIPExceptions.h"
#include "ImageMetadata.h"
#include "RawImagePlanarMetadata.h"

#if defined(__arm__) || defined(__aarch64__)
#include "DMAFDWrapper.h"
#include "DMAAllocator.h"
#endif

#define PI 3.14159265

class Detail
{
public:
	Detail(AffineTransformProps &_props) : props(_props), shiftX(0), shiftY(0), mFrameType(FrameMetadata::GENERAL), mOutputFrameLength(0) {}
	int setInterPolation(AffineTransformProps::Interpolation eInterpolation)
	{
		switch (props.eInterpolation)
		{
		case AffineTransformProps::NN:
			return NppiInterpolationMode::NPPI_INTER_NN;
		case AffineTransformProps::LINEAR:
			return NppiInterpolationMode::NPPI_INTER_LINEAR;
		case AffineTransformProps::CUBIC:
			return NppiInterpolationMode::NPPI_INTER_CUBIC;
		case AffineTransformProps::UNDEFINED:
			return NppiInterpolationMode::NPPI_INTER_UNDEFINED; // not supported 
		case AffineTransformProps::CUBIC2P_BSPLINE:
			return NppiInterpolationMode::NPPI_INTER_CUBIC2P_BSPLINE;// not supported 
		case AffineTransformProps::CUBIC2P_CATMULLROM:
			return NppiInterpolationMode::NPPI_INTER_CUBIC2P_CATMULLROM;
		case AffineTransformProps::CUBIC2P_B05C03:
			return NppiInterpolationMode::NPPI_INTER_CUBIC2P_B05C03; // not supported 
		case AffineTransformProps::SUPER:
			return NppiInterpolationMode::NPPI_INTER_SUPER;  // not supported
		case AffineTransformProps::LANCZOS:
			return NppiInterpolationMode::NPPI_INTER_LANCZOS; // not supported 
		case AffineTransformProps::LANCZOS3_ADVANCED:
			return NppiInterpolationMode::NPPI_INTER_LANCZOS3_ADVANCED; // not supported
		default:
			throw new AIPException(AIP_NOTEXEPCTED, "Unknown value for Interpolation!");
		}
	}

	~Detail()
	{
	}

	virtual bool setPtrs() = 0;

	void initMatImages(framemetadata_sp& input)
	{
		iImg = Utils::getMatHeader(FrameMetadataFactory::downcast<RawImageMetadata>(input));
		oImg = Utils::getMatHeader(FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata));
	}
	void setMetadata(framemetadata_sp &metadata)
	{
		ImageMetadata::ImageType imageType;
		FrameMetadata::MemType memType = metadata->getMemType();
		if (mFrameType != metadata->getFrameType()) 
		{
			mFrameType = metadata->getFrameType();
			switch (mFrameType)
		    {
			case FrameMetadata::RAW_IMAGE:
			mOutputMetadata = framemetadata_sp(new RawImageMetadata(memType));
			break;
		    case FrameMetadata::RAW_IMAGE_PLANAR:
			mOutputMetadata = framemetadata_sp(new RawImagePlanarMetadata(memType));
			break;
			default:
			throw AIPException(AIP_FATAL, "Unsupported frameType<" + std::to_string(mFrameType) + ">");
			}
		}
		if (!metadata->isSet())
		{
			return;
		}

		if (mFrameType == FrameMetadata::RAW_IMAGE)
		{
			auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
			int w, h;
			h = rawMetadata->getHeight();
			w = rawMetadata->getWidth();
			RawImageMetadata outputMetadata(w*props.scale, h*props.scale, rawMetadata->getImageType(), rawMetadata->getType(), rawMetadata->getStep(), rawMetadata->getDepth(), memType, true);
			auto rawOutMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata);
			rawOutMetadata->setData(outputMetadata);
			imageType = rawMetadata->getImageType();
			depth = rawMetadata->getDepth();
		}

		if (mFrameType == FrameMetadata::RAW_IMAGE_PLANAR)
		{
			auto rawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
			int w, h;
			w = rawMetadata->getWidth(0);
			h = rawMetadata->getHeight(0);
			RawImagePlanarMetadata outputMetadata(w * props.scale, h * props.scale, rawMetadata->getImageType(), rawMetadata->getStep(0), rawMetadata->getDepth(), memType);
			auto rawOutMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(mOutputMetadata);
			rawOutMetadata->setData(outputMetadata);
			imageType = rawMetadata->getImageType();
			depth = rawMetadata->getDepth();
		}

		switch (imageType)
		{
		case ImageMetadata::MONO:
		case ImageMetadata::BGR:
		case ImageMetadata::RGB:
		case ImageMetadata::RGBA:
		case ImageMetadata::BGRA:
		case ImageMetadata::YUV444:
			if (depth != CV_8U)
			{
				throw AIPException(AIP_NOTIMPLEMENTED, "Rotate not supported for bit depth<" + std::to_string(depth) + ">");
			}
			break;
		}

		mOutputFrameLength = mOutputMetadata->getDataSize();
		setMetadataHelper(metadata, mOutputMetadata);

		int inWidth = srcSize[0].width;
		int inHeight = srcSize[0].height;
		int outWidth = dstSize[0].width;
		int outHeight = dstSize[0].height;

		double inCenterX = inWidth / 2.0;
		double inCenterY = inHeight / 2.0;

		double outCenterX = outWidth / 2.0;
		double outCenterY = outHeight / 2.0;

		double tx = (outCenterX - inCenterX); // translation factor which is used to shift image to center in output image
		double ty = (outCenterY - inCenterY);

		double si, co;
		si = props.scale * sin(props.angle * PI / 180);
		co = props.scale * cos(props.angle * PI / 180);

		double cx = props.x + (srcSize[0].width / 2); // rotating the image through its center
		double cy = props.y + (srcSize[0].height / 2);

		acoeff[0][0] = co;
		acoeff[0][1] = -si;
		acoeff[0][2] = ((1 - co) * cx + si * cy) + tx; //after rotation we translate it to center of output frame
		acoeff[1][0] = si;
		acoeff[1][1] = co;
		acoeff[1][2] = (-si * cx + (1 - co) * cy) + ty;

	}

	
	bool compute(void* buffer, void *outBuffer, framemetadata_sp&mInput)
	{
		FrameMetadata::MemType memType = mInput->getMemType();

		if(memType == FrameMetadata::MemType::HOST)
		{
			cv::Point2f srcTri[3];
			srcTri[0] = cv::Point2f(0.f, 0.f);
			srcTri[1] = cv::Point2f(iImg.cols - 1.f, 0.f);
			srcTri[2] = cv::Point2f(0.f, iImg.rows - 1.f);

			cv::Point2f dstTri[3];
			dstTri[0] = cv::Point2f(0.f, iImg.rows * 0.33f);
			dstTri[1] = cv::Point2f(iImg.cols * 0.85f, iImg.rows * 0.25f);
			dstTri[2] = cv::Point2f(iImg.cols * 0.15f, iImg.rows * 0.7f);

			cv::Mat warp_mat = cv::getAffineTransform(srcTri, dstTri);
			cv::Mat warp_dst = cv::Mat::zeros(iImg.rows, iImg.cols, iImg.type());
			cv::warpAffine(iImg, warp_dst, warp_mat, warp_dst.size());

			double cx = props.x + (warp_dst.cols / 2.0);
			double cy = props.y + (warp_dst.rows / 2.0);
			cv::Point2f center(cx, cy); // Center of rotation
			
			cv::Mat rot_mat = cv::getRotationMatrix2D(center, props.angle, props.mscale);
			cv::warpAffine(warp_dst, oImg, rot_mat, warp_dst.size());
		}

		if (memType == FrameMetadata::MemType::CUDA_DEVICE || memType == FrameMetadata::MemType::DMABUF)
		{
			auto status = NPP_SUCCESS;
			auto bufferNPP = static_cast<Npp8u*>(buffer);
			auto outBufferNPP = static_cast<Npp8u*>(outBuffer);

			if (mFrameType == FrameMetadata::RAW_IMAGE_PLANAR) // currently only YUV444 is supported
			{
				for (auto i = 0; i < channels; i++)
				{
					src[i] = bufferNPP + srcNextPtrOffset[i];
					dst[i] = outBufferNPP + dstNextPtrOffset[i];

					status = nppiWarpAffine_8u_C1R_Ctx(src[i],
						srcSize[i],
						srcPitch[i],
						srcRect[i],
						dst[i],
						dstPitch[i],
						dstRect[i],
						acoeff,
						setInterPolation(props.eInterpolation),
						nppStreamCtx);
				}
			}

			if (mFrameType == FrameMetadata::RAW_IMAGE)
			{

				if (channels == 1 && depth == CV_8UC1)
				{
					status = nppiWarpAffine_8u_C1R_Ctx(const_cast<const Npp8u*>(bufferNPP),
						srcSize[0],
						srcPitch[0],
						srcRect[0],
						outBufferNPP,
						dstPitch[0],
						dstRect[0],
						acoeff,
						setInterPolation(props.eInterpolation),
						nppStreamCtx);
				}
				else if (channels == 3)
				{
					status = nppiWarpAffine_8u_C3R_Ctx(const_cast<const Npp8u*>(bufferNPP),
						srcSize[0],
						srcPitch[0],
						srcRect[0],
						outBufferNPP,
						dstPitch[0],
						dstRect[0],
						acoeff,
						setInterPolation(props.eInterpolation),
						nppStreamCtx);
				}
				else if (channels == 4)
				{
					status = nppiWarpAffine_8u_C4R_Ctx(const_cast<const Npp8u*>(bufferNPP),
						srcSize[0],
						srcPitch[0],
						srcRect[0],
						outBufferNPP,
						dstPitch[0],
						dstRect[0],
						acoeff,
						setInterPolation(props.eInterpolation),
						nppStreamCtx);
				}
			}

			if (status != NPP_SUCCESS)
			{
				LOG_ERROR << "Affine Transform failed<" << status << ">";
				throw AIPException(AIP_FATAL, "Failed to tranform the image");
			}
		}
		return true;
	}

	void setProps(AffineTransformProps &mprops)
	{
		if (!mOutputMetadata.get())
		{
			return;
		}
		auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata);
		props = mprops;
	}

public:
	size_t mOutputFrameLength;
	frame_sp InputFrame;
	frame_sp OutputFrame;
	cv::Mat iImg;
	cv::Mat oImg;
	int rotateFlag;
	void* OutputPtr;
	void* InputPtr;
	std::string mOutputPinId;
	framemetadata_sp mOutputMetadata;
	AffineTransformProps props;

	bool setMetadataHelper(framemetadata_sp &input, framemetadata_sp &output)
	{
		if (mFrameType == FrameMetadata::RAW_IMAGE)
		{
			auto inputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(input);
			auto outputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(output);

			channels = inputRawMetadata->getChannels();
			srcSize[0] = {inputRawMetadata->getWidth(), inputRawMetadata->getHeight()};
			srcRect[0] = {0, 0, inputRawMetadata->getWidth(), inputRawMetadata->getHeight()};
			srcPitch[0] = static_cast<int>(inputRawMetadata->getStep());
			srcNextPtrOffset[0] = 0;
			dstSize[0] = {outputRawMetadata->getWidth(), outputRawMetadata->getHeight()};
			dstRect[0] = {0, 0, outputRawMetadata->getWidth(), outputRawMetadata->getHeight()};
			dstPitch[0] = static_cast<int>(outputRawMetadata->getStep());
			dstNextPtrOffset[0] = 0;
		}
		else if (mFrameType == FrameMetadata::RAW_IMAGE_PLANAR)
		{
			auto inputRawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(input);
			auto outputRawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(output);

			channels = inputRawMetadata->getChannels();

			for (auto i = 0; i < channels; i++)
			{
				srcSize[i] = {inputRawMetadata->getWidth(i), inputRawMetadata->getHeight(i)};
				srcRect[i] = {0, 0, inputRawMetadata->getWidth(i), inputRawMetadata->getHeight(i)};
				srcPitch[i] = static_cast<int>(inputRawMetadata->getStep(i));
				srcNextPtrOffset[i] = inputRawMetadata->getNextPtrOffset(i);

				dstSize[i] = {outputRawMetadata->getWidth(i), outputRawMetadata->getHeight(i)};
				dstRect[i] = {0, 0, outputRawMetadata->getWidth(i), outputRawMetadata->getHeight(i)};
				dstPitch[i] = static_cast<int>(outputRawMetadata->getStep(i));
				dstNextPtrOffset[i] = outputRawMetadata->getNextPtrOffset(i);
			}
		}
		return true;
	}

	FrameMetadata::FrameType mFrameType;
	int depth;
	int channels;
	NppiSize srcSize[4];
	NppiRect srcRect[4];
	int srcPitch[4];
	size_t srcNextPtrOffset[4];
	NppiSize dstSize[4];
	NppiRect dstRect[4];
	int dstPitch[4];
	size_t dstNextPtrOffset[4];

	double shiftX;
	double shiftY;
	void *ctx;
	NppStreamContext nppStreamCtx;

	Npp8u *src[3];
	Npp8u *dst[3];
	double acoeff[2][3] = {{-1, -1, -1}, {-1, -1, -1}};
};

class DetailDMA : public Detail
{
public:
	DetailDMA(AffineTransformProps& _props) : Detail(_props)
	{
		nppStreamCtx.hStream = props.stream->getCudaStream();
	}
	bool setPtrs()
	{
        #if defined(__arm__) || defined(__aarch64__)
		InputPtr = static_cast<DMAFDWrapper*>(InputFrame->data());
		OutputPtr = static_cast<DMAFDWrapper*>(OutputFrame->data());
		cudaMemset(mDetail->OutputPtr, 0, (mDetail->OutputFrame)->size());
        #endif
		return true;
	}
};

class DeatilCUDA: public Detail
{
public:
	DeatilCUDA(AffineTransformProps& _props) : Detail(_props)
	{
		nppStreamCtx.hStream = props.stream->getCudaStream();
	}

	bool setPtrs()
	{
		InputPtr = InputFrame->data();
		OutputPtr = OutputFrame->data();
		cudaMemset(OutputPtr, 0, OutputFrame->size());
		return true;
	}
};

class DetailHost : public Detail
{
public:
	DetailHost(AffineTransformProps& _props) : Detail(_props) {}

	bool setPtrs()
	{
		iImg.data = static_cast<uint8_t*>(InputFrame->data());
		oImg.data = static_cast<uint8_t*>(OutputFrame->data());
		InputPtr = iImg.data;
		OutputPtr = oImg.data ;
		return true;
	}
};

AffineTransform::AffineTransform(AffineTransformProps props) : Module(TRANSFORM, "AffineTransform", props), mProp( props) {}

AffineTransform::~AffineTransform() {}

bool AffineTransform::validateInputPins()
{
	if (getNumberOfInputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstInputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::RAW_IMAGE && frameType != FrameMetadata::RAW_IMAGE_PLANAR)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be RAW_IMAGE or RAW_IMAGE_PLANAR. Actual<" << frameType << ">";
		return false;
	}

	FrameMetadata::MemType memType = metadata->getMemType();
	if (memType != FrameMetadata::MemType::CUDA_DEVICE && memType != FrameMetadata::MemType::DMABUF && memType != FrameMetadata::MemType::HOST)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is expected to be CUDA_DEVICE or DMABUF. Actual<" << memType << ">";
		return false;
	}

	return true;
}

bool AffineTransform::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstOutputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::RAW_IMAGE && frameType != FrameMetadata::RAW_IMAGE_PLANAR)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input frameType is expected to be RAW_IMAGE or RAW_IMAGE_PLANAR. Actual<" << frameType << ">";
		return false;
	}

	FrameMetadata::MemType memType = metadata->getMemType();
	if (memType != FrameMetadata::MemType::CUDA_DEVICE && memType != FrameMetadata::MemType::DMABUF && memType != FrameMetadata::MemType::HOST)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input memType is expected to be CUDA_DEVICE or DMABUF. Actual<" << memType << ">";
		return false;
	}

	return true;
}

void AffineTransform::addInputPin(framemetadata_sp &metadata, string &pinId)
{

	Module::addInputPin(metadata, pinId);
	FrameMetadata::MemType memType = metadata->getMemType();

	if (memType == FrameMetadata::MemType::CUDA_DEVICE)
	{
		mDetail.reset(new DeatilCUDA(mProp));
	}

	else if (memType == FrameMetadata::MemType::DMABUF)
	{
		mDetail.reset(new DetailDMA(mProp));
	}

	else if (memType == FrameMetadata::MemType::HOST)
	{
		mDetail.reset(new DetailHost(mProp));
	}
	else
	{
		throw std::runtime_error("Memory Type not supported");
	}
	mDetail->setMetadata(metadata);
	if(memType == FrameMetadata::MemType::HOST)
	{
		mDetail->initMatImages(metadata);
	}
	mDetail->mOutputPinId = addOutputPin(mDetail->mOutputMetadata);
	
}

bool AffineTransform::init()
{
	if (!Module::init())
	{
		return false;
	}
	return Module::init();
}

bool AffineTransform::term()
{
	mDetail.reset();
	return Module::term();
}

bool AffineTransform::process(frame_container &frames)
{
	mDetail->InputFrame = frames.cbegin()->second;
	auto metadata = (mDetail->InputFrame)->getMetadata();
	mDetail->OutputFrame = makeFrame(mDetail->mOutputFrameLength);
	mDetail->setPtrs();

	mDetail->compute(mDetail->InputPtr, mDetail->OutputPtr,metadata);
	frames.insert(make_pair(mDetail->mOutputPinId, mDetail->OutputFrame));
	send(frames);

	return true;
}

bool AffineTransform::processSOS(frame_sp &frame)
{
	auto metadata = frame->getMetadata();
	mDetail->setMetadata(metadata);
	return true;
}

bool AffineTransform::shouldTriggerSOS()
{
	return mDetail->mOutputFrameLength == 0;
}

bool AffineTransform::processEOS(string &pinId)
{
	mDetail->mOutputFrameLength = 0;
	return true;
}

void AffineTransform::setProps(AffineTransformProps &props)
{
	Module::addPropsToQueue(props);
}

AffineTransformProps AffineTransform::getProps()
{
	fillProps(mDetail->props);
	return mDetail->props;
}

bool AffineTransform::handlePropsChange(frame_sp &frame)
{
	AffineTransformProps props(mDetail->props.stream, 0);
	bool ret = Module::handlePropsChange(frame, props);
	mDetail->setProps(props);
	return ret;
}