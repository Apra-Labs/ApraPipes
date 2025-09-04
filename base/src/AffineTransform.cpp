#ifdef APRA_CUDA_ENABLED
#include <npp.h>
#include <CuCtxSynchronize.h>
#include  <nppdefs.h> 
#endif
#include <opencv2/core.hpp> 
#include "AffineTransform.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "AIPExceptions.h"
#include "ImageMetadata.h"
#include "RawImagePlanarMetadata.h"

#if 1
#include "DMAFDWrapper.h"
#include "DMAAllocator.h"
#endif

#define PI 3.14159265

class DetailMemoryAbstract
{
public:
	DetailMemoryAbstract(AffineTransformProps &_props) : props(_props), shiftX(0), shiftY(0), mFrameType(FrameMetadata::GENERAL), mOutputFrameLength(0) {}
	int setInterPolation(AffineTransformProps::Interpolation interpolation)
	{
#ifdef APRA_CUDA_ENABLED
		switch (props.interpolation)
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
#endif
	}

	~DetailMemoryAbstract() {}

	virtual bool setPtrs() = 0;
	virtual bool compute() = 0;
	virtual void mSetMetadata(framemetadata_sp& metadata) = 0;

	void setMetadata(framemetadata_sp& metadata)
	{
		mInputMetadata = metadata;
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
			height = rawMetadata->getHeight();
			width = rawMetadata->getWidth();
			RawImageMetadata outputMetadata(width * props.scale, height * props.scale, rawMetadata->getImageType(), rawMetadata->getType(), rawMetadata->getStep(), rawMetadata->getDepth(), memType, true);
			auto rawOutMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata);
			rawOutMetadata->setData(outputMetadata);
			imageType = rawMetadata->getImageType();
			depth = rawMetadata->getDepth();
		}

		else if (mFrameType == FrameMetadata::RAW_IMAGE_PLANAR)
		{
			auto rawPlanarMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
			width = rawPlanarMetadata->getWidth(0);
			height = rawPlanarMetadata->getHeight(0);
			RawImagePlanarMetadata outputMetadata(width * props.scale, height * props.scale, rawPlanarMetadata->getImageType(), rawPlanarMetadata->getStep(0), rawPlanarMetadata->getDepth(), memType);
			auto rawOutMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(mOutputMetadata);
			rawOutMetadata->setData(outputMetadata);
			imageType = rawPlanarMetadata->getImageType();
			depth = rawPlanarMetadata->getDepth();
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
	}

	void setProps(AffineTransformProps &mprops)
	{
		if (!mOutputMetadata.get())
		{
			return;
		}
		props = mprops;
		mSetMetadata(mInputMetadata);
	}

public:
	size_t mOutputFrameLength;
	frame_sp inputFrame;
	frame_sp outputFrame;
	std::string mOutputPinId;
	framemetadata_sp mOutputMetadata;
	AffineTransformProps props;
protected:
	framemetadata_sp mInputMetadata;
	FrameMetadata::FrameType mFrameType;
	int width, height;
	int depth;
	double shiftX;
	double shiftY;
};

class DetailGPU : public DetailMemoryAbstract
{
public:
	DetailGPU(AffineTransformProps& _props) : DetailMemoryAbstract(_props) {}
	void mSetMetadata(framemetadata_sp& metadata)
	{
	    setMetadata(metadata);
		if (!metadata->isSet())
		{
			return;
		}
		setMetadataHelper(metadata, mOutputMetadata);
#ifdef APRA_CUDA_ENABLED
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
#endif
	}

	bool setMetadataHelper(framemetadata_sp& input, framemetadata_sp& output)
	{
#ifdef APRA_CUDA_ENABLED
		if (mFrameType == FrameMetadata::RAW_IMAGE)
		{
			auto inputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(input);
			auto outputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(output);

			channels = inputRawMetadata->getChannels();
			srcSize[0] = { inputRawMetadata->getWidth(), inputRawMetadata->getHeight() };
			srcRect[0] = { 0, 0, inputRawMetadata->getWidth(), inputRawMetadata->getHeight() };
			srcPitch[0] = static_cast<int>(inputRawMetadata->getStep());
			srcNextPtrOffset[0] = 0;
			dstSize[0] = { outputRawMetadata->getWidth(), outputRawMetadata->getHeight() };
			dstRect[0] = { 0, 0, outputRawMetadata->getWidth(), outputRawMetadata->getHeight() };
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
				srcSize[i] = { inputRawMetadata->getWidth(i), inputRawMetadata->getHeight(i) };
				srcRect[i] = { 0, 0, inputRawMetadata->getWidth(i), inputRawMetadata->getHeight(i) };
				srcPitch[i] = static_cast<int>(inputRawMetadata->getStep(i));
				srcNextPtrOffset[i] = inputRawMetadata->getNextPtrOffset(i);

				dstSize[i] = { outputRawMetadata->getWidth(i), outputRawMetadata->getHeight(i) };
				dstRect[i] = { 0, 0, outputRawMetadata->getWidth(i), outputRawMetadata->getHeight(i) };
				dstPitch[i] = static_cast<int>(outputRawMetadata->getStep(i));
				dstNextPtrOffset[i] = outputRawMetadata->getNextPtrOffset(i);
			}
		}
#endif 
		return true;
	}

	bool compute()
	{
#ifdef APRA_CUDA_ENABLED
		auto status = NPP_SUCCESS;
		auto bufferNPP = static_cast<Npp8u*>(inputPtr);
		auto outBufferNPP = static_cast<Npp8u*>(outputPtr);

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
					setInterPolation(props.interpolation),
					nppStreamCtx);
			}
		}

		else if (mFrameType == FrameMetadata::RAW_IMAGE)
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
					setInterPolation(props.interpolation),
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
					setInterPolation(props.interpolation),
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
					setInterPolation(props.interpolation),
					nppStreamCtx);
			}
		}

		if (status != NPP_SUCCESS)
		{
			LOG_ERROR << "Affine Transform failed<" << status << ">";
			throw AIPException(AIP_FATAL, "Failed to tranform the image");
		}
#endif

		return true;
	}

protected:
#ifdef APRA_CUDA_ENABLED
	NppStreamContext nppStreamCtx;
	void* outputPtr;
	void* inputPtr;
	double acoeff[2][3] = { {-1, -1, -1}, {-1, -1, -1} };

	int channels;
	NppiSize srcSize[4];
	NppiRect srcRect[4];
	int srcPitch[4];
	size_t srcNextPtrOffset[4];
	NppiSize dstSize[4];
	NppiRect dstRect[4];
	int dstPitch[4];
	size_t dstNextPtrOffset[4];

	void* ctx;

	Npp8u* src[3];
	Npp8u* dst[3];
#endif 
};


class DetailDMA : public DetailGPU
{
public:
	DetailDMA(AffineTransformProps& _props) : DetailGPU(_props)
	{
#ifdef APRA_CUDA_ENABLED
		nppStreamCtx.hStream = props.stream->getCudaStream();
#endif
	}

	bool setPtrs()
	{
        #if 1
		inputPtr = (static_cast<DMAFDWrapper*>(inputFrame->data()))->getCudaPtr();
		outputPtr = (static_cast<DMAFDWrapper*>(outputFrame->data()))->getCudaPtr();
		cudaMemset(outputPtr, 0, outputFrame->size());
        #endif
		return true;
	}
};

class DeatilCUDA: public DetailGPU
{
public:
	DeatilCUDA(AffineTransformProps& _props) : DetailGPU(_props)
	{
#ifdef APRA_CUDA_ENABLED
		nppStreamCtx.hStream = props.stream->getCudaStream();
#endif
	}

	bool setPtrs()
	{
        #ifdef APRA_CUDA_ENABLED
		inputPtr = inputFrame->data();
		outputPtr = outputFrame->data();
		cudaMemset(outputPtr, 0, outputFrame->size());
        #endif
		return true;
	}
};

class DetailHost : public DetailMemoryAbstract
{
public:
	DetailHost(AffineTransformProps& _props) : DetailMemoryAbstract(_props) {}
	void mSetMetadata(framemetadata_sp& metadata)
	{
		setMetadata(metadata);
		iImg = Utils::getMatHeader(FrameMetadataFactory::downcast<RawImageMetadata>(metadata));
		oImg = Utils::getMatHeader(FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata));

		int inWidth = iImg.cols;
		int inHeight = iImg.rows;
		int outWidth = oImg.cols;
		int outHeight = oImg.rows;

		double inCenterX = inWidth / 2.0;
		double inCenterY = inHeight / 2.0;

		double outCenterX = outWidth / 2.0;
		double outCenterY = outHeight / 2.0;

		double tx = outCenterX - inCenterX; // translation factor in the x-axis
		double ty = outCenterY - inCenterY; // translation factor in the y-axis

		double cx = props.x + inCenterX; // x-coordinate of the center of rotation
		double cy = props.y + inCenterY; // y-coordinate of the center of rotation
		cv::Point2f center(cx, cy); // Center of rotation

		double scale = props.scale;
		rot_mat = cv::getRotationMatrix2D(center, props.angle, scale); // Get the rotation matrix
		rot_mat.at<double>(0, 2) += tx; // Apply translation in the x-axis
		rot_mat.at<double>(1, 2) += ty; // Apply translation in the y-axis
	}

	bool compute()
	{
		cv::warpAffine(iImg, oImg, rot_mat, oImg.size()); // Apply the rotation and translation to the output image
		return true;
	}

	bool setPtrs()
	{
		iImg.data = static_cast<uint8_t*>(inputFrame->data());
		oImg.data = static_cast<uint8_t*>(outputFrame->data());
		return true;
	}
protected:
	cv::Mat iImg;
	cv::Mat oImg;
	cv::Mat rot_mat;
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
#ifdef APRA_CUDA_ENABLED
	if (memType != FrameMetadata::MemType::CUDA_DEVICE && memType != FrameMetadata::MemType::DMABUF && memType != FrameMetadata::MemType::HOST)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is expected to be CUDA_DEVICE or DMABUF. Actual<" << memType << ">";
		return false;
	}
#else
	if (memType != FrameMetadata::MemType::HOST)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is expected to be HOST. Actual<" << memType << ">";
		return false;
	}
#endif

#ifdef APRA_CUDA_ENABLED
	if (mProp.type == AffineTransformProps::TransformType::USING_OPENCV && memType == FrameMetadata::MemType::CUDA_DEVICE)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is CUDA_DEVICE, but the transform type is USING_OPENCV";
		return false;
	}

	else if (mProp.type == AffineTransformProps::TransformType::USING_OPENCV && memType == FrameMetadata::MemType::DMABUF)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is DMABUF, but the transform type is USING_OPENCV";
		return false;
	}

	else if (mProp.type == AffineTransformProps::TransformType::USING_NPPI && memType == FrameMetadata::MemType::HOST)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is HOST, but the transform type is USING_NPPI";
		return false;
	}
#else
	if (mProp.type == AffineTransformProps::TransformType::USING_NPPI && memType == FrameMetadata::MemType::HOST)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is HOST, but the transform type is USING_NPPI";
		return false;
	}
#endif
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
#ifdef APRA_CUDA_ENABLED
	if (memType != FrameMetadata::MemType::CUDA_DEVICE && memType != FrameMetadata::MemType::DMABUF && memType != FrameMetadata::MemType::HOST)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input memType is expected to be CUDA_DEVICE or DMABUF . Actual<" << memType << ">";
		return false;
	}
#else
	if (memType != FrameMetadata::MemType::HOST)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input memType is expected to be HOST . Actual<" << memType << ">";
		return false;
	}
#endif
	return true;
}

void AffineTransform::addInputPin(framemetadata_sp &metadata, string &pinId)
{

	Module::addInputPin(metadata, pinId);
	FrameMetadata::MemType memType = metadata->getMemType();
#ifdef APRA_CUDA_ENABLED
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
#else
	if (memType == FrameMetadata::MemType::HOST)
	{
		mDetail.reset(new DetailHost(mProp));
	}
#endif
	else
	{
		throw std::runtime_error("Memory Type not supported");
	}

	mDetail->mSetMetadata(metadata);
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
	mDetail->inputFrame = frames.cbegin()->second;
	mDetail->outputFrame = makeFrame(mDetail->mOutputFrameLength);
	mDetail->setPtrs();

	mDetail->compute();
	frames.insert(make_pair(mDetail->mOutputPinId, mDetail->outputFrame));
	send(frames);

	return true;
}

bool AffineTransform::processSOS(frame_sp &frame)
{
	auto metadata = frame->getMetadata();
	mDetail->mSetMetadata(metadata);
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
	AffineTransformProps props(mDetail->props.type, 0);
	bool ret = Module::handlePropsChange(frame, props);
	mDetail->setProps(props);
	return ret;
}