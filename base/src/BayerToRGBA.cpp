#include "BayerToRGBA.h"
#include "FrameMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include <opencv2/imgcodecs.hpp>
#include "Utils.h"
#define WIDTH 800
#define HEIGHT 800

class BayerToRGBA::Detail
{
public:
	Detail(BayerToRGBAProps &_props) : props(_props)
	{
	}
	~Detail() {}

public:
	size_t mFrameLength;
	framemetadata_sp mOutputMetadata;
	std::string mOutputPinId;
	int currPixVal = 0;

private:
	BayerToRGBAProps props;
};

BayerToRGBA::BayerToRGBA(BayerToRGBAProps _props) : Module(TRANSFORM, "BayerToRGBA", _props), props(_props), mFrameType(FrameMetadata::GENERAL)
{
	mDetail.reset(new Detail(_props));
}

BayerToRGBA::~BayerToRGBA() {}

bool BayerToRGBA::validateInputPins()
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
		LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be Raw_Image. Actual<" << frameType << ">";
		return false;
	}

	return true;
}

bool BayerToRGBA::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
		return false;
	}
	return true;
}

void BayerToRGBA::addInputPin(framemetadata_sp &metadata, string &pinId)
{
	Module::addInputPin(metadata, pinId);
	// auto rawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
	// RawImageMetadata outputMetadata(rawMetadata->getWidth(), rawMetadata->getHeight(), ImageMetadata::RGBA, CV_8UC4, 0, CV_8U, FrameMetadata::HOST, true);
	// mDetail->mOutputMetadata = framemetadata_sp(new RawImageMetadata());
	mDetail->mOutputMetadata = boost::shared_ptr<FrameMetadata>(new RawImagePlanarMetadata(WIDTH, HEIGHT, ImageMetadata::YUV420, size_t(0), CV_8U, FrameMetadata::HOST));
	// mDetail->mOutputMetadata = boost::shared_ptr<FrameMetadata>(new RawImagePlanarMetadata(400, 400, ImageMetadata::YUV420, size_t(0), CV_8U, FrameMetadata::HOST));
	mDetail->mOutputMetadata->copyHint(*metadata.get());
	// setMetadata(metadata);
	mDetail->mOutputPinId = addOutputPin(mDetail->mOutputMetadata);
}

std::string BayerToRGBA::addOutputPin(framemetadata_sp &metadata)
{
	// mDetail->mFrameLength = metadata->getDataSize();
	return Module::addOutputPin(metadata);
}

bool BayerToRGBA::init()
{
	return Module::init();
}

bool BayerToRGBA::term()
{
	return Module::term();
}

bool BayerToRGBA::process(frame_container &frames)
{
	auto frame = getFrameByType(frames, FrameMetadata::RAW_IMAGE);
	auto outFrame = makeFrame(WIDTH * HEIGHT * 1.5);

	if (isFrameEmpty(frame))
	{
		LOG_ERROR << "Frame is Empty";
		return true;
	}

	auto inpPtr = static_cast<uint16_t *>(frame->data());
	auto outPtr = static_cast<uint8_t *>(outFrame->data());

	memset(outPtr, 128, WIDTH * HEIGHT * 1.5); // remove for mono // yash earlier it was 128 

	for (auto i = 0; i < HEIGHT; i++)
	{
		auto inpPtr1 = inpPtr + i * 800;
		auto outPtr1 = outPtr + i * WIDTH;
		for (auto j = 0; j < WIDTH; j++)
		{
			*outPtr1++ = (*inpPtr1++) >> 2;
		}
	}	
	frames.insert(make_pair(mDetail->mOutputPinId, outFrame));
	send(frames);
	return true;

			// LOG_ERROR << "Coming Inside Process of Bayer To Bgra";

			/// check
			// LOG_ERROR << "Coming Inside Bayer To RGBA";
			// auto outFrame1 = makeFrame();
			// LOG_ERROR << "Coming Inside Bayer To RGBA 1";
			// frames.insert(make_pair(mDetail->mOutputPinId, outFrame1));
			// LOG_ERROR << "Coming Inside Bayer To RGBA 2";
			// send(frames);
			// LOG_ERROR << "Sending Frames From Bayer To RGBA";
			// return true;
			//check

			// auto frame = getFrameByType(frames, FrameMetadata::RAW_IMAGE);
			// struct timeval time_now{};
			// gettimeofday(&time_now, nullptr);
			// time_t msecs_time = (time_now.tv_sec * 1000) + (time_now.tv_usec / 1000);
			// LOG_ERROR << msecs_time;
			// if (isFrameEmpty(frame))
			// {
			//     LOG_ERROR << "Frame is Empty";
			// 	return true;
			// }

			// auto outFrame = makeFrame();

			// int width = 1024;
			// int height = 800;
			// auto inpPtr = static_cast<uint16_t *>(frame->data());
			// auto outPtr = static_cast<uint16_t *>(outFrame->data());

			// for (auto i = 0; i < 800; i++)
			// {
			// 	auto inpPtr1 = inpPtr + i * 800;
			// 	auto outPtr1 = outPtr + i * 800;
			// 	for (auto j = 0; j < 800; j++)
			// 	{
			// 		*outPtr1++ = (*inpPtr1++) >> 2;
			// 	}
			// }
			// memset(outPtr + 800 * 800, 0, 400 * 400);

			// int width = 400;
			// int height = 400;
			// auto inpPtr = static_cast<uint16_t *>(frame->data());
			// auto outPtr = static_cast<uint16_t *>(outFrame->data());

			// for (auto i = 0; i < 400; i++)
			// {
			// 	auto inpPtr1 = inpPtr + i * 400;
			// 	auto outPtr1 = outPtr + i * 400;
			// 	for (auto j = 0; j < 400; j++)
			// 	{
			// 		*outPtr1++ = (*inpPtr1++) >> 2;
			// 	}
			// }
			// memset(outPtr + 400 * 400, 0, 100 * 100);

			// // mDetail->oImg.data = static_cast<uint8_t *>(outFrame->data());
			// LOG_ERROR << "Coming Inside Process and before cvtcolor of Bayer To Bgra";
			// cv::Mat mat16uc1_bayer(height, width, CV_16UC1, mDetail->iImg.data);

			// cv::Mat mat16uc3_rgb(width, height, CV_16UC3);
			// cv::cvtColor(mat16uc1_bayer, mat16uc3_rgb, cv::COLOR_BayerGR2RGB);

			// // Convert the 16-bit per channel RGB image to 8-bit per channel
			// cv::Mat mat8uc3_rgb(width, height, CV_8UC3);
			// mat16uc3_rgb.convertTo(mat8uc3_rgb, CV_8UC3, 1.0 / 256); //this coul
			// LOG_ERROR << "printing size inside Bayer to bggr" << sizeof(mat8uc3_rgb.data);
			// memcpy(outFrame->data(), mat8uc3_rgb.data , sizeof(mat8uc3_rgb.data));

			// cv::cvtColor(mDetail->iImg, mDetail->oImg, cv::COLOR_BayerBG2RGBA);
			// cv::demosaicing(mDetail->iImg, mDetail->oImg, cv::COLOR_BayerBG2GRAY, 0);
			// LOG_ERROR << "Coming Inside Process and after cvtcolor of Bayer To Bgra";

			// auto frame = getFrameByType(frames, FrameMetadata::RAW_IMAGE);

			// if (isFrameEmpty(frame))
			// {
			// 	LOG_ERROR << "Frame is Empty";
			// 	return true;
			// }

			// auto outFrame = makeFrame();

			// auto outPtr = static_cast<uint16_t *>(outFrame->data());
			// LOG_ERROR << "Current Pix Value is " << mDetail->currPixVal;
			// if (mDetail->currPixVal > 255)
			// 	mDetail->currPixVal = (mDetail->currPixVal / 256);
			// memset(outPtr, mDetail->currPixVal, 800 * 800);
			// mDetail->currPixVal++;
			// frames.insert(make_pair(mDetail->mOutputPinId, outFrame));
			// send(frames);
			// return true;
		}

		void BayerToRGBA::setMetadata(framemetadata_sp & metadata)
		{
			if (!metadata->isSet())
			{
				return;
			}
			// auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
			mDetail->mOutputMetadata = boost::shared_ptr<FrameMetadata>(new RawImagePlanarMetadata(WIDTH, HEIGHT, ImageMetadata::YUV420, size_t(0), CV_8U, FrameMetadata::HOST));

			// auto rawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
			// RawImageMetadata outputMetadata(rawMetadata->getWidth(), rawMetadata->getHeight(), ImageMetadata::RGBA, CV_8UC4, 0, CV_8U, FrameMetadata::HOST, true);
			// mDetail->mOutputMetadata = framemetadata_sp(new RawImageMetadata());
			// mDetail->mOutputMetadata = boost::shared_ptr<FrameMetadata>(new RawImagePlanarMetadata(800, 800, ImageMetadata::YUV420, size_t(0), CV_8U, FrameMetadata::HOST));
			// RawImageMetadata outputMetadata(rawMetadata->getWidth(), rawMetadata->getHeight(), ImageMetadata::RGBA, CV_8UC4, 0, CV_8U, FrameMetadata::HOST, true);
			// mDetail->mOutputMetadata = framemetadata_sp(new RawImageMetadata());
			// DMAAllocator::setMetadata(mDetail->outputMetadata, 800, 800, ImageMetadata::YUV420);
			// mDetail->mOutputMetadata = boost::shared_ptr<FrameMetadata>(new RawImagePlanarMetadata(800, 800, ImageMetadata::YUV420, size_t(0), CV_8U, FrameMetadata::HOST));
			// auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
			// RawImageMetadata outputMetadata(rawMetadata->getWidth(), rawMetadata->getHeight(), ImageMetadata::RGBA, CV_8UC4, 0, CV_8U, FrameMetadata::HOST, true);
			// auto rawOutMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(mDetail->mOutputMetadata);
			// rawOutMetadata->setData(outputMetadata);
			// auto imageType = rawMetadata->getImageType();

			// mDetail->mFrameLength = mDetail->mOutputMetadata->getDataSize();

			// switch (imageType)
			// {
			// // case ImageMetadata::MONO:
			// // case ImageMetadata::BGR:
			// // case ImageMetadata::BGRA:
			// // case ImageMetadata::RGB:
			// // case ImageMetadata::RGBA:
			// case ImageMetadata::BG10:
			// 	break;
			// default:
			// 	throw AIPException(AIP_NOTIMPLEMENTED, "Conversion Not Supported for ImageType<" + std::to_string(imageType) + ">");
			// }
		}

		bool BayerToRGBA::processSOS(frame_sp & frame)
		{
			auto metadata = frame->getMetadata();
			setMetadata(metadata);
			return true;
		}
