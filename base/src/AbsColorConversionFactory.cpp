#include "AbsColorConversionFactory.h"
#include "ColorConversionStrategy.h"
#include "RawImageMetadata.h"
#include "RawImagePlanarMetadata.h"


std::shared_ptr<DetailAbstract> AbsColorConversionFactory::create(framemetadata_sp input, framemetadata_sp output, cv::Mat& inpImg, cv::Mat& outImg)
{
	std::shared_ptr<DetailAbstract> mapper;

	auto memType = input->getMemType();
	auto inputFrameType = input->getFrameType();
	auto outputFrameType = output->getFrameType();
	ImageMetadata::ImageType inputImageType;
	ImageMetadata::ImageType outputImageType;

	if (memType == FrameMetadata::HOST && inputFrameType == FrameMetadata::RAW_IMAGE && outputFrameType == FrameMetadata::RAW_IMAGE)
	{
		auto rawInputMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(input);
		auto rawOutputMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(output);
		inputImageType = rawInputMetadata->getImageType();
		outputImageType = rawOutputMetadata->getImageType();
		
		inpImg = Utils::getMatHeader(rawInputMetadata);
		outImg = Utils::getMatHeader(rawOutputMetadata);

		if (inputImageType == ImageMetadata::RGB && outputImageType == ImageMetadata::BGR)
		{
			mapper = std::shared_ptr<DetailAbstract>(new CpuRGB2BGR(inpImg, outImg));
		}
		else if (inputImageType == ImageMetadata::BGR && outputImageType == ImageMetadata::RGB)
		{
			mapper = std::shared_ptr<DetailAbstract>(new CpuBGR2RGB(inpImg, outImg));
		}
		else if (inputImageType == ImageMetadata::BGR && outputImageType == ImageMetadata::MONO)
		{
			mapper = std::shared_ptr<DetailAbstract>(new CpuBGR2MONO(inpImg, outImg));
		}
		else if (inputImageType == ImageMetadata::RGB && outputImageType == ImageMetadata::MONO)
		{
			mapper = std::shared_ptr<DetailAbstract>(new CpuRGB2MONO(inpImg, outImg));
		}
		else if (inputImageType == ImageMetadata::BAYERBG8 && outputImageType == ImageMetadata::RGB)
		{
			mapper = std::shared_ptr<DetailAbstract>(new CpuBayerBG82RGB(inpImg, outImg));
		}
		else if (inputImageType == ImageMetadata::BAYERBG8 && outputImageType == ImageMetadata::MONO)
		{
			mapper = std::shared_ptr<DetailAbstract>(new CpuBayerBG82Mono(inpImg, outImg));
		}
		else if (inputImageType == ImageMetadata::BAYERGB8 && outputImageType == ImageMetadata::RGB)
		{
			mapper = std::shared_ptr<DetailAbstract>(new CpuBayerGB82RGB(inpImg, outImg));
		}
		else if (inputImageType == ImageMetadata::BAYERRG8 && outputImageType == ImageMetadata::RGB)
		{
			mapper = std::shared_ptr<DetailAbstract>(new CpuBayerRG82RGB(inpImg, outImg));
		}
		else if (inputImageType == ImageMetadata::BAYERGR8 && outputImageType == ImageMetadata::RGB)
		{
			mapper = std::shared_ptr<DetailAbstract>(new CpuBayerGR82RGB(inpImg, outImg));
		}
		else
		{
			throw AIPException(AIP_FATAL, "This conversion is not supported");
		}
	}
	else if (memType == FrameMetadata::HOST && inputFrameType == FrameMetadata::RAW_IMAGE && outputFrameType == FrameMetadata::RAW_IMAGE_PLANAR)
	{
		auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(input);
		auto rawOutputMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(output);
		inputImageType = rawMetadata->getImageType();
		outputImageType = rawOutputMetadata->getImageType();

		if (inputImageType == ImageMetadata::RGB && outputImageType == ImageMetadata::YUV420)
		{
			auto height = rawOutputMetadata->getHeight(0);
			int outputRows = height * 1.5;

			inpImg = Utils::getMatHeader(rawMetadata);
			outImg = Utils::getMatHeader(rawOutputMetadata, outputRows);
			mapper = std::shared_ptr<DetailAbstract>(new CpuRGB2YUV420Planar(inpImg, outImg));//inpImg,outImg
		}

		else
		{
			throw AIPException(AIP_FATAL, "This conversion is not supported");
		}

	}
	else if (memType == FrameMetadata::HOST && inputFrameType == FrameMetadata::RAW_IMAGE_PLANAR && outputFrameType == FrameMetadata::RAW_IMAGE)
	{
		auto rawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(input);
		auto rawPlanarMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(output);
		inputImageType = rawMetadata->getImageType();
		outputImageType = rawPlanarMetadata->getImageType();

		if (inputImageType == ImageMetadata::YUV420 && outputImageType == ImageMetadata::RGB)
		{
			auto height = rawMetadata->getHeight(0);
			int inputRows = height * 1.5;

			inpImg = Utils::getMatHeader(rawMetadata, inputRows);
			outImg = Utils::getMatHeader(rawPlanarMetadata);
			mapper = std::shared_ptr<DetailAbstract>(new CpuYUV420Planar2RGB(inpImg, outImg));
		}

		else
		{
			throw AIPException(AIP_FATAL, "This conversion is not supported");
		}

	}
	else
	{
		throw AIPException(AIP_FATAL, "This conversion is not supported");
	}

	return mapper;
}