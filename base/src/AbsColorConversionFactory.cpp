#include "AbsColorConversionFactory.h"
#include "ColorConversionStrategy.h"

boost::shared_ptr<DetailAbstract> AbsColorConversionFactory::create(framemetadata_sp input, framemetadata_sp output)
{
	boost::shared_ptr<DetailAbstract> mapper;
	static std::map<std::pair<ImageMetadata::ImageType, ImageMetadata::ImageType>, boost::shared_ptr<DetailAbstract>> cache;

	auto memType = input->getMemType();
	auto inputFrameType = input->getFrameType();
	auto outputFrameType = output->getFrameType();
	ImageMetadata::ImageType inputImageType;
	ImageMetadata::ImageType outputImageType;

	std::pair<ImageMetadata::ImageType, ImageMetadata::ImageType> requiredMapper;

	if (cache.find(requiredMapper) != cache.end())
	{
		return cache[requiredMapper];
	}

	if (memType == FrameMetadata::HOST && inputFrameType == FrameMetadata::RAW_IMAGE && outputFrameType == FrameMetadata::RAW_IMAGE)
	{
		auto rawInputMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(input);
		auto rawOutputMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(output);
		inputImageType = rawInputMetadata->getImageType();
		outputImageType = rawOutputMetadata->getImageType();

		if (inputImageType == ImageMetadata::RGB && outputImageType == ImageMetadata::BGR)
		{
			mapper = boost::shared_ptr<DetailAbstract>(new CpuRGB2BGR());
		}
		else if (inputImageType == ImageMetadata::BGR && outputImageType == ImageMetadata::RGB)
		{
			mapper = boost::shared_ptr<DetailAbstract>(new CpuBGR2RGB());
		}
		else if (inputImageType == ImageMetadata::BGR && outputImageType == ImageMetadata::MONO)
		{
			mapper = boost::shared_ptr<DetailAbstract>(new CpuBGR2MONO());
		}
		else if (inputImageType == ImageMetadata::RGB && outputImageType == ImageMetadata::MONO)
		{
			mapper = boost::shared_ptr<DetailAbstract>(new CpuRGB2MONO());
		}
		else if (inputImageType == ImageMetadata::BAYERBG8 && outputImageType == ImageMetadata::RGB)
		{
			mapper = boost::shared_ptr<DetailAbstract>(new CpuBayerBG82RGB());
		}
		else if (inputImageType == ImageMetadata::BAYERBG8 && outputImageType == ImageMetadata::MONO)
		{
			mapper = boost::shared_ptr<DetailAbstract>(new CpuBayerBG82Mono());
		}
		else if (inputImageType == ImageMetadata::BAYERGB8 && outputImageType == ImageMetadata::RGB)
		{
			mapper = boost::shared_ptr<DetailAbstract>(new CpuBayerGB82RGB());
		}
		else if (inputImageType == ImageMetadata::BAYERRG8 && outputImageType == ImageMetadata::RGB)
		{
			mapper = boost::shared_ptr<DetailAbstract>(new CpuBayerRG82RGB());
		}
		else if (inputImageType == ImageMetadata::BAYERGR8 && outputImageType == ImageMetadata::RGB)
		{
			mapper = boost::shared_ptr<DetailAbstract>(new CpuBayerGR82RGB());
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
			mapper = boost::shared_ptr<DetailAbstract>(new CpuRGB2YUV420Planar());
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
			mapper = boost::shared_ptr<DetailAbstract>(new CpuYUV420Planar2RGB());
		}
	}
	else
	{
		throw AIPException(AIP_FATAL, "this conversion is not supported");
	}
	
	requiredMapper = std::make_pair(inputImageType, outputImageType);
	cache[requiredMapper] = mapper;
	return mapper;
}