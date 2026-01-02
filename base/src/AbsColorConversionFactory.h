#pragma once
#include <memory>
#include "FrameMetadata.h"
#include "ColorConversionXForm.h"

class AbsColorConversionFactory
{
public:
	static std::shared_ptr<DetailAbstract> create(framemetadata_sp input, framemetadata_sp output, cv::Mat& inpImg, cv::Mat& outImg);
};