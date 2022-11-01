#pragma once
#include "boost/shared_ptr.hpp"
#include "FrameMetadata.h"
#include "ColorConversionXForm.h"

class AbsColorConversionFactory
{
public:
	static boost::shared_ptr<DetailAbstract> create(framemetadata_sp input, framemetadata_sp output, cv::Mat& inpImg, cv::Mat& outImg);
};