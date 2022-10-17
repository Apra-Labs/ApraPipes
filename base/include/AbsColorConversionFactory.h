#pragma once
#include "boost/shared_ptr.hpp"
#include "FrameMetadata.h"

class DetailAbstract;
class AbsColorConversionFactory
{
public:
	static boost::shared_ptr<DetailAbstract> create(framemetadata_sp input, framemetadata_sp output);
};