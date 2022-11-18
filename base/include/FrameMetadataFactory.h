#pragma once

#include "FrameMetadata.h"

class FrameMetadataFactory
{
public:
	template<class T>
	static T* downcast(framemetadata_sp metadata)
	{
		auto ptr = dynamic_cast<T*>(metadata.get());
		if (!ptr)
		{
			throw AIPException(AIP_FATAL, "Wrong casting.");
		}

		return ptr;
	}	
};