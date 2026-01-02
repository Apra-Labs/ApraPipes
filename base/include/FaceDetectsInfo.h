#pragma once

#include "FrameMetadata.h"
#include "Frame.h"
#include "Utils.h"
#include "Module.h"
#include <vector>
#include <boost/serialization/vector.hpp>
#include "ApraFaceInfo.h"
#include <opencv2/core/types.hpp>

class FaceDetectsInfo
{
public:
	FaceDetectsInfo() : facesFound(false)
	{
	}

	static FaceDetectsInfo deSerialize(frame_container &frames)
	{
		FaceDetectsInfo result;
		auto frame = Module::getFrameByType(frames, FrameMetadata::FACEDETECTS_INFO);
		if (!frame.get())
		{
			return result;
		}
		Utils::deSerialize<FaceDetectsInfo>(result, frame->data(), frame->size());

		return result;
	}

	void serialize(void *buffer, size_t size)
	{
		auto &info = *this;
		facesFound = faces.size() > 0;
		Utils::serialize<FaceDetectsInfo>(info, buffer, size);
	}

	size_t getSerializeSize()
	{
		size_t totalFacesSize = 0;
		for (auto &face : faces)
		{
			totalFacesSize += face.getSerializeSize();
		}
		return sizeof(FaceDetectsInfo) + sizeof(facesFound) + sizeof(faces) + totalFacesSize;
	}

	bool facesFound;
	std::vector<ApraFaceInfo> faces;

private:
	friend class boost::serialization::access;
	template <class Archive>
	void serialize(Archive &ar, const unsigned int /*file_version*/)
	{
		ar &facesFound;
		ar &faces;
	}
};