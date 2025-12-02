#pragma once

#include <opencv2/core/types_c.h>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/array.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include "IMetadataConvertible.h"
#include <sstream>
#include <iomanip>

// Forward declaration to avoid circular dependency
class DrawingOverlay;

/**
 * @brief Face detection information structure with metadata conversion support
 *
 * Represents a detected face with bounding box coordinates and confidence score.
 * Implements IMetadataConvertible to enable automatic conversion to overlay visualization.
 */
class ApraFaceInfo : public IMetadataConvertible
{
public:
	float x1, x2, y1, y2, score;

	ApraFaceInfo(): x1(0), y1(0), x2(0), y2(0), score(0)
	{

	}

	size_t getSerializeSize()
	{
		return sizeof(ApraFaceInfo) + sizeof(x1) + sizeof(x2) + sizeof(y1) + sizeof(y2) + sizeof(score) + 32;
	}

	// IMetadataConvertible interface implementation
	std::vector<FrameMetadata::FrameType> getConvertibleTypes() const override;
	void* convertTo(FrameMetadata::FrameType targetType, size_t& outSize) const override;
	FrameMetadata::FrameType getNativeType() const override;

private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar &x1 &x2 &y1 &y2 &score;
	}
};