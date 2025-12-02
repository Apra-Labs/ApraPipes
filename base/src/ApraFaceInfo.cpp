#include "ApraFaceInfo.h"
#include "Overlay.h"

// IMetadataConvertible interface implementation

std::vector<FrameMetadata::FrameType> ApraFaceInfo::getConvertibleTypes() const
{
	// ApraFaceInfo can convert to overlay visualization
	return {FrameMetadata::OVERLAY_INFO_IMAGE};
}

FrameMetadata::FrameType ApraFaceInfo::getNativeType() const
{
	return FrameMetadata::FACEDETECTS_INFO;
}

void* ApraFaceInfo::convertTo(FrameMetadata::FrameType targetType, size_t& outSize) const
{
	// Check if requested conversion is supported
	if (targetType != FrameMetadata::OVERLAY_INFO_IMAGE) {
		outSize = 0;
		return nullptr;
	}

	// Create DrawingOverlay to hold both rectangle and text
	// DrawingOverlay is the external interface with mGetSerializeSize() support
	DrawingOverlay* drawing = new DrawingOverlay();

	// Add rectangle for the bounding box
	RectangleOverlay* rect = new RectangleOverlay();
	rect->x1 = x1;
	rect->y1 = y1;
	rect->x2 = x2;
	rect->y2 = y2;
	drawing->add(rect);

	// Add text overlay for the confidence score
	TextOverlay* text = new TextOverlay();
	text->x = x1;
	text->y = y1 - 5;  // Position above the rectangle

	// Format confidence score as percentage
	std::ostringstream scoreStream;
	scoreStream << std::fixed << std::setprecision(1) << (score * 100.0f) << "%";
	text->text = scoreStream.str();
	text->fontSize = 0.5;

	drawing->add(text);

	// Calculate output size (for future frame allocation)
	outSize = drawing->mGetSerializeSize();

	return drawing;
}
