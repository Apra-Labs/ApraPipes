#pragma once
#include "Overlay.h"
#include "OverlayFactory.h"

void CircleOverlay::serialize(boost::archive::binary_oarchive &oa)
{
	oa << primitiveType << x1 << y1 << radius;
}

size_t CircleOverlay::getSerializeSize()
{
	return sizeof(CircleOverlay) + sizeof(x1) + sizeof(y1) + sizeof(radius) + sizeof(primitiveType) + 32;
}

void CircleOverlay::deserialize(boost::archive::binary_iarchive &ia)
{
	ia >> x1 >> y1 >> radius;
}

void LineOverlay::serialize(boost::archive::binary_oarchive &oa)
{
	oa << primitiveType << x1 << y1 << x2 << y2;
}

size_t LineOverlay::getSerializeSize()
{
	return sizeof(LineOverlay) + sizeof(x1) + sizeof(y1) + sizeof(x2) + sizeof(y2) + sizeof(primitiveType) + 32;
}

void LineOverlay::deserialize(boost::archive::binary_iarchive &ia)
{
	ia >> x1 >> y1 >> x2 >> y2;
}

void RectangleOverlay::serialize(boost::archive::binary_oarchive &oa)
{
	oa << primitiveType << x1 << y1 << x2 << y2;
}

size_t RectangleOverlay::getSerializeSize()
{
	return sizeof(RectangleOverlay) + sizeof(x1) + sizeof(y1) + sizeof(x2) + sizeof(y2) + sizeof(primitiveType) + 32;
}

void RectangleOverlay::deserialize(boost::archive::binary_iarchive &ia)
{
	ia >> x1 >> y1 >> x2 >> y2;
}

void CompositeOverlay::add(OverlayInfo *component)
{
	gList.push_back(component);
}

void CompositeOverlay::serialize(boost::archive::binary_oarchive &oa)
{
	oa << primitiveType << gList.size();
}

void CompositeOverlay::accept(OverlayInfoVisitor *visitor)
{
	visitor->visit(this);
	for (auto comp : gList)
	{
		comp->accept(visitor);
	}
}

vector<OverlayInfo *> CompositeOverlay::getList()
{
	return gList;
}

void CompositeOverlay::deserialize(boost::archive::binary_iarchive &ia)
{
	size_t archive_size;
	ia >> archive_size;
	Primitive childPrimitiveType;
	for (int i = 0; i < archive_size; i++)
	{
		ia >> childPrimitiveType;
		OverlayInfo *overlayInfo = OverlayFactory::create(childPrimitiveType);
		gList.push_back(overlayInfo);
		overlayInfo->deserialize(ia);
	}
}

size_t DrawingOverlay::mGetSerializeSize()
{
	OverlayInfoSerializeSizeVisitor *visitor = new OverlayInfoSerializeSizeVisitor();
	accept(visitor);
	return visitor->totalSize;
}

void DrawingOverlay::deserialize(frame_sp frame)
{
	boost::iostreams::basic_array_source<char> device((char *)frame->data(), frame->size());
	boost::iostreams::stream<boost::iostreams::basic_array_source<char>> sink(device);
	boost::archive::binary_iarchive ia(sink);

	ia >> primitiveType;
	size_t archive_size;
	ia >> archive_size;
	Primitive childPrimitiveType;
	for (int i = 0; i < archive_size; i++)
	{
		ia >> childPrimitiveType;
		DrawingOverlayBuilder *drawBuilderInfo = BuilderOverlayFactory::create(childPrimitiveType);
		OverlayInfo *overlayInfo = drawBuilderInfo->deserialize(ia);
		gList.push_back(overlayInfo);
	}
}

void DrawingOverlay::serialize(frame_sp frame)
{
	boost::iostreams::basic_array_sink<char> device_sink((char *)frame->data(), frame->size());
	boost::iostreams::stream<boost::iostreams::basic_array_sink<char>> s_sink(device_sink);
	boost::archive::binary_oarchive oa(s_sink);

	OverlayInfoSerializerVisitor *visitor = new OverlayInfoSerializerVisitor(oa);

	accept(visitor);
}

OverlayInfo *CompositeOverlayBuilder::deserialize(boost::archive::binary_iarchive &ia)
{
	compositeOverlay->deserialize(ia);
	return compositeOverlay;
}

OverlayInfo *LineOverlayBuilder::deserialize(boost::archive::binary_iarchive &ia)
{
	lineOverlay->deserialize(ia);
	return lineOverlay;
}

OverlayInfo *RectangleOverlayBuilder::deserialize(boost::archive::binary_iarchive &ia)
{
	rectangleOverlay->deserialize(ia);
	return rectangleOverlay;
}

OverlayInfo *CircleOverlayBuilder::deserialize(boost::archive::binary_iarchive &ia)
{
	circleOverlay->deserialize(ia);
	return circleOverlay;
}

DrawingOverlayBuilder *BuilderOverlayFactory::create(Primitive primitiveType)
{
	if (primitiveType == Primitive::RECTANGLE)
	{
		RectangleOverlayBuilder *rectangleOverlaybuilder = new RectangleOverlayBuilder();
		return rectangleOverlaybuilder;
	}

	else if (primitiveType == Primitive::LINE)
	{
		LineOverlayBuilder *lineOverlaybuilder = new LineOverlayBuilder();
		return lineOverlaybuilder;
	}

	else if (primitiveType == Primitive::CIRCLE)
	{
		CircleOverlayBuilder *circleOverlaybuilder = new CircleOverlayBuilder();
		return circleOverlaybuilder;
	}

	else if (primitiveType == Primitive::COMPOSITE)
	{
		CompositeOverlayBuilder *compositeOverlaybuilder = new CompositeOverlayBuilder();
		return compositeOverlaybuilder;
	}

	else
		return NULL;
}