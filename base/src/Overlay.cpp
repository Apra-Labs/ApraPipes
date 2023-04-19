#pragma once 
#include "Overlay.h"
#include "OverlayFactory.h"

void CircleOverlay::serialize(boost::archive::binary_oarchive& oa)
{
	oa << primitiveType << x1 << y1 << radius;
}

size_t CircleOverlay::getSerializeSize()
{
	return sizeof(CircleOverlay) + sizeof(x1) + sizeof(y1) + sizeof(radius) + sizeof(primitiveType) + 32;
}

void CircleOverlay::deserialize(boost::archive::binary_iarchive& ia)
{
	ia >> x1 >> y1 >> radius;
}

void LineOverlay::serialize(boost::archive::binary_oarchive& oa)
{
	oa << primitiveType << x1 << y1 << x2 << y2;
}

size_t LineOverlay::getSerializeSize()
{
	return sizeof(LineOverlay) + sizeof(x1) + sizeof(y1) + sizeof(x2) + sizeof(y2) + sizeof(primitiveType) + 32;
}

void LineOverlay::deserialize(boost::archive::binary_iarchive& ia)
{
	ia >> x1 >> y1 >> x2 >> y2;
}

void RectangleOverlay::serialize(boost::archive::binary_oarchive& oa)
{
	oa << primitiveType << x1 << y1 << x2 << y2;
}

size_t RectangleOverlay::getSerializeSize()
{
	return sizeof(RectangleOverlay) + sizeof(x1) + sizeof(y1) + sizeof(x2) + sizeof(y2) + sizeof(primitiveType) + 32;
}

void RectangleOverlay::deserialize(boost::archive::binary_iarchive& ia)
{
	ia >> x1 >> y1 >> x2 >> y2;
}

void CompositeOverlay::add(OverlayInfo* component)
{
	gList.push_back(component);
}

void CompositeOverlay::serialize(boost::archive::binary_oarchive& oa)
{
	oa << primitiveType << gList.size();
}

void CompositeOverlay::accept(OverlayInfoVisitor* visitor)
{
	visitor->visit(this);
	for (auto comp : gList)
	{
		comp->accept(visitor);
	}
}

void CompositeOverlay::deserialize(boost::archive::binary_iarchive& ia)
{
	size_t archive_size;
	ia >> archive_size;
	for (int i = 0; i < archive_size; i++)
	{
		ia >> primitiveType;
		OverlayInfo* overlayInfo = OverlayFactory::create(primitiveType);
		overlayInfo->deserialize(ia);
		gList.push_back(overlayInfo);
	}
}

size_t CompositeOverlay::getSerializeSize()
{
	OverlayInfoSerializeSizeVisitor* visitor = new OverlayInfoSerializeSizeVisitor();
	accept(visitor);
	return visitor->totalSize;
}

void DrawingOverlay::deserialize(frame_sp frame)
{
	boost::iostreams::basic_array_source<char> device((char*)frame->data(), frame->size());
	boost::iostreams::stream<boost::iostreams::basic_array_source<char> > sink(device);
	boost::archive::binary_iarchive ia(sink);

	Primitive primitiveType;
	ia >> primitiveType;
	size_t archive_size;
	ia >> archive_size;
	for (int i = 0; i < archive_size; i++)
	{
		ia >> primitiveType;
		DrawingOverlayBuilder* drawBuilderInfo = BuilderOverlayFactory::create(primitiveType);
		drawBuilderInfo->deserialize(ia);
	}
}

void DrawingOverlay::serialize(frame_sp frame)
{
	boost::iostreams::basic_array_sink<char> device_sink((char*)frame->data(), frame->size());
	boost::iostreams::stream<boost::iostreams::basic_array_sink<char> > s_sink(device_sink);
	boost::archive::binary_oarchive oa(s_sink);

	OverlayInfoSerializerVisitor* visitor = new OverlayInfoSerializerVisitor(oa);

	accept(visitor);
}

void CompositeOverlayBuilder::deserialize(boost::archive::binary_iarchive& ia)
{
	compositeOverlay->deserialize(ia);
}

void LineOverlayBuilder::deserialize(boost::archive::binary_iarchive& ia)
{
	lineOverlay->deserialize(ia);
}

void RectangleOverlayBuilder::deserialize(boost::archive::binary_iarchive& ia)
{
	rectangleOverlay->deserialize(ia);
}

void CircleOverlayBuilder::deserialize(boost::archive::binary_iarchive& ia)
{
	circleOverlay->deserialize(ia);
}

DrawingOverlayBuilder* BuilderOverlayFactory::create(Primitive primitiveType)
{
	if (primitiveType == Primitive::RECTANGLE)
	{
		RectangleOverlayBuilder* rectangleOverlaybuilder = new RectangleOverlayBuilder();
		return rectangleOverlaybuilder;
	}

	else if (primitiveType == Primitive::LINE)
	{
		LineOverlayBuilder* lineOverlaybuilder = new LineOverlayBuilder();
		return lineOverlaybuilder;
	}

	else if (primitiveType == Primitive::CIRCLE)
	{
		CircleOverlayBuilder* circleOverlaybuilder = new CircleOverlayBuilder();
		return circleOverlaybuilder;
	}

	else if (primitiveType == Primitive::COMPOSITE)
	{
		CompositeOverlayBuilder* compositeOverlaybuilder = new CompositeOverlayBuilder();
		return compositeOverlaybuilder;
	}

	else return NULL;
}