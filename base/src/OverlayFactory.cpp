#include "OverlayFactory.h"

OverlayInfo* OverlayFactory::create(Primitive primitiveType)
{
	if (primitiveType == Primitive::RECTANGLE)
	{
		RectangleOverlay* rectangleOverlay = new RectangleOverlay();
		return rectangleOverlay;
	}

	else if (primitiveType == Primitive::LINE)
	{
		LineOverlay* lineOverlay = new LineOverlay();
		return lineOverlay;
	}

	else if (primitiveType == Primitive::CIRCLE)
	{
		CircleOverlay* circleOverlay = new CircleOverlay();
		return circleOverlay;
	}

	else if (primitiveType == Primitive::COMPOSITE)
	{
		CompositeOverlay* compositeOverlay = new CompositeOverlay();
		return compositeOverlay;
	}
	return nullptr;
}