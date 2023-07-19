#pragma once 
#include "Overlay.h"

class OverlayFactory
{
public:
	static OverlayInfo* create(Primitive primitiveType);
};

class BuilderOverlayFactory
{
public:
	static DrawingOverlayBuilder* create(Primitive primitiveType);
	void accept(OverlayInfoVisitor* visitor);
};
