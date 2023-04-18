#pragma once 
#include "Overlay.h"

class OverlayFactory
{
public:
	static OverlayInfo* create(Primitive primitiveType);
};
