#pragma once

#include "nppdefs.h"

void launchYUV420Effects(const Npp8u* y, const Npp8u* u, const Npp8u* v, Npp8u* Y, Npp8u* U, Npp8u* V,
	Npp32f brighness, Npp32f contrast, Npp32f hue, Npp32f saturation,
	int step_y, int step_uv, NppiSize size, cudaStream_t stream);

// brightness - given value will be added and clamped to 255/128 - [-255 255] - 0 means no change
// contrast - given value will be multiplied and clamped to 255/128 - any value  1 means no change

// hsv space - range of pixel values is 0 - 1 - didn't convert to 255 to save computation

// hue - given value will be added to "h" and clamped to 1 in hsv space - [-1 1] - 0 means no change
// satuarion - given value will be multiplied to "s" and clamped to 1 in hsv space - any value - 1 means no change