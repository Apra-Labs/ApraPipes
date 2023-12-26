#pragma once

#include "nppdefs.h"
#include <stdint.h>

void applySquareRotationIndicator(unsigned char *src, unsigned char *dst, int width, int height, int rotationDegree, cudaStream_t stream);