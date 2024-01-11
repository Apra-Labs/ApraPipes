#pragma once

#include "nppdefs.h"
#include <stdint.h>


void applySquareMaskForRGBA(unsigned char *src, unsigned char *dst, int width, int height, int maskSize, cudaStream_t stream);