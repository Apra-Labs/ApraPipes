#pragma once

#include "nppdefs.h"

void lanuchAPPYUV411ToYUV444(const Npp8u* src, int nSrcStep, Npp8u* dst[3], int rDstStep, NppiSize oSizeROI, cudaStream_t stream);