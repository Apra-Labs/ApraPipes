#pragma once

#include "nppdefs.h"

void launchYUVOverlayKernel(const Npp8u* src[3], const Npp8u* overlay[3], Npp8u* dst[3], Npp32f alpha, int srcStep[2], int overlayStep[2], NppiSize size, cudaStream_t stream);