#pragma once

#include "nppdefs.h"
#include <stdint.h>

void applyCircularMask(uint8_t* src, uint8_t* dst, int width, int height, int cx, int cy, int radius, cudaStream_t stream);

void applyCircularMaskRGBA(uint8_t* src, uint8_t* dst, int width, int height, int cx, int cy, int radius, cudaStream_t stream);

void applyCircularMaskYUV444(unsigned char* src, unsigned char* dst, int width, int height, int cx, int cy, int radius, cudaStream_t stream);

void applyDiamondMaskYUV444(unsigned char *src, unsigned char *dst, int width, int height, int stride, int cx, int cy, int radius, cudaStream_t stream);

void addKernelIndicatorSquareMask(unsigned char *src, unsigned char *dst, int width, int height, int stride, cudaStream_t stream);

void applyOctagonalMask(unsigned char * src, unsigned char * dst, int width, int height, float* triangle1X, float* triangle1Y, float* triangle2X, float* triangle2Y, float* triangle3X, float* triangle3Y, float* triangle4X, float* triangle4Y, cudaStream_t stream);

void applyCircularMaskForRGBANew(unsigned char *src, unsigned char *dst, int width, int height, int cx, int cy, int radius, cudaStream_t stream);

void generateRGBAOctagonalKernel(unsigned char *src, unsigned char *dst, int width, int height, int cx, int cy, int radius, cudaStream_t stream);

void applySquareRotationIndicator(unsigned char *src, unsigned char *dst, int width, int height, cudaStream_t stream);