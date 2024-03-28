#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "SquareMask.h"

__global__ void rgbaSquareMaskKernel(unsigned char* frame, int width, int height, int maskX, int maskY, int maskSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if within the image bounds
    if (x < width && y < height) 
    {
        int index = y * width + 4 * x;
        // int maskIndex = (y - maskY) * maskSize + (x - maskX);

        // Check if within the mask bounds
        if (x >= maskX && x < maskX + maskSize && y >= maskY && y < maskY + maskSize) {
            // Overlay the mask on the frame (you can customize this part)
            // frame[index] = 0;  // Set pixel to white (or any other value you prefer)

            frame[index + 0] = 0;
            frame[index + 1] = 0;
            frame[index + 2] = 0;
            frame[index + 3] = 0;
        }
    }
}

void applySquareMaskForRGBA(unsigned char *src, unsigned char *dst, int width, int height, int maskSize, cudaStream_t stream)
{
    dim3 dim_block(32, 32);
    dim3 dim_grid((1000 + dim_block.x - 1) / dim_block.x, (height + dim_block.y - 1) / dim_block.y);
    int maskX = 50;
    int maskY = 50;
    rgbaSquareMaskKernel<<<dim_grid, dim_block, 0, stream>>>(src, width, height, maskX, maskY, maskSize);
}