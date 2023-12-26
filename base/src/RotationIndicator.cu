#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "RotationIndicator.h"

__global__ void applySquareRotationIndicatorKernel(unsigned char *ptr, int width, int height, int rotationDegree)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= width || y >= height)
    {
        return;
    }
    int index = (y * width + 4 * x);

    // Calculate common values outside the loop
    int edgeLimit = width - 4;
    int isEdgePixel = 0;

    switch (rotationDegree)
    {
    case 90:
        isEdgePixel = (x > (1000 - 8));
        break;
    case 180:
        isEdgePixel = (y > (height - 8));
        break;
    case 270:
        isEdgePixel = (x < 8);
        break;
    default:
        isEdgePixel = (y < 8);
    }

    if (isEdgePixel)
    {
        // Set common color for all cases
        ptr[index + 0] = 0;
        ptr[index + 1] = 159;
        ptr[index + 2] = 148;
        // ptr[index + 3] = 160; // Uncomment if needed
    }
    
}


void applySquareRotationIndicator(unsigned char *src, unsigned char *dst, int width, int height, int rotationDegree, cudaStream_t stream)
{
    dim3 dim_block(32, 32);
    dim3 dim_grid((1000 + dim_block.x - 1) / dim_block.x, (height + dim_block.y - 1) / dim_block.y);
    applySquareRotationIndicatorKernel<<<dim_grid, dim_block, 0, stream>>>(src, width, height, rotationDegree);
}
