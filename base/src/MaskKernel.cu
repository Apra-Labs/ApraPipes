#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "MaskKernel.h"

// here cx and cy are center coordinates , radius is radius of circle
__global__ void applyCircularKernelMask(uint8_t* src, uint8_t* dst, int width, int height, int cx, int cy, int radius) 
{
    #define BYTEPERPIXEL 2 
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * BYTEPERPIXEL;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if ( x >= width ||  y >= height) {
        return;
    }

    int yuyv_index = (y * width ) + x * BYTEPERPIXEL;

    int dx = (x << 1 ) - cx;
    int dy = y - cy;
    int distance = dx*dx + dy*dy;


    if (distance <= radius*radius) { //distance <= radius*radius
        dst[yuyv_index] =   threadIdx.x;
        dst[yuyv_index + 1] = threadIdx.x;
        // dst[yuyv_index+2] =   0xFF;
        // dst[yuyv_index+3] = 0x80;
    }


     
}

__global__ void applyCircularKernelMaskRGBA(uint8_t* src, uint8_t* dst, int width, int height, int cx, int cy, int radius) 
{
    int x = (threadIdx.x + blockIdx.x * blockDim.x);
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if ( x >= width ||  y >= height) {
        return;
    }

    int rgba = (y * width ) + x;

    int dx = x - cx;
    int dy = y - cy;
    int distance = sqrtf(dx*dx + dy*dy);


    if (distance > radius) { //distance <= radius*radius
        dst[rgba] = 1;
        dst[rgba + 1] = 1;
        dst[rgba + 2] = 1;
        dst[rgba + 3] = 255;
        
    }    
}

__global__ void circular_mask_kernel(unsigned char* input_yuyv, unsigned char* output_mask, int width, int height, int center_x, int center_y, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + 2 * x;

    if (x < width && y < height) {
        int dist_x = x - center_x;
        int dist_y = y - center_y;
        int dist_squared = dist_x * dist_x + dist_y * dist_y;

        if (dist_squared <= radius * radius) {
            output_mask[idx] = 255;
        }
        //  else {
        //     output_mask[idx] = 0;
        // }
    }
}

// __global__ void kernelYUV444(unsigned char *input_yuyv, unsigned char *output_mask, int width, int height, int center_x, int center_y, int radius)
// {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//     if (x > width || y > height)
//     {
//         return;
//     }
//     int idx = y * width + x;

//     if( x < (width/2))
//     {
//         // auto pointToY = (uint8_t *) output_mask + idx;
//         // *pointToY = 0; 
//         output_mask[idx] =0;       
//     }
// }

__global__ void applyCircularKernelMaskYUV444(unsigned char *input_yuyv, unsigned char *output_mask, int width, int height, int center_x, int center_y, int radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > width || y > height)
    {
        return;
    }
    int idx = y * width + x;
    int offset = width * height;
    float dist_x = x - center_x;
    float dist_y = y - center_y;
    
    float distance = sqrtf(dist_x * dist_x + dist_y * dist_y);

    
    if (distance > radius)
    {
        // auto pointToY = (uint8_t *) output_mask + idx;
        // *pointToY = 0;
        input_yuyv[idx] = 0;
        input_yuyv[idx + offset] = 0;
        input_yuyv[idx + 2 * offset] = 0;

        // output_mask[idx + offset] = 128;
        // output_mask[idx + 2*offset] = 0;


        // output_mask[idx + width * height] = 0;
        // output_mask[idx + 2 * width * height] = 0; 
    }
    //  else {
    //     output_mask[idx] = 0;
    // }
}

__global__ void square_indicator_kernel(unsigned char *input_yuyv, unsigned char *output_mask, int width, int height, int stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
    {
        return;
    }
    int idx = y * stride + x;
    int ushift = (stride * height);
    if( y < 20)
    {
        input_yuyv[idx] = 0.0f;
        input_yuyv[idx + ushift] = 128.0f;
        input_yuyv[idx + 2 * ushift] = 220.0f;
    }
}
// __global__ void octagon_mask_kernel(unsigned char *src, unsigned char *dst, int width, int height, int stride, int cx, int cy, int radius) {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;

//     if (x >= width || y >= height) {
//         return;
//     }

//     // Calculate the distance from the center of the octagon
//     int dx = abs(x - cx);
//     int dy = abs(y - cy);
//     float dist = sqrtf((float)(dx*dx + dy*dy));

//     // Check if the current pixel is inside the octagon
//     if (dist <= radius ||
//         (dx <= radius && dy <= 2 * radius && dist >= radius * sqrtf(2)) ||
//         (dx <= 2 * radius && dy <= radius && dist >= radius * sqrtf(2))) {
//         // Calculate the index of the current pixel in the YUV444 frame
//         int index = y * stride + x;

//         // Apply the mask to the Y component
//         dst[index] = 0;
//     }
// }

// __global__ void octagon_mask_kernel(unsigned char *src, unsigned char *dst, int width, int height, int stride, int cx, int cy, int radius) {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;

//     if (x >= width || y >= height) {
//         return;
//     }
//     // Calculate the distance and angle from the current pixel to the center of the octagon
//     int dx = x - cx;
//     int dy = y - cy;
//     float dist = sqrtf((float)(dx*dx + dy*dy));
//     float angle = atan2f(dy, dx);
//     int index = y * stride + x;
//     // Check if the current pixel is inside the octagon
//     bool inside = false;
//     for (int i = 0; i < 8; i++) {
//         float a = i * M_PI / 4 + M_PI / 8;
//         float b = a + M_PI / 4;
//         if (angle >= a && angle < b) {
//             inside = true;
//             break;
//         }
//     }
//     if (inside && dist <= radius) {
//         // Calculate the index of the current pixel in the YUV444 frame
        

//         // Apply the mask to the Y component
//         dst[index] = 0;
//     }
// }

__global__ void octagon_mask_kernel(unsigned char *src, unsigned char *dst, int width, int height, int stride, int cx, int cy, int r) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) {
        return;
    }

    int x = i - cx;
    int y = j - cy;

    if (x * x + y * y > r * r) {
        dst[j * stride + i] = 0.0f;
        return;
    }

    int octagon_x = r * 92682 / 131072; // 0.707106781 * 2^16
    int octagon_y = r * 92682 / 131072; // 0.707106781 * 2^16

    if (x >= 0 && y >= 0 && x + y > r + octagon_x) {
        dst[j * stride + i] = 0.0f;
        return;
    }

    if (x >= 0 && y < 0 && x - y > r + octagon_x) {
        dst[j * stride + i] = 0.0f;
        return;
    }

    if (x < 0 && y < 0 && x + y < -r - octagon_x) {
        dst[j * stride + i] = 0.0f;
        return;
    }

    if (x < 0 && y >= 0 && x - y < -r - octagon_x) {
        dst[j * stride + i] = 0.0f;
        return;
    }

    // if (x > octagon_x && y > octagon_y) {
    //     // output[j * stride + i] = 0.0f;
    //     return;
    // }

    // if (x > octagon_x && y < -octagon_y) {
    //     // output[j * stride + i] = 0.0f;
    //     return;
    // }

    // if (x < -octagon_x && y < -octagon_y) {
    //     // output[j * stride + i] = 0.0f;
    //     return;
    // }

    // if (x < -octagon_x && y > octagon_y) {
    //     // output[j * stride + i] = 0.0f;
    //     return;
    // }

    // dst[j * stride + i] = 0.0f;
    // int x = blockIdx.x * blockDim.x + threadIdx.x;
    // int y = blockIdx.y * blockDim.y + threadIdx.y;

    // if (x >= width || y >= height) {
    //     return;
    // }
    // // Calculate the distance and angle from the current pixel to the center of the octagon
    // int dx = x - cx;
    // int dy = y - cy;
    // if (dx * dx + dy * dy > radius * radius) {
    //     dst[y * stride + x] = 0.0f;
    //     return;
    // }

    // if (dx >= 0 && dy >= 0 && dx >= dy && dx <= radius && dy <= radius - dx) {
    //     // dst[y * width + x] = 0.0f;
    //     return;
    // }

    // if (dx < 0 && dy < 0 && -dx >= -dy && -dx <= radius && -dy <= radius + dx) {
    //     // dst[y * width + x] = 0.0f;
    //     return;
    // }

    // if (dx >= 0 && dy < 0 && dx >= -dy && dx <= radius && -dy <= radius - dx) {
    // //    dst[y * width + x] = 0.0f;
    //     return;
    // }

    // if (dx < 0 && dy >= 0 && -dx >= dy && -dx <= radius && dy <= radius + dx) {
    //     // dst[y * width + x] = 0.0f;
    //     return;
    // }
    // dst[y * stride + x] = 0.0f;
}

void applyCircularMask(uint8_t* src, uint8_t* dst, int width, int height, int cx, int cy, int radius, cudaStream_t stream)
{
	// dim3 block(32, 32);
	// dim3 grid((width + block.x - 1) / 2 * block.x, (height + block.y - 1) / block.y);
    // applyCircularKernelMask <<< grid, block, 0, stream>>> (src, dst, width, height, 250, 140, 30);

    // dim3 dim_block(32, 32);
    dim3 dim_block(32, 32);
    dim3 dim_grid((width + dim_block.x - 1) /  dim_block.x, (height + dim_block.y - 1) / dim_block.y);
    applyCircularKernelMask <<< dim_grid, dim_block, 0, stream>>> (src, dst, width, height, 960, 480, 100);
    // circular_mask_kernel<<<dim_grid, dim_block>>>(src, dst, width, height, 200, 200, 40);

}


void applyCircularMaskRGBA(uint8_t* src, uint8_t* dst, int width, int height, int cx, int cy, int radius, cudaStream_t stream)
{
    dim3 dim_block(32, 32);
    dim3 dim_grid((width + dim_block.x - 1) /  dim_block.x, (height + dim_block.y - 1) / dim_block.y);
    applyCircularKernelMaskRGBA <<< dim_grid, dim_block, 0, stream>>> (src, dst, width, height, 400, 400, 100);
}


void applyCircularMaskYUV444(unsigned char* src, unsigned char* dst, int width, int height, int cx, int cy, int radius, cudaStream_t stream)
{
    dim3 dim_block(32, 32);
    dim3 dim_grid((width + dim_block.x - 1) /  dim_block.x, (height + dim_block.y - 1) / dim_block.y);
    applyCircularKernelMaskYUV444 <<< dim_grid, dim_block, 0, stream>>> (src, dst, width, height, 320, 240, 150);
}

void applyDiamondMaskYUV444(unsigned char *src, unsigned char *dst, int width, int height, int stride, int cx, int cy, int radius, cudaStream_t stream)
{
    dim3 dim_block(16, 16);
    dim3 dim_grid((width + dim_block.x - 1) /  dim_block.x, (height + dim_block.y - 1) / dim_block.y);
    octagon_mask_kernel <<< dim_grid, dim_block, 0, stream>>> (src, dst, width, height, stride, 600, 500, 400);
}

void addKernelIndicatorSquareMask(unsigned char *src, unsigned char *dst, int width, int height, int stride, cudaStream_t stream)
{
    dim3 dim_block(16, 16);
    dim3 dim_grid((width + dim_block.x - 1) /  dim_block.x, (height + dim_block.y - 1) / dim_block.y);
    square_indicator_kernel <<< dim_grid, dim_block, 0, stream>>> (src, dst, width, height, stride);
}