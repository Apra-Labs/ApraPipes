#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "MaskKernel.h"

// here cx and cy are center coordinates , radius is radius of circle
__global__ void applyCircularKernelMask(uint8_t *src, uint8_t *dst, int width, int height, int cx, int cy, int radius)
{
#define BYTEPERPIXEL 2
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * BYTEPERPIXEL;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height)
    {
        return;
    }

    int yuyv_index = (y * width) + x * BYTEPERPIXEL;

    int dx = (x << 1) - cx;
    int dy = y - cy;
    int distance = dx * dx + dy * dy;

    if (distance <= radius * radius)
    { // distance <= radius*radius
        dst[yuyv_index] = threadIdx.x;
        dst[yuyv_index + 1] = threadIdx.x;
        // dst[yuyv_index+2] =   0xFF;
        // dst[yuyv_index+3] = 0x80;
    }
}

__global__ void applyCircularMaskRGBANew(unsigned char *ptr, int width, int height, int centerX, int centerY, int radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
    {
        return;
    }

    if (x < width && y < height)
    {
        int index = (y * width + 4 * x); // we are mulltiplying x by 4 because width which is stride, already multiplied by 4 and as we are dealing with rgba we multiplied with 4
        int dx = x - centerX;
        int dy = y - centerY;
        int distance = (dx * dx + dy * dy);

        if (distance > radius * radius)
        {
            ptr[index + 0] = 0;
            ptr[index + 1] = 0;
            ptr[index + 2] = 0;
            ptr[index + 3] = 0;
        }
    }
}

__global__ void applySquareRotationIndicatorKernel(unsigned char *ptr, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
    {
        return;
    }


    int index = (y * width + 4 * x); // we are mulltiplying x by 4 because width which is stride, already multiplied by 4 and as we are dealing with rgba we multiplied with 4
    if(y <= 5)
    {
        ptr[index + 0] = 0;
        ptr[index + 1] = 215;
        ptr[index + 2] = 0;
        // ptr[index + 3] = 160;
    }
}

__global__ void generateOctagonKernelRGBA(unsigned char* inPtr, int width, int height, int centerX, int centerY, int radius) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int dx = x - centerX;
    int dy = y - centerY;
    float distance = sqrtf(dx * dx + dy * dy);

    // Check if the point lies on the octagon
    float octagonRadius = radius * (1 + sqrtf(2));
    float minDistance = octagonRadius - radius;
    float maxDistance = octagonRadius + radius;

    if (distance >= minDistance && distance <= maxDistance) {
        // Set the RGB values for the octagon
        int offset = (y * width + 4* x) ;
        inPtr[offset] = 0;
        inPtr[offset + 1] = 0;
        inPtr[offset + 2] = 0;
        inPtr[offset + 3] = 0;
    }
}

__global__ void applyCircularKernelMaskRGBA(uint8_t *src, uint8_t *dst, int width, int height, int cx, int cy, int radius)
{
    int x = (threadIdx.x + blockIdx.x * blockDim.x);
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height)
    {
        return;
    }

    int rgba = (y * width) + x;

    int dx = x - cx;
    int dy = y - cy;
    int distance = sqrtf(dx * dx + dy * dy);

    if (distance > radius)
    { // distance <= radius*radius
        dst[rgba] = 1;
        dst[rgba + 1] = 1;
        dst[rgba + 2] = 1;
        dst[rgba + 3] = 255;
    }
}

__global__ void circular_mask_kernel(unsigned char *input_yuyv, unsigned char *output_mask, int width, int height, int center_x, int center_y, int radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + 2 * x;

    if (x < width && y < height)
    {
        int dist_x = x - center_x;
        int dist_y = y - center_y;
        int dist_squared = dist_x * dist_x + dist_y * dist_y;

        if (dist_squared <= radius * radius)
        {
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
    int offset = width * 512;
    float dist_x = x - center_x;
    float dist_y = y - center_y;

    float distance = sqrtf(dist_x * dist_x + dist_y * dist_y);

    if (distance > radius)
    {
        // auto pointToY = (uint8_t *) output_mask + idx;
        // *pointToY = 0;
        // output_mask[idx] = 0;  /// yashrajWorking
        input_yuyv[idx] = 0;
        input_yuyv[idx + offset] = 128;
        input_yuyv[idx + 2 * offset] = 128;

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
    if (y < 2)
    {
        // input_yuyv[idx] = 0.0f;
        input_yuyv[idx + (stride * 512)] = 128.0f;
        input_yuyv[idx + (stride * 512) * 2] = 128.0f;
        // input_yuyv[idx + (stride * 512) * 2] = 220.0f;
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

__global__ void octagon_mask_kernel(unsigned char *src, unsigned char *dst, int width, int height, int stride, int cx, int cy, int r)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > width || j > height)
    {
        return;
    }

    int x = i - cx;
    int y = j - cy;

    if (x * x + y * y > r * r)
    {
        dst[j * stride + i] = 0.0f;
        return;
    }

    int octagon_x = r * 92682 / 131072; // 0.707106781 * 2^16
    int octagon_y = r * 92682 / 131072; // 0.707106781 * 2^16

    if (x >= 0 && y >= 0 && x + y > r + octagon_x)
    {
        dst[j * stride + i] = 0.0f;
        return;
    }

    if (x >= 0 && y < 0 && x - y > r + octagon_x)
    {
        dst[j * stride + i] = 0.0f;
        return;
    }

    if (x < 0 && y < 0 && x + y < -r - octagon_x)
    {
        dst[j * stride + i] = 0.0f;
        return;
    }

    if (x < 0 && y >= 0 && x - y < -r - octagon_x)
    {
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

void applyCircularMask(uint8_t *src, uint8_t *dst, int width, int height, int cx, int cy, int radius, cudaStream_t stream)
{
    // dim3 block(32, 32);
    // dim3 grid((width + block.x - 1) / 2 * block.x, (height + block.y - 1) / block.y);
    // applyCircularKernelMask <<< grid, block, 0, stream>>> (src, dst, width, height, 250, 140, 30);

    // dim3 dim_block(32, 32);
    dim3 dim_block(32, 32);
    dim3 dim_grid((width + dim_block.x - 1) / dim_block.x, (height + dim_block.y - 1) / dim_block.y);
    applyCircularKernelMask<<<dim_grid, dim_block, 0, stream>>>(src, dst, width, height, 960, 480, 100);
    // circular_mask_kernel<<<dim_grid, dim_block>>>(src, dst, width, height, 200, 200, 40);
}

void applyCircularMaskRGBA(uint8_t *src, uint8_t *dst, int width, int height, int cx, int cy, int radius, cudaStream_t stream)
{
    dim3 dim_block(32, 32);
    dim3 dim_grid((width + dim_block.x - 1) / dim_block.x, (height + dim_block.y - 1) / dim_block.y);
    applyCircularKernelMaskRGBA<<<dim_grid, dim_block, 0, stream>>>(src, dst, width, height, 400, 400, 100);
}

__device__ bool is_point_inside_triangle(float x, float y,
                                         float x1, float y1,
                                         float x2, float y2,
                                         float x3, float y3)
{

    float alpha = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3));
    float beta = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3));
    float gamma = 1.0f - alpha - beta;
    if (alpha >= 0 && beta >= 0 && gamma >= 0)
    {
        return true;
    }
    return false;
    // float A = 0.5f * (-y2 * x3 + y1 * (-x2 + x3) + x1 * (y2 - y3) + x2 * y3);
    // float sign = A < 0 ? -1.0f : 1.0f;
    // float s = (y1 * x3 - x1 * y3 + (y3 - y1) * px + (x1 - x3) * py) * sign;
    // float t = (x1 * y2 - y1 * x2 + (y1 - y2) * px + (x2 - x1) * py) * sign;
    // return s > 0 && t > 0 && (s + t) < 2 * A * sign;
}

__global__ void detect_triangles(uint8_t *yuv, uint8_t *dst, int width, int height,
                                 float *x1, float *y1,
                                 float *x2, float *y2,
                                 float *x3, float *y3,
                                 float *x4, float *y4)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
    {
        return;
    }

    int offset = y * width * 3 + x * 3;
    float px = (float)x + 0.5f;
    float py = (float)y + 0.5f;

    // Check if pixel is inside any of the four triangles
    if (!is_point_inside_triangle(px, py, x1[0], y1[0], x1[1], y1[1], x1[2], y1[2]) &&
        !is_point_inside_triangle(px, py, x2[0], y2[0], x2[1], y2[1], x2[2], y2[2]) &&
        !is_point_inside_triangle(px, py, x3[0], y3[0], x3[1], y3[1], x3[2], y3[2]) &&
        !is_point_inside_triangle(px, py, x4[0], y4[0], x4[1], y4[1], x4[2], y4[2]))
    {
        // Pixel is not inside any triangle, set to black
        yuv[offset] = 0;       // Y component
        yuv[offset + 1] = 128; // U component
        yuv[offset + 2] = 128; // V component
    }

    else
    {
        yuv[offset] = 100;     // Y component
        yuv[offset + 1] = 128; // U component
        yuv[offset + 2] = 128; // V component
    }
}

//

// __global__ void detect_triangles2(unsigned char *yuv, unsigned char *dst, int width, int height,
//                                  float *x1, float *y1)
// {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//     if (x > width || y > height)
//     {
//         return;
//     }
//     int idx = y * width + x;
//     int offset = y * width * 3 + x * 3;
//     float px = (float)x + 0.5f;
//     float py = (float)y + 0.5f;
//     yuv[idx] = 0;
//     // Check if pixel is inside any of the four triangles
//     if (!is_point_inside_triangle(x, y, x1[0], y1[0], x1[1], y1[1], x1[2], y1[2]))
//     {
//         // Pixel is not inside any triangle, set to black
//         yuv[idx] = 0;       // Y component
//         // yuv[idx + 1] = 128; // U component
//         // yuv[offset + 2] = 128; // V component
//     }

//     else
//     {
//         yuv[idx] = 100;       // Y component
//         // yuv[offset + 1] = 128; // U component
//         // yuv[offset + 2] = 128; // V component

//     }
// }

// __global__ void detect_triangles2(unsigned char *yuv, unsigned char *dst, int width, int height,
//                                   float *x12, float *y12)
// {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//     if (x > width || y > height)
//     {
//         return;
//     }
//     int idx = y * width + x;
//     float x1, x2, x3, y1, y2, y3;
//     x1 = 0.0f;
//     y1 = 0.0f;
//     x2 = 100.0f;
//     y2 = 0.0f;
//     x3 = 0.0f;
//     y3 = 100.0f;
//     float alpha = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3));
//     float beta = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3));
//     float gamma = 1.0f - alpha - beta;

//     float x21, x22, x23, y21, y22, y23;
//     // x21 = (float)width;
//     // y21 = 0.0f;
//     // x22 = (float)width - 150.0f;
//     // y22 = 0.0f;
//     // x23 = (float)width;
//     // y23 = 150.0f;
//     x21 = 400.0f;
//     y21 = 0.0f;
//     x22 = 300.0f;
//     y22 = 0.0f;
//     x23 = 400.0f;
//     y23 = 100.0f;
//     float alpha1 = ((y22 - y23) * (x - x23) + (x23 - x22) * (y - y23)) / ((y22 - y23) * (x21 - x23) + (x23 - x22) * (y21 - y23));
//     float beta1 = ((y23 - y21) * (x - x23) + (x21 - x23) * (y - y23)) / ((y22 - y23) * (x21 - x23) + (x23 - x22) * (y21 - y23));
//     float gamma1 = 1.0f - alpha1 - beta1;

//     float x31, x32, x33, y31, y32, y33;
//     x31 = 0.0f;
//     y31 = 400.0f;
//     x32 = 0.0f;
//     y32 = 300.0f;
//     x33 = 100.0f;
//     y33 = 400.0f;
//     float alpha2 = ((y32 - y33) * (x - x33) + (x33 - x32) * (y - y33)) / ((y32 - y33) * (x31 - x33) + (x33 - x32) * (y31 - y33));
//     float beta2 = ((y33 - y31) * (x - x33) + (x31 - x33) * (y - y33)) / ((y32 - y33) * (x31 - x33) + (x33 - x32) * (y31 - y33));
//     float gamma2 = 1.0f - alpha2- beta2;

//     float x41, x42, x43, y41, y42, y43;
//     x41 = 400.0f;
//     y41 = 400.0f;
//     x42 = 400.0f;
//     y42 = 300.0f;
//     x43 = 300.0f;
//     y43 = 400.0f;
//     float alpha3 = ((y42 - y43) * (x - x43) + (x43 - x42) * (y - y43)) / ((y42 - y43) * (x41 - x43) + (x43 - x42) * (y41 - y43));
//     float beta3 = ((y43 - y41) * (x - x43) + (x41 - x43) * (y - y43)) / ((y42 - y43) * (x41 - x43) + (x43 - x42) * (y41 - y43));
//     float gamma3 = 1.0f - alpha3 - beta3;


//     if ((alpha >= 0 && beta >= 0 && gamma >= 0) ||( alpha1 >= 0 && beta1 >= 0 && gamma1 >= 0) || (alpha2 >= 0 && beta2 >= 0 && gamma2 >= 0) || (alpha3 >= 0 && beta3 >= 0 && gamma3) >= 0)
//     {
//         yuv[idx] = 0;
//     }
//     else
//     {
//     }

//     // Check if pixel is inside any of the four triangles
// }

__global__ void detect_triangles2(unsigned char *yuv, unsigned char *dst, int width, int height,
                                  float *x12, float *y12)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > width || y > height)
    {
        return;
    }
    int idx = y * width + x;
    int offset = width * 512;
    float x1, x2, x3, y1, y2, y3;
    x1 = 0.0f;
    y1 = 0.0f;
    x2 = 100.0f;//130.0f;
    y2 = 0.0f;
    x3 = 0.0f;
    y3 = 100.0f;//130.0f;
    float alpha = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3));
    float beta = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3));
    float gamma = 1.0f - alpha - beta;


    if (alpha >= 0 && beta >= 0 && gamma >= 0)
    {
        yuv[idx] = 0;
        yuv[idx + offset] = 128;
        yuv[idx + 2 * offset] = 128;
    }
    else
    {
    }

    float x41, x42, x43, y41, y42, y43;
    x41 = 300.0f;
    y41 = 400.0f;
    x42 = 400.0f;
    y42 = 400.0f;
    x43 = 400.0f;
    y43 = 300.0f;
    float alpha3 = ((y42 - y43) * (x - x43) + (x43 - x42) * (y - y43)) / ((y42 - y43) * (x41 - x43) + (x43 - x42) * (y41 - y43));
    float beta3 = ((y43 - y41) * (x - x43) + (x41 - x43) * (y - y43)) / ((y42 - y43) * (x41 - x43) + (x43 - x42) * (y41 - y43));
    float gamma3 = 1.0f - alpha3 - beta3;
    if (alpha3 >= 0 && beta3 >= 0 && gamma3 >= 0)
    {
        yuv[idx] = 0;
         yuv[idx + offset] = 128;
        yuv[idx + 2 * offset] = 128;
    }
    else
    {
    }

    float x31, x32, x33, y31, y32, y33;
    x31 = 300.0f;//270.0f;
    y31 = 0.0f;
    x32 = 400.0f;
    y32 = 0.0f;
    x33 = 400.0f;
    y33 = 100.0f;//130.0f;
    float alpha2 = ((y32 - y33) * (x - x33) + (x33 - x32) * (y - y33)) / ((y32 - y33) * (x31 - x33) + (x33 - x32) * (y31 - y33));
    float beta2 = ((y33 - y31) * (x - x33) + (x31 - x33) * (y - y33)) / ((y32 - y33) * (x31 - x33) + (x33 - x32) * (y31 - y33));
    float gamma2 = 1.0f - alpha2- beta2;

    if (alpha2 >= 0 && beta2 >= 0 && gamma2 >= 0)
    {
        yuv[idx] = 0;
        yuv[idx + offset] = 128;
        yuv[idx + 2 * offset] = 128;
    }
    else
    {
    }

    float x21, x22, x23, y21, y22, y23;
    x21 = 0.0f;
    y21 = 300.0f;
    x22 = 0.0f;
    y22 = 400.0f;
    x23 = 100.0f;
    y23 = 400.0f;
    float alpha1 = ((y22 - y23) * (x - x23) + (x23 - x22) * (y - y23)) / ((y22 - y23) * (x21 - x23) + (x23 - x22) * (y21 - y23));
    float beta1 = ((y23 - y21) * (x - x23) + (x21 - x23) * (y - y23)) / ((y22 - y23) * (x21 - x23) + (x23 - x22) * (y21 - y23));
    float gamma1 = 1.0f - alpha1 - beta1;
    
    if (alpha1 >= 0 && beta1 >= 0 && gamma1 >= 0)
    {
        yuv[idx] = 0;
         yuv[idx + offset] = 128;
        yuv[idx + 2 * offset] = 128;

    }
    else
    {
    }
    // Check if pixel is inside any of the four triangles
}

void applyCircularMaskYUV444(unsigned char *src, unsigned char *dst, int width, int height, int cx, int cy, int radius, cudaStream_t stream)
{
    dim3 dim_block(32, 32);
    dim3 dim_grid((width + dim_block.x - 1) / dim_block.x, (height + dim_block.y - 1) / dim_block.y);
    applyCircularKernelMaskYUV444<<<dim_grid, dim_block, 0, stream>>>(src, dst, width, height, cx, cy, radius);
}

void applyCircularMaskForRGBANew(unsigned char *src, unsigned char *dst, int width, int height, int cx, int cy, int radius, cudaStream_t stream)
{
    dim3 dim_block(32, 32);
    dim3 dim_grid((1000 + dim_block.x - 1) / dim_block.x, (height + dim_block.y - 1) / dim_block.y);
    applyCircularMaskRGBANew<<<dim_grid, dim_block, 0, stream>>>(src, width, height, 500, 500, 480);
}

void applySquareRotationIndicator(unsigned char *src, unsigned char *dst, int width, int height, cudaStream_t stream)
{
    dim3 dim_block(32, 32);
    dim3 dim_grid((1000 + dim_block.x - 1) / dim_block.x, (height + dim_block.y - 1) / dim_block.y);
    applySquareRotationIndicatorKernel<<<dim_grid, dim_block, 0, stream>>>(src, width, height);
}

void generateRGBAOctagonalKernel(unsigned char *src, unsigned char *dst, int width, int height, int cx, int cy, int radius, cudaStream_t stream)
{
    dim3 dim_block(32, 32);
    dim3 dim_grid((1000 + dim_block.x - 1) / dim_block.x, (height + dim_block.y - 1) / dim_block.y);
    generateOctagonKernelRGBA<<<dim_grid, dim_block, 0, stream>>>(src, width, height, cx, cy, radius);
}

void applyDiamondMaskYUV444(unsigned char *src, unsigned char *dst, int width, int height, int stride, int cx, int cy, int radius, cudaStream_t stream)
{
    dim3 dim_block(16, 16);
    dim3 dim_grid((width + dim_block.x - 1) / dim_block.x, (height + dim_block.y - 1) / dim_block.y);
    octagon_mask_kernel<<<dim_grid, dim_block, 0, stream>>>(src, dst, width, height, stride, 600, 500, 400);
}

void addKernelIndicatorSquareMask(unsigned char *src, unsigned char *dst, int width, int height, int stride, cudaStream_t stream)
{
    dim3 dim_block(16, 16);
    dim3 dim_grid((width + dim_block.x - 1) / dim_block.x, (height + dim_block.y - 1) / dim_block.y);
    square_indicator_kernel<<<dim_grid, dim_block, 0, stream>>>(src, dst, width, height, stride);
}

// void applyOctagonalMask(uint8_t *src, uint8_t *dst, int width, int height, float *triangle1X, float *triangle1Y, float *triangle2X, float *triangle2Y, float *triangle3X, float *triangle3Y, float *triangle4X, float *triangle4Y, cudaStream_t stream)
// {
//     dim3 dim_block(32, 32);
//     dim3 dim_grid((width + dim_block.x - 1) / dim_block.x, (height + dim_block.y - 1) / dim_block.y);
//     detect_triangles<<<dim_grid, dim_block, 0, stream>>>(src, dst, width, height, triangle1X, triangle1Y, triangle2X, triangle2Y, triangle3X, triangle3Y, triangle4X, triangle4Y);
// }

void applyOctagonalMask(unsigned char *src, unsigned char *dst, int width, int height, float *triangle1X, float *triangle1Y, float *triangle2X, float *triangle2Y, float *triangle3X, float *triangle3Y, float *triangle4X, float *triangle4Y, cudaStream_t stream)
{
    dim3 dim_block(32, 32);
    dim3 dim_grid((width + dim_block.x - 1) / dim_block.x, (height + dim_block.y - 1) / dim_block.y);
    detect_triangles2<<<dim_grid, dim_block, 0, stream>>>(src, dst, width, height, triangle1X, triangle1Y);
}