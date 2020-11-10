#include "OverlayKernel.h"

#define CLAMP_255(x) x < 0 ? 0 : (x > 255 ? 255 : x)
#define CLAMP_INT8(x) x < -128 ? -128 : (x > 127 ? 127 : x)

/* __fsub_rn(overlay_pixel, 16) was added for libstreamer because in libstreamer setRGBAData for y componenet 16 was added so dont use this function for any other purpose */

#define OVERLAY_Y_ALPHA(src_pixel, overlay_pixel, dst_pixel, alpha)                 \
do                                                                                  \
{                                                                                   \
    Npp32f temp_ = __fmul_rn(overlay_pixel, alpha);                                 \
    temp_ = __fadd_rn(src_pixel, temp_);                                            \
    dst_pixel = CLAMP_255(temp_);                                                   \
} while(0)

#define UV_OVERLAY_ALPHA(src_pixel, overlay_pixel, dst_pixel, alpha)                \
do                                                                                  \
{                                                                                   \
    Npp32f temp = __fmul_rn(__fsub_rn(overlay_pixel, 128), alpha);                  \
    temp = __fadd_rn(__fsub_rn(src_pixel, 128), temp);                              \
    dst_pixel = 128 + (CLAMP_INT8(temp));                                           \
} while(0)

#define OVERLAY_Y(src_pixel, overlay_pixel, dst_pixel, alpha)                       \
do                                                                                  \
{                                                                                   \
	Npp32f temp = __fsub_rn(overlay_pixel, 16);                                  	\
	if(alpha != -1)																	\
	{																				\
		OVERLAY_Y_ALPHA(src_pixel, temp, dst_pixel, alpha);							\
		break;																		\
	}																				\
	if(temp == 0)																	\
	{																				\
		dst_pixel = src_pixel;														\
	}																				\
	else 																			\
	{																				\
		dst_pixel = CLAMP_255(temp);                                                \
	}																				\
} while(0)

#define UV_OVERLAY(src_pixel, overlay_pixel, dst_pixel, alpha)                      \
do                                                                                  \
{                                                                                   \
	if(alpha != -1)																	\
	{																				\
		UV_OVERLAY_ALPHA(src_pixel, overlay_pixel, dst_pixel, alpha);				\
		break;																		\
	}																				\
    dst_pixel = overlay_pixel == 128 ? src_pixel:overlay_pixel;						\
} while(0)

__global__ void yuvOverlayKernel(const uchar4* Y, const uchar4* U, const uchar4* V, const uchar4* overlay_y, const uchar4* overlay_u, const uchar4* overlay_v, uchar4* Yout, uchar4* Uout, uchar4* Vout, 
	float alpha, int step_y, int step_uv, int overlayStep_y, int overlayStep_uv, int width_y, int height_y, int width_uv, int height_uv)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width_y || y >= height_y)
	{
		return;
	}

	int offset = y * step_y + x;
	int overlayOffset = y * overlayStep_y + x;
	OVERLAY_Y(Y[offset].x, overlay_y[overlayOffset].x, Yout[offset].x, alpha);
    OVERLAY_Y(Y[offset].y, overlay_y[overlayOffset].y, Yout[offset].y, alpha);
    OVERLAY_Y(Y[offset].z, overlay_y[overlayOffset].z, Yout[offset].z, alpha);
    OVERLAY_Y(Y[offset].w, overlay_y[overlayOffset].w, Yout[offset].w, alpha);

	
    if(x >= width_uv || y >= height_uv)
    {
        return;
    }
	offset = y * step_uv + x;
	overlayOffset = y * overlayStep_uv + x;

    UV_OVERLAY(U[offset].x, overlay_u[overlayOffset].x, Uout[offset].x, alpha);
    UV_OVERLAY(U[offset].y, overlay_u[overlayOffset].y, Uout[offset].y, alpha);
    UV_OVERLAY(U[offset].z, overlay_u[overlayOffset].z, Uout[offset].z, alpha);
    UV_OVERLAY(U[offset].w, overlay_u[overlayOffset].w, Uout[offset].w, alpha);

    UV_OVERLAY(V[offset].x, overlay_v[overlayOffset].x, Vout[offset].x, alpha);
    UV_OVERLAY(V[offset].y, overlay_v[overlayOffset].y, Vout[offset].y, alpha);
    UV_OVERLAY(V[offset].z, overlay_v[overlayOffset].z, Vout[offset].z, alpha);
    UV_OVERLAY(V[offset].w, overlay_v[overlayOffset].w, Vout[offset].w, alpha);
}

void launchYUVOverlayKernel(const Npp8u* src[3], const Npp8u* overlay[3], Npp8u* dst[3], Npp32f alpha, int srcStep[2], int overlayStep[2], NppiSize size, cudaStream_t stream)
{
	auto mod = size.width % 8;
	if (mod != 0)
	{
		// we would just process few extra pixels - step is anyway bigger than width and is aligned by 512/256
		size.width += 8 - mod;
	}

    auto width = size.width >> 2;
	int srcStep_y = srcStep[0] >> 2;
	int srcStep_uv = srcStep[1] >> 2;
	int overlayStep_y = overlayStep[0] >> 2;
	int overlayStep_uv = overlayStep[1] >> 2;
	dim3 block(32, 32);
	dim3 grid((width + block.x - 1) / block.x, (size.height + block.y - 1) / block.y);
	yuvOverlayKernel << <grid, block, 0, stream >> > (reinterpret_cast<const uchar4*>(src[0]),
		reinterpret_cast<const uchar4*>(src[1]),
		reinterpret_cast<const uchar4*>(src[2]),
		reinterpret_cast<const uchar4*>(overlay[0]),
		reinterpret_cast<const uchar4*>(overlay[1]),
		reinterpret_cast<const uchar4*>(overlay[2]),
		reinterpret_cast<uchar4*>(dst[0]),
		reinterpret_cast<uchar4*>(dst[1]),
		reinterpret_cast<uchar4*>(dst[2]),
		alpha, srcStep_y, srcStep_uv, overlayStep_y, overlayStep_uv, width, size.height, width >> 1, size.height >> 1);
}